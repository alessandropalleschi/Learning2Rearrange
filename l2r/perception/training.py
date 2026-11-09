"""Collect data from the simulator, train the segmentation network, evaluate it (also on the observations it produces).

    python train_perception.py collect --scenes 400 --out data/perception
    python train_perception.py train   --data data/perception --minutes 8 --out assets/perception/segnet.pth
    python train_perception.py evaluate --weights assets/perception/segnet.pth --scenes 20
"""
import argparse
import glob
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from l2r.config import SimParams
from l2r.perception.agnostic import AgnosticPerception, agnostic_targets
from l2r.perception.base import GroundTruthPerception
from l2r.perception.network import INPUT_SCALE, NetworkPerception
from l2r.perception.unet import SegmentationNet
from l2r.sim.constants import OBJECT_CATALOGUE


def type_map(segm, names):
    """Object-type label per pixel (0 = background, i = OBJECT_CATALOGUE[i - 1]) from the ground-truth id image."""
    types = np.zeros(segm.shape, np.uint8)
    for uid, name in names.items():
        types[segm == uid] = OBJECT_CATALOGUE.index(name.split(' ')[1]) + 1
    return types


# ---------------------------------------------------------------------------------------------------- data
def collect(out_dir, scenes, seed=0, min_objects=1, max_objects=10, randomize=True, object_types=None):
    """Render `scenes` random arrangements (camera pose jitter and sensor noise on) and store them downsampled."""
    from l2r.sim.environment import Environment
    os.makedirs(out_dir, exist_ok=True)
    env = Environment(num_obj=[max_objects], params=SimParams(randomize=randomize, sensor_noise=randomize,
                                                                  object_types=tuple(object_types) if object_types else None))
    rng = np.random.default_rng(seed)
    for i in range(scenes):
        np.random.seed(seed + i)
        random.seed(seed + i)
        env.num_obj = [int(rng.integers(min_objects, max_objects + 1))]
        while not env.reset():
            pass
        color, depth, segm, *_ = env.camera.render()
        s = INPUT_SCALE
        np.savez_compressed(os.path.join(out_dir, '%05d.npz' % i), color=color[::s, ::s], depth=depth[::s, ::s].astype(np.float16),
                            types=type_map(segm, env.scene.names)[::s, ::s],
                            agnostic=agnostic_targets(segm[::s, ::s], env.scene.names))
        if i % 50 == 0:
            print('collected', i, '/', scenes, flush=True)


def load_dataset(data_dir, task='types'):
    files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    inputs, targets = [], []
    for f in files:
        d = np.load(f)
        x = np.concatenate([d['color'].astype(np.float32) / 255.0, (d['depth'].astype(np.float32) / 1.5)[..., None]], -1)
        inputs.append(x.transpose(2, 0, 1))
        targets.append(d['agnostic'] if task == 'agnostic' else d['types'])
    return torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(targets)).long()


def augment(x, y, generator):
    """Brightness / contrast jitter, depth noise, random horizontal flip (labels flip with the image)."""
    b = x.shape[0]
    gain = 1 + 0.25 * (torch.rand(b, 1, 1, 1, generator=generator) * 2 - 1)
    bias = 0.1 * (torch.rand(b, 1, 1, 1, generator=generator) * 2 - 1)
    x = x.clone()
    x[:, :3] = (x[:, :3] * gain + bias).clamp(0, 1)
    x[:, 3] += 0.002 * torch.randn(x[:, 3].shape, generator=generator)
    flip = torch.rand(b, generator=generator) < 0.5
    x[flip], y[flip] = x[flip].flip(-1), y[flip].flip(-1)
    return x, y


# ---------------------------------------------------------------------------------------------------- metrics
def confusion(pred, target, classes):
    index = target.reshape(-1) * classes + pred.reshape(-1)
    return torch.bincount(index, minlength=classes * classes).reshape(classes, classes)


def iou_from_confusion(c):
    inter = c.diag().float()
    union = c.sum(0) + c.sum(1) - c.diag()
    return inter / union.clamp(min=1)


@torch.no_grad()
def validate(model, x, y, classes):
    model.eval()
    c = torch.zeros(classes, classes, dtype=torch.long)
    for i in range(0, len(x), 16):
        c += confusion(model(x[i:i + 16]).argmax(1), y[i:i + 16], classes)
    return iou_from_confusion(c)


def train(data_dir, out, minutes=8.0, batch=8, lr=2e-3, seed=0, threads=4, task='types'):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    x, y = load_dataset(data_dir, task)
    num_types = 2 if task == 'agnostic' else len(OBJECT_CATALOGUE)  # agnostic: interior and boundary
    classes = 1 + num_types
    split = max(1, int(0.9 * len(x)))
    (xt, yt), (xv, yv) = (x[:split], y[:split]), (x[split:], y[split:])
    frequency = torch.bincount(yt.reshape(-1), minlength=classes).float()
    weights = (frequency.sum() / frequency.clamp(min=1)).sqrt()
    weights = weights / weights[0]
    model = SegmentationNet(num_types)
    print('%d training / %d validation images, %.2f M parameters, class weights %s'
          % (len(xt), len(xv), sum(p.numel() for p in model.parameters()) / 1e6, np.round(weights.numpy(), 1)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator, deadline, best, epoch = torch.Generator().manual_seed(seed), time.time() + 60 * minutes, -1, 0
    while time.time() < deadline:
        model.train()
        order = torch.randperm(len(xt), generator=generator)
        losses = []
        for i in range(0, len(order) - batch + 1, batch):
            idx = order[i:i + batch]
            xb, yb = augment(xt[idx], yt[idx], generator)
            loss = F.cross_entropy(model(xb), yb, weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if time.time() > deadline:
                break
        epoch += 1
        iou = validate(model, xv, yv, classes)
        miou = float(iou[1:].mean())  # object types only
        if miou > best:
            best = miou
            os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
            torch.save(model.state_dict(), out)
        print('epoch %2d loss %.3f  validation IoU: background %.3f, types %s, mean over types %.3f%s'
              % (epoch, np.mean(losses), iou[0], np.round(iou[1:].numpy(), 3), miou, '  (saved)' if miou == best else ''), flush=True)
    return best


# ---------------------------------------------------------------------------------------------------- evaluation
def match_instances(pred, gt, min_area=40, threshold=0.5):
    """Instance recall / precision: a GT object counts as found if a predicted instance covers it with IoU >= threshold."""
    gt_ids = [i for i in np.unique(gt) if i != 0 and (gt == i).sum() >= min_area]
    pred_ids = [i for i in np.unique(pred) if i != 0]
    found, matched = 0, set()
    for g in gt_ids:
        region = gt == g
        for p in np.unique(pred[region]):
            if p == 0:
                continue
            candidate = pred == p
            if (region & candidate).sum() / (region | candidate).sum() >= threshold:
                found += 1
                matched.add(p)
                break
    return found, len(gt_ids), len(matched), len(pred_ids)


def evaluate(weights, scenes=20, seed=1000, max_objects=8):
    """Compare ground-truth perception with the network on fresh simulator scenes: pixel IoU of the types, instance
    recall / precision, and how different the observation maps of the policy are."""
    from l2r.sim.environment import Environment
    classes = 1 + len(OBJECT_CATALOGUE)
    network = NetworkPerception(weights)
    env = Environment(num_obj=[max_objects], params=SimParams())
    rng = np.random.default_rng(seed)
    c = torch.zeros(classes, classes, dtype=torch.long)
    inst, grid = np.zeros(4), np.zeros(4)
    class_acc, free_iou, speed = [], [], []
    for i in range(scenes):
        np.random.seed(seed + i)
        random.seed(seed + i)
        env.num_obj = [int(rng.integers(2, max_objects + 1))]
        while not env.reset():
            pass
        color, depth, segm, *_ = env.camera.render()
        truth = type_map(segm, env.scene.names)
        t0 = time.perf_counter()
        types = network.predict_types(color, depth)
        speed.append(time.perf_counter() - t0)
        c += confusion(torch.from_numpy(types.astype(np.int64)), torch.from_numpy(truth.astype(np.int64)), classes)
        seg = network.instances_from_types(types, segm, env.scene)
        inst += match_instances(seg.instances, np.where(np.isin(segm, list(env.scene.names)), segm, 0))
        # the observation the policy gets from each perception on the same scene
        env.perception = GroundTruthPerception()
        reference = env.observe()
        env.perception = network
        observed = env.observe()
        env.perception = GroundTruthPerception()
        grid += match_instances(observed['idmap'].astype(np.int64), reference['idmap'].astype(np.int64), min_area=8)
        objects = (reference['classmap'] > 0) | (observed['classmap'] > 0)
        class_acc.append(float((reference['classmap'][objects] == observed['classmap'][objects]).mean()) if objects.any() else 1.0)
        a, b = reference['freespace'] > 0, observed['freespace'] > 0
        free_iou.append(float((a & b).sum() / max((a | b).sum(), 1)))
    iou = iou_from_confusion(c)
    print('pixel IoU: background %.3f, types %s (mean over types %.3f)' % (iou[0], np.round(iou[1:].numpy(), 3), float(iou[1:].mean())))
    print('instances in the image (connected components): recall %.3f (%d/%d), precision %.3f (%d/%d)'
          % (inst[0] / max(inst[1], 1), inst[0], inst[1], inst[2] / max(inst[3], 1), inst[2], inst[3]))
    print('instances in the top view (what the policy sees, split in 3D): recall %.3f (%d/%d), precision %.3f (%d/%d)'
          % (grid[0] / max(grid[1], 1), grid[0], grid[1], grid[2] / max(grid[3], 1), grid[2], grid[3]))
    print('observation of the policy vs ground truth: class map agreement on object cells %.3f, free-space IoU %.3f'
          % (np.mean(class_acc), np.mean(free_iou)))
    print('network inference: %.0f ms per image' % (1000 * np.mean(speed)))
    return {'mean_iou': float(iou[1:].mean()), 'recall': grid[0] / max(grid[1], 1), 'precision': grid[2] / max(grid[3], 1),
            'class_agreement': float(np.mean(class_acc)), 'free_space_iou': float(np.mean(free_iou))}


def evaluate_agnostic(weights, scenes=20, seed=1000, max_objects=8, object_types=None):
    """Object-agnostic network on fresh scenes (optionally restricted to `object_types`, e.g. including a held-out type):
    pixel IoU of interior / boundary, instance recall / precision in the image, and the policy's observation against ground truth."""
    from l2r.sim.environment import Environment
    perception = AgnosticPerception(weights)
    env = Environment(num_obj=[max_objects], params=SimParams(object_types=tuple(object_types) if object_types else None))
    rng = np.random.default_rng(seed)
    c = torch.zeros(3, 3, dtype=torch.long)
    grid, class_acc, free_iou = np.zeros(4), [], []
    for i in range(scenes):
        np.random.seed(seed + i)
        random.seed(seed + i)
        env.num_obj = [int(rng.integers(2, max_objects + 1))]
        while not env.reset():
            pass
        color, depth, segm, *_ = env.camera.render()
        pred = perception.predict(color, depth)
        c += confusion(torch.from_numpy(pred.astype(np.int64)), torch.from_numpy(agnostic_targets(segm, env.scene.names).astype(np.int64)), 3)
        env.perception = GroundTruthPerception()
        reference = env.observe()
        env.perception = perception
        observed = env.observe()
        env.perception = GroundTruthPerception()
        grid += match_instances(observed['idmap'].astype(np.int64), reference['idmap'].astype(np.int64), min_area=8)
        objects = (reference['classmap'] > 0) | (observed['classmap'] > 0)
        class_acc.append(float((reference['classmap'][objects] == observed['classmap'][objects]).mean()) if objects.any() else 1.0)
        a, b = reference['freespace'] > 0, observed['freespace'] > 0
        free_iou.append(float((a & b).sum() / max((a | b).sum(), 1)))
    iou = iou_from_confusion(c)
    result = {'iou_background': float(iou[0]), 'iou_interior': float(iou[1]), 'iou_boundary': float(iou[2]),
              'recall': grid[0] / max(grid[1], 1), 'precision': grid[2] / max(grid[3], 1),
              'class_agreement': float(np.mean(class_acc)), 'free_space_iou': float(np.mean(free_iou)), 'object_types': object_types}
    print('agnostic perception:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in result.items()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Segmentation network for the object masks of the observation.')
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--scenes', type=int, default=400)
    c.add_argument('--out', default='data/perception')
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--max_objects', type=int, default=10)
    c.add_argument('--object_types', nargs='+', default=None, help='YCB types of the scenes (>= 3), e.g. all but a held-out one')
    t = sub.add_parser('train')
    t.add_argument('--data', default='data/perception')
    t.add_argument('--out', default='assets/perception/segnet.pth')
    t.add_argument('--minutes', type=float, default=8.0)
    t.add_argument('--threads', type=int, default=4)
    t.add_argument('--task', choices=['types', 'agnostic'], default='types', help="'agnostic': object-agnostic masks (no type catalogue)")
    e = sub.add_parser('evaluate')
    e.add_argument('--weights', default='assets/perception/segnet.pth')
    e.add_argument('--scenes', type=int, default=20)
    e.add_argument('--task', choices=['types', 'agnostic'], default='types')
    e.add_argument('--object_types', nargs='+', default=None, help='restrict the evaluation scenes (agnostic task): include a held-out type')
    args = parser.parse_args(argv)
    if args.command == 'collect':
        collect(args.out, args.scenes, args.seed, max_objects=args.max_objects, object_types=args.object_types)
    elif args.command == 'train':
        train(args.data, args.out, args.minutes, threads=args.threads, task=args.task)
    elif args.task == 'agnostic':
        evaluate_agnostic(args.weights, args.scenes, object_types=args.object_types)
    else:
        evaluate(args.weights, args.scenes)
