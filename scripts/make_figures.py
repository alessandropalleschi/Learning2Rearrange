"""Figures for the README: the scene and robot, the camera, the observation channels and the Q-maps of the trained agents.

    python scripts/make_figures.py --out docs/figures \
        --dense_snapshot logs/rl_convnext_belief/models/snapshot-0005000.pth \
        --object_snapshot logs/rl_object_belief/models/snapshot-0008000.pth

The dense snapshot must be a `--arch_skips --arch_encoder convnext --push_lengths 0.04 0.08 0.12 --belief spatial` network and the
object one an `--agent object --hierarchy --belief spatial` network (both `--placement learned --footprint_placement`); without
snapshots the Q-map figures are skipped. Everything runs headless (TinyRenderer)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from l2r.arch_benchmark import make_env  # noqa: E402
from l2r.config import Config  # noqa: E402
from l2r.demos import ScriptedExpert  # noqa: E402
from l2r.learning.factory import make_trainer  # noqa: E402
from l2r.learning.object_centric import crop  # noqa: E402

MODEL = dict(placement='learned', footprint_placement=True, belief='spatial')
DENSE = dict(MODEL, arch_skips=True, arch_encoder='convnext', push_lengths=(0.04, 0.08, 0.12))
OBJECT = dict(MODEL, agent='object', hierarchy=True)


OVERVIEW = dict(target=(0.45, 0.0, 0.4), yaw=130, pitch=-25, distance=1.9)  # robot and cabinet
INSIDE = dict(target=(0.0, 0.0, 0.45), yaw=90, pitch=-30, distance=0.95)  # through the opening, from behind the robot


def external_view(env, target, yaw, pitch, distance, size=(480, 640)):
    """A view of robot, cabinet and objects from outside (not the robot's camera)."""
    p = env.sim.p
    view = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=list(target), distance=distance, yaw=yaw, pitch=pitch,
                                               roll=0, upAxisIndex=2)
    projection = p.computeProjectionMatrixFOV(50, size[1] / size[0], 0.05, 5.0)
    _, _, color, _, _ = p.getCameraImage(width=size[1], height=size[0], viewMatrix=view, projectionMatrix=projection,
                                         renderer=p.ER_TINY_RENDERER)
    return np.array(color, np.uint8).reshape(size[0], size[1], 4)[..., :3]


def show(ax, image, title, **kwargs):
    ax.imshow(image, **kwargs)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def masked(q, valid):
    out = np.where(valid, q, np.nan)
    return np.ma.masked_invalid(out)


def regions_overlay(ax, obs):
    for low, high in np.asarray(obs['regions']):
        ax.axhline(low, color='k', lw=0.6, ls='--')
        ax.axhline(high, color='k', lw=0.6, ls='--')


def scene_figure(env, path):
    color, depth, _, _, _, _ = env.sim.camera.render()
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    show(axes[0], external_view(env, **OVERVIEW), 'UR5e with the suction tool in front of the cabinet')
    show(axes[3], external_view(env, **INSIDE), 'inside the cabinet (outside view)')
    show(axes[1], color, "robot's RGB camera (front)")
    show(axes[2], np.ma.masked_greater(depth, 2.0), "robot's depth camera (m)", cmap='viridis')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def observation_figure(obs, held, path):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    show(axes[0], crop(obs['colormap']), 'colour map: class colours,\ngrey = occluded, target regions (dashed)')
    regions_overlay(axes[0], obs)
    im = axes[1].imshow(crop(obs['depthmap']), cmap='magma')
    show(axes[1], crop(obs['depthmap']), 'height map (m)', cmap='magma')
    fig.colorbar(im, ax=axes[1], fraction=0.04)
    belief = np.asarray(obs['belief'])
    show(axes[2], crop(belief[4]).astype(np.float32) / 255.0, 'belief: age of the memory\n(0 = seen now, 1 = stale)', cmap='Greys',
         vmin=0, vmax=1)
    show(axes[3], crop(held['colormap']), 'object held: free cells it fits in\n(learned placement candidates)')
    axes[3].imshow(np.ma.masked_equal(np.asarray(held['freespace']), 0), cmap='autumn', alpha=0.45)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def dense_figure(trainer, obs, held, path):
    q = trainer.forward_batch([obs])
    qh = trainer.forward_batch([held])
    o = trainer.offset_px
    depth = obs['depthmap']
    grasp_valid = trainer.valid_mask(depth, 'grasp')
    push_valid = trainer.valid_mask(depth, 'push')
    n_len = len(trainer.cfg.push_lengths)
    base = crop(obs['colormap'], o)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    panels = [(q.pick[0, 0], grasp_valid, 'pick Q (front-face cells)')]
    for direction, name in ((0, 'push Q, direction 0'), (1, 'push Q, direction 1')):
        stack = q.push[0, direction * n_len:(direction + 1) * n_len]
        valid = push_valid[direction * n_len:(direction + 1) * n_len]
        best = np.where(valid, stack, -np.inf).max(0)
        panels.append((best, valid.any(0), name + '\n(best of the 3 lengths)'))
    place_valid = np.asarray(held['freespace']) > 0
    panels.append((qh.place[0, 0], place_valid, 'place Q while holding\n(free cells)'))
    for ax, (values, valid, title) in zip(axes, panels):
        ax.imshow(base, alpha=0.35)
        im = ax.imshow(masked(values, valid), cmap='viridis')
        fig.colorbar(im, ax=ax, fraction=0.04)
        flat = np.where(valid, values, -np.inf)
        r, c = np.unravel_index(np.argmax(flat), flat.shape)
        ax.plot(c, r, 'r*', ms=12)
        show(ax, np.zeros((1, 1)), title, alpha=0)
        ax.set_xlim(0, base.shape[1] - 1)
        ax.set_ylim(base.shape[0] - 1, 0)
    fig.suptitle('Dense U-Net + ConvNeXt Q-maps (red star: greedy action)', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def object_figure(trainer, obs, held, path):
    info, info_h = trainer.info(obs), trainer.info(held)
    q = trainer.forward_batch([obs])
    qh = trainer.forward_batch([held])
    base = crop(obs['colormap'])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    show(axes[0], base, 'object tokens: extracted instances\n(boxes) and ghost tokens (dashed)')
    for inst in info['instances']:
        axes[0].add_patch(plt.Rectangle((inst.cols.min() - 0.5, inst.rows.min() - 0.5), np.ptp(inst.cols) + 1, np.ptp(inst.rows) + 1,
                                        fill=False, ec='k', lw=1.2))
    for ghost in info['ghosts']:
        axes[0].add_patch(plt.Rectangle((ghost.cols.min() - 0.5, ghost.rows.min() - 0.5), np.ptp(ghost.cols) + 1,
                                        np.ptp(ghost.rows) + 1, fill=False, ec='m', lw=1.2, ls='--'))
    regions_overlay(axes[0], obs)
    show(axes[1], base, 'skill values per object\n(o pick, ^ v push; label: Q)', alpha=0.6)
    values = []
    for i in range(len(info['instances'])):
        for slot in range(info['valid'].shape[1]):
            if info['valid'][i, slot]:
                _, row, col = info['actions'][i, slot]
                value = q.pick[0, 0, row, col] if slot == 0 else q.push[0, slot - 1, row, col]
                values.append((value, row, col, slot))
    if values:
        vmin, vmax = min(v[0] for v in values), max(v[0] for v in values)
        best = max(values)
        for value, row, col, slot in values:
            marker = 'o' if slot == 0 else ('^' if slot == 1 else 'v')
            sc = axes[1].scatter(col, row, c=[value], cmap='viridis', vmin=vmin, vmax=vmax, marker=marker, s=80, edgecolors='k')
            axes[1].annotate('%.1f' % value, (col, row), textcoords='offset points', xytext=(4, 4), fontsize=7)
        axes[1].plot(best[2], best[1], 'r*', ms=14)
        fig.colorbar(sc, ax=axes[1], fraction=0.04)
    show(axes[2], crop(held['colormap']), 'place candidates while holding\n(colour: Q, red star: greedy)', alpha=0.6)
    actions = info_h['place_actions']
    if len(actions):
        place = qh.place[0, 0, actions[:, 1], actions[:, 2]]
        sc = axes[2].scatter(actions[:, 2], actions[:, 1], c=place, cmap='viridis', s=10)
        k = int(np.argmax(place))
        axes[2].plot(actions[k, 2], actions[k, 1], 'r*', ms=14)
        fig.colorbar(sc, ax=axes[2], fraction=0.04)
    fig.suptitle('Object-centric agent: tokens, skill values and placement', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def episode_figure(env, seed, num_obj, path, max_steps=6):
    obs = env.reset(num_obj=num_obj, seed=seed)
    expert = ScriptedExpert(env)
    frames = [(external_view(env, **INSIDE), crop(obs['colormap']), 'start')]
    for _ in range(max_steps):
        choice = expert.act()
        if choice is None:
            break
        primitive, pix = choice
        grasped = expert.object_at(pix) if primitive == 'grasp' else None
        obs, _, terminated, truncated, info = env.step({'primitive': primitive, 'pix': pix})
        expert.report(primitive, pix, grasped, info['executed'])
        label = '%s at (%d, %d)' % (primitive, pix[1], pix[2])
        frames.append((external_view(env, **INSIDE), crop(obs['colormap']), label))
        if terminated or truncated:
            break
    fig, axes = plt.subplots(2, len(frames), figsize=(3.2 * len(frames), 5.6))
    axes = np.asarray(axes).reshape(2, -1)
    for k, (outside, colormap, label) in enumerate(frames):
        show(axes[0, k], outside, '%d. %s' % (k, label))
        show(axes[1, k], colormap, 'observation')
    fig.suptitle('One episode of the scripted expert: robot state (top) and the observation the agent gets (bottom)', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def held_state(env, obs):
    """Grasp the expert's object and return the observation while it is held (learned placement), or None."""
    expert = ScriptedExpert(env)
    for _ in range(6):
        choice = expert.act()
        if choice is None:
            return None
        primitive, pix = choice
        new, _, terminated, _, info = env.step({'primitive': primitive, 'pix': pix})
        if info.get('phase') == 'place':
            return new
        if terminated:
            return None
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', default='docs/figures')
    parser.add_argument('--dense_snapshot', default='')
    parser.add_argument('--object_snapshot', default='')
    parser.add_argument('--seed', type=int, default=7003)
    parser.add_argument('--objects', type=int, default=5)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    env = make_env(Config(**DENSE), args.objects, max_actions=20)
    episode_figure(env, args.seed, args.objects, os.path.join(args.out, 'episode.png'))
    obs = env.reset(num_obj=args.objects, seed=args.seed)
    scene_figure(env, os.path.join(args.out, 'scene.png'))
    held = None
    for seed in range(args.seed, args.seed + 20):  # a scene where, after one pick-and-place, the expert grasps again
        obs = env.reset(num_obj=args.objects, seed=seed)
        expert = ScriptedExpert(env)
        choice = expert.act()  # one action first, so the belief has something to remember
        if choice is None:
            continue
        obs, _, terminated, _, _ = env.step({'primitive': choice[0], 'pix': choice[1]})
        if env.holding:
            obs, _, terminated, _, _ = env.step({'primitive': 'place', 'pix': expert.act()[1]})
        if terminated:
            continue
        held = held_state(env, obs)
        if held is not None:
            break
    if held is None:
        print('could not reach a holding state; observation/Q-map figures skipped')
        return
    observation_figure(obs, held, os.path.join(args.out, 'observation.png'))
    if args.dense_snapshot:
        trainer = make_trainer(Config(**DENSE), is_testing=True, load_snapshot=True, snapshot=args.dense_snapshot)
        dense_figure(trainer, obs, held, os.path.join(args.out, 'qmaps_dense.png'))
    if args.object_snapshot:
        trainer = make_trainer(Config(**OBJECT), is_testing=True, load_snapshot=True, snapshot=args.object_snapshot)
        object_figure(trainer, obs, held, os.path.join(args.out, 'qmaps_object.png'))
    print('figures written to', args.out)


if __name__ == '__main__':
    main()
