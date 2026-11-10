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
from l2r.learning.object_centric import CLASS_RGB, COLOR_TOLERANCE, crop  # noqa: E402
from l2r.sim.constants import CLASSES, COLORS  # noqa: E402

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


CLASS_COLOURS = [COLORS[name][:3] for name in CLASSES]
QMAP = 'magma'


def scene_canvas(ax, obs, title):
    """A clean top view: light floor, occluded cells slightly darker, objects in their class colour with a dark outline, target regions
    as coloured bars on the left edge (the robot is on the right, looking left into the cabinet)."""
    colour, depth = crop(obs['colormap']).astype(int), crop(obs['depthmap'])
    h, w = depth.shape
    canvas = np.full((h, w, 3), 0.97)
    grey = (np.abs(colour - 105).sum(-1) <= 6)
    canvas[grey] = 0.86
    for k, rgb in enumerate(CLASS_RGB):
        cells = (depth > 0.008) & (np.abs(colour - rgb).sum(-1) <= COLOR_TOLERANCE)
        canvas[cells] = np.asarray(CLASS_COLOURS[k]) * 0.75 + 0.25
    ax.imshow(canvas, extent=(-0.5, w - 0.5, h - 0.5, -0.5))
    ax.contour(depth > 0.008, levels=[0.5], colors='0.25', linewidths=0.8)
    for k, (low, high) in enumerate(np.asarray(obs['regions'])):
        ax.add_patch(plt.Rectangle((-5.5, low - 0.5), 4, high - low + 1, color=CLASS_COLOURS[k], clip_on=False))
        ax.add_patch(plt.Rectangle((-0.5, low - 0.5), w, high - low + 1, fill=False, ec=CLASS_COLOURS[k], lw=0.8, ls=(0, (4, 3))))
    ax.set_xlim(-6, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, fontsize=10)
    return h, w


def overlay(fig, ax, values, valid, label='Q'):
    """Q-values on the valid cells only, with a slim horizontal colour bar under the panel."""
    data = np.ma.masked_where(~valid, values)
    im = ax.imshow(data, cmap=QMAP, alpha=0.85, extent=(-0.5, values.shape[1] - 0.5, values.shape[0] - 0.5, -0.5), interpolation='nearest')
    bar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.045, pad=0.03, aspect=30)
    bar.set_label(label, fontsize=8)
    bar.ax.tick_params(labelsize=7)
    bar.outline.set_visible(False)


def mark(ax, row, col, push=None):
    """The greedy action: a star, and for a push an arrow along the blade's path (`push` = (direction, length in cells))."""
    if push is not None:
        direction, length = push
        ax.annotate('', xy=(col, row + (length if direction == 1 else -length)), xytext=(col, row),
                    arrowprops=dict(arrowstyle='-|>', color='#00b4d8', lw=2.2, mutation_scale=14))
    ax.plot(col, row, marker='*', color='#00b4d8', mec='white', mew=1.0, ms=17)


def dense_figure(trainer, obs, held, path):
    q = trainer.forward_batch([obs])
    qh = trainer.forward_batch([held])
    depth = obs['depthmap']
    grasp_valid = trainer.valid_mask(depth, 'grasp')
    push_valid = trainer.valid_mask(depth, 'push')
    lengths_px = trainer.push_lengths_px
    n_len = len(lengths_px)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.2))
    scene_canvas(axes[0], obs, 'pick')
    values = q.pick[0, 0]
    overlay(fig, axes[0], values, grasp_valid)
    r, c = np.unravel_index(np.argmax(np.where(grasp_valid, values, -np.inf)), values.shape)
    mark(axes[0], r, c)
    for k, direction in enumerate((0, 1)):
        ax = axes[1 + k]
        scene_canvas(ax, obs, 'push %s (best length per cell)' % ('up' if direction == 0 else 'down'))
        stack = np.where(push_valid[direction * n_len:(direction + 1) * n_len], q.push[0, direction * n_len:(direction + 1) * n_len], -np.inf)
        best = stack.max(0)
        overlay(fig, ax, np.where(np.isfinite(best), best, 0), np.isfinite(best))
        r, c = np.unravel_index(np.argmax(best), best.shape)
        mark(ax, r, c, push=(direction, lengths_px[int(np.argmax(stack[:, r, c]))]))
    place_valid = np.asarray(held['freespace']) > 0
    scene_canvas(axes[3], held, 'place (object held)')
    values = qh.place[0, 0]
    overlay(fig, axes[3], values, place_valid)
    r, c = np.unravel_index(np.argmax(np.where(place_valid, values, -np.inf)), values.shape)
    mark(axes[3], r, c)
    fig.suptitle('Dense agent (U-Net + ConvNeXt): Q-values on the feasible cells, star = greedy action', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def object_figure(trainer, obs, held, path):
    info, info_h = trainer.info(obs), trainer.info(held)
    q = trainer.forward_batch([obs])
    qh = trainer.forward_batch([held])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), gridspec_kw=dict(width_ratios=[1, 0.8, 1]))
    # tokens
    ax = axes[0]
    scene_canvas(ax, obs, 'tokens: objects (numbered), ghosts (dashed)')
    for i, inst in enumerate(info['instances']):
        ax.text(inst.cols.mean(), inst.rows.mean(), str(i + 1), ha='center', va='center', fontsize=10, weight='bold', color='0.1')
    for ghost in info['ghosts']:
        ax.add_patch(plt.Rectangle((ghost.cols.min() - 0.5, ghost.rows.min() - 0.5), np.ptp(ghost.cols) + 1, np.ptp(ghost.rows) + 1,
                                   fill=False, ec='#c51b7d', lw=1.5, ls='--'))
    # skill values per object: a table, the greedy entry outlined and drawn on the scene
    ax = axes[1]
    n = len(info['instances'])
    table = np.full((max(n, 1), 3), np.nan)
    for i in range(n):
        for slot in range(3):
            if info['valid'][i, slot]:
                _, row, col = info['actions'][i, slot]
                table[i, slot] = q.pick[0, 0, row, col] if slot == 0 else q.push[0, slot - 1, row, col]
    im = ax.imshow(np.ma.masked_invalid(table), cmap=QMAP, aspect='auto')
    for i in range(n):
        for slot in range(3):
            text = 'n/a' if np.isnan(table[i, slot]) else '%.2f' % table[i, slot]
            bright = not np.isnan(table[i, slot]) and table[i, slot] > np.nanmin(table) + 0.6 * (np.nanmax(table) - np.nanmin(table))
            ax.text(slot, i, text, ha='center', va='center', fontsize=9, color='0.1' if bright else 'white' if text != 'n/a' else '0.5')
    if np.isfinite(table).any():
        i, slot = np.unravel_index(np.nanargmax(table), table.shape)
        ax.add_patch(plt.Rectangle((slot - 0.5, i - 0.5), 1, 1, fill=False, ec='#00b4d8', lw=3))
        _, row, col = info['actions'][i, slot]
        if slot == 0:
            mark(axes[0], row, col)
        else:
            cfg = trainer.cfg
            low, high = cfg.push_length_range
            length = low + (high - low) * float(q.lengths[0, slot - 1, row, col]) if q.lengths is not None else cfg.push_lengths[0]
            mark(axes[0], row, col, push=(slot - 1, length / cfg.pixel_size))
    ax.set_xticks(range(3))
    ax.set_xticklabels(['pick', 'push up', 'push down'], fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(['object %d' % (i + 1) for i in range(n)], fontsize=9)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.045, pad=0.06, aspect=30)
    bar.set_label('Q', fontsize=8)
    bar.ax.tick_params(labelsize=7)
    bar.outline.set_visible(False)
    ax.set_title('Q of each skill on each object (outlined: greedy)', fontsize=10)
    # placement
    ax = axes[2]
    scene_canvas(ax, held, 'place candidates (object held)')
    actions = info_h['place_actions']
    if len(actions):
        values = qh.place[0, 0, actions[:, 1], actions[:, 2]]
        grid = np.full(qh.place.shape[2:], np.nan)
        stride = int(np.min(np.diff(np.unique(actions[:, 2])))) if len(np.unique(actions[:, 2])) > 1 else 1
        for (_, row, col), value in zip(actions, values):
            grid[row:row + stride, col:col + stride] = value
        overlay(fig, ax, np.nan_to_num(grid, nan=values.min()), np.isfinite(grid))
        k = int(np.argmax(values))
        mark(ax, actions[k, 1], actions[k, 2])
    fig.suptitle('Object-centric agent: what it sees and how it values its options (star = greedy action)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
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
