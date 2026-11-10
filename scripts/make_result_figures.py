"""Result figures for the README from the logged runs (logs/abl_*/validation.csv) and the diagnostics (runs/diag/*.json, runs/*.json).

    python scripts/make_result_figures.py --out docs/figures [--render]   (--render: also the paddle pusher in the simulator)"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e1'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']  # categorical, fixed order
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'font.size': 9, 'axes.titlesize': 10,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6})


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def validations(run):
    rows = []
    try:
        for r in csv.reader(open('logs/%s/validation.csv' % run)):
            rows.append((int(r[1]), int(r[2]), float(r[3])))
    except OSError:
        pass
    return rows


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8, color=INK2)
    ax.plot([x], [y], 'o', color=color, ms=4)


def learning_curves(out):
    runs = [('abl_D', 'dense ConvNeXt (DQN)'), ('abl_C', 'object, DQN'), ('abl_J', 'object, IQL'),
            ('abl_Jp', 'object, IQL + low eps'), ('abl_Jpp', 'object, IQL + 3-step + low eps')]
    fig, (stage, five) = plt.subplots(1, 2, figsize=(11, 3.8))
    for (run, name), color in zip(runs, SERIES):
        rows = validations(run)
        if not rows:
            continue
        steps, objects, rate = map(np.array, zip(*rows))
        stage.step(steps, objects + 0.04 * (SERIES.index(color) - 2), where='post', color=color, lw=2, label=name)
        at5 = objects == 5
        if at5.sum() >= 3:
            s, r = steps[at5], rate[at5]
            smooth = np.convolve(r, np.ones(3) / 3, mode='valid')
            five.plot(s[2:], smooth, color=color, lw=2)
            five.plot([s[-1]], [smooth[-1]], 'o', color=color, ms=4)
    stage.set(title='Curriculum stage (objects) during RL', xlabel='environment steps', ylabel='objects')
    stage.set_ylim(0.5, 6.8)
    five.set(title='Greedy validation success at 5 objects (mean of 3 validations)', xlabel='environment steps', ylabel='success')
    five.axhline(0.66, color=INK2, lw=1, ls='--')
    five.annotate('scripted expert (0.66)', (1, 0.66), xycoords=('axes fraction', 'data'), xytext=(-4, 4), textcoords='offset points',
                  ha='right', fontsize=8, color=INK2)
    five.set_ylim(0, 1)
    stage.set_yticks(range(1, 7))
    fig.legend(*stage.get_legend_handles_labels(), loc='lower center', ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(out, 'learning_curves.png'), dpi=130)
    plt.close(fig)


def failure_breakdown(out):
    agents = [('runs/diag/expert_straight_5obj.json', 'scripted expert'), ('runs/diag/D_latest_5obj.json', 'dense ConvNeXt, DQN (6k steps)'),
              ('runs/diag/J_0001000_5obj.json', 'object, IQL (1k steps)'), ('runs/diag/J_late_5obj.json', 'object, IQL (2k steps)'),
              ('runs/diag/H_late_5obj.json', 'object, DQN + 3-step (3k steps)'), ('runs/diag/I_late_5obj.json', 'object, twin + 3-step (3k steps)')]
    names, parts, regrasp = [], [], []
    for path, name in agents:
        d = load(path)
        if d is None:
            continue
        s, outcomes = d['summary'], d['summary']['outcomes']
        n = s['episodes']
        invalid = sum(v for k, v in outcomes.items() if k.startswith('invalid'))
        loop = sum(v for k, v in outcomes.items() if 'looping' in k)
        names.append(name)
        parts.append((s['success'], invalid / n, loop / n, 1 - s['success'] - invalid / n - loop / n))
        regrasp.append(1 - s['grasp on a misplaced object'])
    fig, (bars, dots) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[2, 1]))
    y = np.arange(len(names))[::-1]
    labels = ['sorted', 'invalid scene', 'out of actions: looping', 'out of actions: other']
    left = np.zeros(len(names))
    for k, (label, color) in enumerate(zip(labels, SERIES)):
        values = np.array([p[k] for p in parts])
        bars.barh(y, values, left=left, color=color, height=0.6, edgecolor=SURFACE, linewidth=2, label=label)
        for yi, l, v in zip(y, left, values):
            if v >= 0.08:
                bars.text(l + v / 2, yi, '%.0f%%' % (100 * v), ha='center', va='center', fontsize=8, color='white')
        left += values
    bars.set_yticks(y)
    bars.set_yticklabels(names)
    bars.set_xlim(0, 1)
    bars.grid(False)
    bars.set(title='Outcomes of 50 greedy episodes, 5 objects (same fresh scenes)', xlabel='share of episodes')
    bars.legend(ncol=4, fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    dots.scatter(regrasp, y, color=SERIES[1], s=40, zorder=3)
    for yi, v in zip(y, regrasp):
        dots.annotate('%.0f%%' % (100 * v), (v, yi), xytext=(6, 0), textcoords='offset points', va='center', fontsize=8, color=INK2)
    dots.set_yticks(y)
    dots.set_yticklabels([])
    dots.set_xlim(0, 1)
    dots.set(title='Grasps of objects already in their region', xlabel='share of grasps')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'failure_breakdown.png'), dpi=130)
    plt.close(fig)


def overestimation(out):
    runs = [('runs/diag/Q_D.json', 'dense ConvNeXt, DQN'), ('runs/diag/Q_C.json', 'object, DQN'), ('runs/diag/Q_F.json', 'object, DQN, reward x3')]
    names, all_, done = [], [], []
    for path, name in runs:
        d = load(path)
        if d is None:
            continue
        names.append(name)
        all_.append(d['summary']['Q - return (all)'])
        done.append(d['summary']['Q - return (finished episodes only)'])
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = np.arange(len(names))
    for k, (vals, label, color) in enumerate(((all_, 'all episodes', SERIES[0]), (done, 'finished episodes only', SERIES[2]))):
        ax.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=color, label=label, edgecolor=SURFACE, linewidth=2)
        for xi, v in zip(x, vals):
            ax.text(xi + (k - 0.5) * 0.36, v + 0.08, '+%.1f' % v, ha='center', fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set(title='Predicted Q of the chosen action minus the realised return (4 objects)', ylabel='overestimation')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'overestimation.png'), dpi=130)
    plt.close(fig)


def primitives(out):
    tip, paddle, approach = load('runs/diag/push_tip.json'), load('runs/diag/push_paddle.json'), load('runs/approach_profile.json')
    fig, (push, grasp) = plt.subplots(1, 2, figsize=(11, 3.4))
    if tip and paddle:
        kinds = ['ChipsCan', 'CrackerBox', 'MustardBottle', 'PottedMeatCan']
        x = np.arange(len(kinds))
        for k, (data, label, color) in enumerate(((tip, 'tool tip (2.6 cm)', SERIES[0]), (paddle, 'paddle pusher', SERIES[1]))):
            vals = [100 * data['push']['%s low' % kind]['moved (m)'] for kind in kinds]
            push.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=color, label=label, edgecolor=SURFACE, linewidth=2)
            for xi, v in zip(x, vals):
                push.text(xi + (k - 0.5) * 0.36, v + 0.1, '%.1f' % v, ha='center', fontsize=8, color=INK2)
        push.set_xticks(x)
        push.set_xticklabels(['chips can', 'cracker box', 'mustard bottle', 'meat can'])
        push.set(title='Object displacement of an 8 cm push (median, 40 scenes)', ylabel='cm')
        push.legend(frameon=False, fontsize=8)
    if approach:
        s = approach['summary']['straight corridor blocked']
        metrics = [('held', 'object held after the grasp'), ('invalid', 'scene invalid (something toppled)')]
        x = np.arange(len(metrics))
        for k, (mode, label, color) in enumerate((('straight', 'straight in', SERIES[0]), ('clear', 'clearest approach line', SERIES[2]))):
            vals = [s[mode][m] for m, _ in metrics]
            grasp.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=color, label=label, edgecolor=SURFACE, linewidth=2)
            for xi, v in zip(x, vals):
                grasp.text(xi + (k - 0.5) * 0.36, v + 0.015, '%.2f' % v, ha='center', fontsize=8, color=INK2)
        grasp.set_xticks(x)
        grasp.set_xticklabels([label for _, label in metrics])
        grasp.set_ylim(0, 1.1)
        grasp.set(title='Grasps whose straight corridor is blocked (%d, 40 scenes)' % s['straight']['n'], ylabel='share')
        grasp.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'primitives.png'), dpi=130)
    plt.close(fig)


def render_paddle(out):
    from l2r.arch_benchmark import make_env
    from l2r.config import Config
    from l2r.sim.primitives import PUSH_HEIGHT, PUSH_QUATERNION
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_figures import INSIDE, external_view
    cfg = Config(placement='learned', footprint_placement=True)
    env = make_env(cfg, 4)
    env.sim.params.pusher = 'paddle'
    env.reset(num_obj=4, seed=7003)
    sim, m = env.sim, env.sim.manipulator
    b = sim.ws.bounds
    uid = max(sim.scene.object_ids, key=lambda i: sim.p.getBasePositionAndOrientation(i)[0][0])
    pos = sim.p.getBasePositionAndOrientation(uid)[0]
    start = (pos[0], pos[1] - 0.09, b[2][0] + PUSH_HEIGHT)
    front = m._front(start[1], start[2])
    sim.arm.move_to_pose((front, PUSH_QUATERNION), 0.01)
    paddle = m._attach_paddle(front_plate=True)
    sim.arm.move_straight(front, start, PUSH_QUATERNION, 0.01)
    image = external_view(sim and env, **INSIDE)
    m._detach_paddle(paddle)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title('Paddle pusher (--pusher paddle), during a push', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'paddle.png'), dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', default='docs/figures')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    learning_curves(args.out)
    failure_breakdown(args.out)
    overestimation(args.out)
    primitives(args.out)
    if args.render:
        render_paddle(args.out)
    print('figures written to', args.out)


if __name__ == '__main__':
    main()
