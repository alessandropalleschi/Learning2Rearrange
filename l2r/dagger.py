"""DAgger with the scripted expert (Ross et al. 2011): the learner drives, the privileged expert labels every state it visits, the
labelled states are aggregated with the demonstrations and the learner is retrained. Fixes the covariate shift of plain behaviour
cloning, where the learner meets states (after its own mistakes) that the expert's demonstrations never contain.

Works with any variant of `arch_benchmark` (`train_step(batch)`, `predict(obs) -> (primitive, pix)`, `cfg`). The aggregated transition is
(state, EXPERT action, state reached by the action actually executed, reward, expert primitive, done): imitation losses only use the
first three, so it is not a valid RL transition and must not be used for TD targets."""
import numpy as np

from l2r.demos import make_expert


def collect_round(variant, env, episodes, num_objects, beta, rng, seed, max_steps=20, log=print):
    """Roll out the learner (each action is the expert's with probability `beta`), label every visited state with the expert's action.
    Returns (labelled transitions, learner statistics). Episodes where the expert has no action end early."""
    from l2r.learning.memory import Transition
    labelled, solved, actions = [], 0, 0
    for episode in range(episodes):
        count = int(rng.integers(num_objects[0], num_objects[1] + 1))
        obs = env.reset(num_obj=count, seed=seed * 100003 + episode)
        expert, steps, success = make_expert(env), 0, False
        while steps < max_steps:
            choice = expert.act()
            if choice is None:
                break
            driver = choice if rng.random() < beta else variant.predict(obs)
            primitive, pix = driver
            grasped = expert.object_at(pix) if primitive == 'grasp' else None
            full = env.obs
            next_obs, reward, terminated, truncated, info = env.step({'primitive': primitive, 'pix': pix})
            expert.report(primitive, pix, grasped, info['executed'])
            labelled.append(Transition(obs, choice[1], next_obs, reward, choice[0], bool(terminated)))
            if info['relabel']:
                obs_r, next_r = env.relabel(full, env.obs)
                labelled.append(Transition(obs_r, choice[1], next_r, env.cfg.reward_goal, choice[0], True))
            obs = next_obs
            if info.get('phase') != 'place':
                steps += 1
            if terminated or truncated:
                success = bool(info['sorted'])
                break
        solved += success
        actions += steps
        log('  dagger episode %d (%d objects, beta %.2f): %s in %d actions' % (episode, count, beta, 'sorted' if success else 'not sorted', steps))
    return labelled, {'success': solved / episodes, 'actions': actions / episodes, 'new': len(labelled)}


def dagger(variant, env, demos, rounds, episodes, train_steps, batch, num_objects=(1, 4), beta0=0.5, seed=0, max_steps=20, log=print,
           after_pretrain=None):
    """Pretrain on `demos`, then `rounds` x (collect `episodes` with a decaying expert mixture, aggregate, train `train_steps`).
    Returns the history, one dict per round (round 0 = behaviour cloning only; its success is measured by the first learner rollout)."""
    rng = np.random.default_rng(seed)
    data, history = list(demos), []

    def fit(steps):
        losses = [variant.train_step([data[i] for i in rng.integers(len(data), size=batch)]) for _ in range(steps)]
        return float(np.mean(losses[-100:]))

    loss = fit(train_steps)
    if after_pretrain is not None:  # e.g. evaluate the behaviour-cloning-only policy (the reference DAgger has to beat)
        after_pretrain()
    for r in range(1, rounds + 1):
        beta = beta0 * 0.5 ** (r - 1) if r < rounds else 0.0  # the last round is pure learner: the cleanest estimate of its success
        new, stats = collect_round(variant, env, episodes, num_objects, beta, rng, seed * 1000 + r, max_steps, log)
        data.extend(new)
        stats.update(round=r, beta=beta, data=len(data), loss_before=loss)
        history.append(stats)
        log('round %d: learner success %.2f (beta %.2f, %.1f actions), aggregated %d transitions' % (r, stats['success'], beta, stats['actions'], len(data)))
        loss = fit(train_steps)
    return history
