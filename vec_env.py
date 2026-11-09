"""Run several simulators in parallel worker processes (simulation is the training bottleneck)."""
import multiprocessing as mp
import traceback


def _worker(conn, env_fn):
    try:
        env = env_fn()
        while True:
            cmd, data = conn.recv()
            if cmd == 'reset':
                conn.send(('ok', env.reset(**data)))
            elif cmd == 'step':
                prev = env.obs
                obs, reward, terminated, truncated, info = env.step(data)
                if info['relabel']:  # hindsight pair is built in the worker, where the simulator state lives
                    info['relabel_pair'] = env.relabel(prev, env.obs)
                conn.send(('ok', (obs, reward, terminated, truncated, info)))
            elif cmd == 'close':
                conn.send(('ok', None))
                break
    except Exception:  # report the traceback to the parent instead of dying silently
        conn.send(('error', traceback.format_exc()))
    finally:
        conn.close()


class VecEnv:
    """N environments, one process each. All calls are synchronous batches."""

    def __init__(self, env_fns, start_method=None):
        ctx = mp.get_context(start_method or 'spawn')  # spawn: safe with CUDA and PyBullet
        self.conns, self.procs = [], []
        for fn in env_fns:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, fn), daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)
        self.num_envs = len(env_fns)

    def _recv(self, i):
        status, data = self.conns[i].recv()
        if status == 'error':
            raise RuntimeError('worker %d failed:\n%s' % (i, data))
        return data

    def reset(self, num_obj, seeds=None, indices=None):
        indices = list(range(self.num_envs)) if indices is None else list(indices)
        for k, i in enumerate(indices):
            self.conns[i].send(('reset', {'num_obj': num_obj, 'seed': None if seeds is None else seeds[k]}))
        return [self._recv(i) for i in indices]

    def step(self, actions):
        for conn, action in zip(self.conns, actions):
            conn.send(('step', action))
        results = [self._recv(i) for i in range(self.num_envs)]
        obs, rewards, terminated, truncated, infos = map(list, zip(*results))
        return obs, rewards, terminated, truncated, infos

    def close(self):
        for i, conn in enumerate(self.conns):
            try:
                conn.send(('close', None))
                conn.recv()
            except (EOFError, OSError, BrokenPipeError):
                pass
        for proc in self.procs:
            proc.join(timeout=5)
