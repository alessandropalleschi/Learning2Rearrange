"""The simulator boundary: everything `RearrangeEnv` (and so the learning code) needs from a physics backend.

`l2r.sim.environment.Environment` (PyBullet) implements it. A different backend (for example Isaac Sim / Isaac Lab) only has to
provide these members; nothing above this line (policy, training, belief, perception, rewards) touches the physics engine.
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class SimBackend(Protocol):
    ws: object  # `Workspace`: cabinet bounds and the heightmap pixel <-> metre conversion
    scene: object  # task state: .object_ids .names .positions() .progress() .relabel_ids / .relabel_classes
    force_stops: int  # number of motions aborted by the contact-force limit (diagnostics)
    force_events: list

    def reset(self) -> bool:
        """New random arrangement; False if it is not valid (the caller retries)."""

    def observe(self) -> dict:
        """'colormap' / 'depthmap' (padded) and 'idmap' / 'classmap' / 'freespace' (not padded), see `l2r.sim.vision`."""

    def push(self, xy, angle, push_length, speed) -> bool: ...

    def pick(self, xy, speed) -> bool: ...

    def place(self, xy, speed) -> bool: ...

    def wait_static(self) -> bool: ...

    def is_valid(self) -> bool: ...

    def fraction_sorted(self) -> float: ...

    def relabel_candidates(self) -> float: ...

    def relabel_colormap(self, colormap, id_map, ids=None, classes=None): ...

    def target_pixels(self): ...
