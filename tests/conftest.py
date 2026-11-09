import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))

try:
    import pybullet  # noqa: F401
except ImportError:  # CI without PyBullet: stub the modules so the package can be imported
    for name in ['pybullet', 'pybullet_data', 'pybullet_utils', 'pybullet_utils.bullet_client']:
        sys.modules[name] = types.ModuleType(name)
    sys.modules['pybullet_utils'].bullet_client = sys.modules['pybullet_utils.bullet_client']
