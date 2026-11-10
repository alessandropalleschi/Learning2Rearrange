"""Supervised architecture benchmark: `python scripts/arch_benchmark.py --help`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l2r.arch_benchmark import main  # noqa: E402

if __name__ == '__main__':
    main()
