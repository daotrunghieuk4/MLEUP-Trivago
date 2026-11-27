"""Compatibility entry point for the original preprocessing CLI and imports."""

import sys

from src import prepare_trivago_top5

if __name__ == "__main__":
    prepare_trivago_top5.main()
else:
    sys.modules[__name__] = prepare_trivago_top5
