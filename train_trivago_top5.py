"""Compatibility entry point for the original trainer CLI and imports."""

import sys

from src import train_trivago_top5

if __name__ == "__main__":
    train_trivago_top5.main()
else:
    sys.modules[__name__] = train_trivago_top5
