"""Compatibility entry point for the original dataset packaging CLI and imports."""

import sys

from src import trivago_data_parts

if __name__ == "__main__":
    trivago_data_parts.main()
else:
    sys.modules[__name__] = trivago_data_parts
