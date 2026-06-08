#!/usr/bin/env python3
"""Policy server entrypoint for multi-image/depth/crop rollout policies.

This is intentionally a thin wrapper around ``scripts/serve_policy.py``.  The
existing server already supports checkpoint loading, RTC chunk conditioning,
mask overlay, policy recording, and metadata echoing.  Keeping this file thin
avoids creating a second divergent server implementation while giving depth /
crop policies a stable command name:

    python scripts/serve_policy_multi.py policy:checkpoint \
      --policy.config pi05_task_Insert_stick_multi_times_with_depth_crop \
      --policy.dir /path/to/checkpoint \
      --rtc-chunk-conditioning
"""

from __future__ import annotations

import logging
import pathlib
import sys

import tyro


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from serve_policy import Args  # noqa: E402
from serve_policy import main  # noqa: E402


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
