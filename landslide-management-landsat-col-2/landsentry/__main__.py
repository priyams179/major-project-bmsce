"""``python -m landsentry <command> [args]``

Commands:
  prepare       download Sentinel-2 pre/post composites (Earth Engine)
  labels        build label masks + spatial splits + QC figures
  train         train one model on one source event (or both)
  evaluate      score one checkpoint on one event / split
  cross-event   full cross-event experiment (all models x sources x seeds)
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd == "prepare":
        from .prepare import main as m
    elif cmd == "labels":
        from .labels import main as m
    elif cmd == "train":
        from .train import main as m
    elif cmd == "evaluate":
        from .evaluate import main as m
    elif cmd in ("cross-event", "cross_event"):
        from .cross_event import main as m
    else:
        print(f"unknown command {cmd!r}\n{__doc__}")
        sys.exit(2)
    m(argv)


if __name__ == "__main__":
    main()
