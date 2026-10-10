"""Command line entry point: python -m evprior <stage>."""
from __future__ import annotations

import argparse
import importlib
import logging
import sys

from .config import load_config

STAGES = ["ingest", "enrich", "segments", "prior", "ved", "evaluate", "ved_eval", "analyze", "ablate", "report"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evprior")
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(a.config)
    for s in (STAGES if a.stage == "all" else [a.stage]):
        try:
            mod = importlib.import_module(f"evprior.{s}")
        except ModuleNotFoundError as e:
            if e.name != f"evprior.{s}":
                raise
            logging.getLogger("evprior").warning("stage %s not implemented yet", s)
            continue
        mod.run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
