"""COMPACT-Bench command line.

  compactbench frontier       accuracy vs bytes-per-token budget (KV methods)
  compactbench reversibility  recoverable vs lossy storage at equal budget
  compactbench attribution    does the system know what it dropped?
  compactbench confidence     is post-compaction confidence calibrated?
  compactbench figures        render figures from runs/*.json
"""
import argparse

from .tasks import frontier, reversibility, attribution, confidence
from . import figures

TASKS = {
    "frontier": frontier,
    "reversibility": reversibility,
    "attribution": attribution,
    "confidence": confidence,
}


def main():
    ap = argparse.ArgumentParser(prog="compactbench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="task", required=True)
    for name, mod in TASKS.items():
        sp = sub.add_parser(name, help=mod.__doc__.splitlines()[0])
        mod.add_args(sp)
    fp = sub.add_parser("figures", help="render figures from runs/*.json")
    fp.add_argument("--runs", default="runs")
    fp.add_argument("--out", default="figures")

    args = ap.parse_args()
    if args.task == "figures":
        figures.render_all(args.runs, args.out)
    else:
        TASKS[args.task].main(args)


if __name__ == "__main__":
    main()
