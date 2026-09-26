"""COMPACT-Bench command line.

  compactbench frontier       accuracy vs bytes-per-token budget (KV methods)
  compactbench reversibility  recoverable vs lossy storage at equal budget
  compactbench attribution    does the system know what it dropped?
  compactbench confidence     is post-compaction confidence calibrated?
  compactbench audit          probe, confidence, and needle survival on one cache
  compactbench mechanisms     eviction, quantization, prompt, summary on one axis
  compactbench figures        render figures from runs/*.json
  compactbench scale          cross-scale analysis of a scale-ladder sweep
"""
import argparse

from .tasks import frontier, reversibility, attribution, confidence, audit, mechanisms
from . import figures, scale

TASKS = {
    "frontier": frontier,
    "reversibility": reversibility,
    "attribution": attribution,
    "confidence": confidence,
    "audit": audit,
    "mechanisms": mechanisms,
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
    sc = sub.add_parser("scale", help="cross-scale analysis of a scale-ladder sweep")
    sc.add_argument("--runs", default="runs/scale")
    sc.add_argument("--out", default="figures")
    sc.add_argument("--no-fig", action="store_true")

    args = ap.parse_args()
    if args.task == "figures":
        figures.render_all(args.runs, args.out)
    elif args.task == "scale":
        scale.main(args)
    else:
        TASKS[args.task].main(args)


if __name__ == "__main__":
    main()
