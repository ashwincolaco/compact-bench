#!/usr/bin/env python
"""Run a job file one job at a time, skipping jobs whose output already exists.

Each non-comment line of the job file is

    <output json> | <compactbench arguments>

The file is re-read after every job, so jobs can be appended, reordered or
removed while the queue runs. A job that fails is logged and marked by a
.failed file next to its output, and is not retried until that file is removed.

    python scripts/queue.py jobs.txt runs/v3/queue.log
"""
import os, subprocess, sys, time


def pending(path):
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        out, cmd = (x.strip() for x in line.split("|", 1))
        if not os.path.exists(out) and not os.path.exists(out + ".failed"):
            return out, cmd
    return None


def main(jobs, log):
    while True:
        job = pending(jobs)
        if job is None:
            print("queue empty", time.strftime("%F %T"), file=open(log, "a"))
            return
        out, cmd = job
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(log, "a") as f:
            print(f"=== {time.strftime('%F %T')} start {out}: {cmd}", file=f, flush=True)
            t = time.time()
            rc = subprocess.call(f"compactbench {cmd} --out {out}", shell=True,
                                 stdout=f, stderr=subprocess.STDOUT)
            print(f"=== {time.strftime('%F %T')} end {out} rc={rc} "
                  f"({(time.time() - t) / 60:.1f} min)", file=f, flush=True)
        if rc != 0 or not os.path.exists(out):
            open(out + ".failed", "w").write(f"rc={rc}\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
