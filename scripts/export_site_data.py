#!/usr/bin/env python
"""Export the run records behind the paper into docs/data.js for the project page.

    python scripts/export_site_data.py [runs_root] [out]

The page reads `window.COMPACT_DATA` from a plain script so it works from any
static host, a subpath, or a local file, with no fetch and no build step.
"""
import glob, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
import v3_report as V  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/data.js"
V.ROOT = ROOT
KEEP = [0.9, 0.75, 0.5, 0.25, 0.1]
METHODS = ["SnapKV", "StreamingLLM", "TOVA", "Knorm", "ExpectedAttn", "Random"]
r3 = lambda x: None if x is None else round(x, 3)


def frontier_entry(tag, f):
    name, params, date, gen = V.MODELS[tag]
    c = V.collapse(f, V.CONTENT + ["StreamingLLM"])
    return {"tag": tag, "name": name.rstrip("*"), "params": params, "released": date,
            "gen": gen, "precision": V.PREC[tag], "fullBpt": f["full_bpt_bytes_per_tok"],
            "baseline": r3(V.base(f)),
            "collapse": 0.9 if isinstance(c, str) else r3(c), "collapseAbove": isinstance(c, str),
            "acc": {m: [r3(V.curve(f, [m])[k]) for k in KEEP] for m in METHODS}}


def main():
    fr = V.frontiers("fr")
    frontiers = [frontier_entry(t, fr[t]) for t in V.FULL_ORDER if t in fr]

    mech = []
    for p in sorted(glob.glob(f"{ROOT}/v3/me-*.json")):
        d = json.load(open(p)); tag = os.path.basename(p)[3:-5]
        mech.append({"tag": tag, "name": V.MODELS[tag][0].rstrip("*"),
                     "points": [{"family": q["family"], "method": q["method"],
                                 "setting": q["setting"], "bpt": r3(q["bpt_frac"]),
                                 "acc": r3(q["acc"]), "n": q["n"]} for q in d["points"]]})

    AG = {"qwen1.5b": "qwen1.5b", "qwen3-1.7b": "qwen3-1.7b", "qwen3b-nf4": "qwen3b",
          "qwen3-4b-nf4": "qwen3-4b-nf4", "qwen0.5b": "qwen0.5b", "qwen3-0.6b": "qwen3-0.6b"}
    qa = []
    for tag, ag in AG.items():
        pw = f"{ROOT}/v3/qa-{tag}.json"
        if not os.path.exists(pw) or ag not in fr:
            continue
        aw = json.load(open(pw))
        qa.append({"tag": tag, "name": V.MODELS[tag][0].rstrip("*"), "gen": V.MODELS[tag][3],
                   "agnostic": {m: [r3(V.curve(fr[ag], [m])[k]) for k in KEEP] for m in METHODS},
                   "aware": {m: [r3(V.curve(aw, [m])[k]) for k in KEEP] for m in METHODS}})

    audit = []
    files = sorted(glob.glob(f"{ROOT}/v3/au-*.json"))
    for p in files:
        tag = os.path.basename(p)[3:-5]
        R = json.load(open(p))["records"]
        old_parser = tag == "qwen14b-nf4"
        mean = lambda v: r3(sum(v) / len(v)) if v else None
        sel = lambda m, r=None: [x for x in R if x["method"] == m and (r is None or x["ratio"] == r)]
        conf = lambda xs: None if old_parser else mean([x["conf"] for x in xs if x["conf"] is not None])
        pres, absn = sel("full-present"), sel("full-absent")
        e = {"tag": tag, "name": V.MODELS[tag][0].rstrip("*"), "gen": V.MODELS[tag][3],
             "presentYes": mean([x["probe_yes"] for x in pres]),
             "absentYes": mean([x["probe_yes"] for x in absn]),
             "confAbsent": conf(absn), "confPresent": conf(pres),
             "confOmitted": old_parser, "methods": {}}
        e["probeValid"] = e["presentYes"] >= 0.5 and e["absentYes"] <= 0.1
        for m in ["SnapKV", "StreamingLLM", "Knorm", "Random"]:
            mm = {}
            for r in (0.5, 0.9):
                xs = sel(m, r); wrong = [x for x in xs if not x["correct"]]
                mm[str(r)] = {"acc": mean([x["correct"] for x in xs]),
                              "overclaim": mean([x["probe_yes"] for x in wrong]),
                              "nWrong": len(wrong)}
            wrong = [x for x in sel(m) if not x["correct"]]
            mm["confWrong"] = conf(wrong)
            for said in (1, 0):
                w = [x for x in wrong if x["probe_yes"] == said and x.get("key_kept") is not None]
                mm["survival" + ("Yes" if said else "No")] = {
                    "n": len(w), "key": mean([x["key_kept"] for x in w]),
                    "value": mean([x["value_kept"] for x in w])}
            e["methods"][m] = mm
        audit.append(e)

    rev = []
    for tag, path in V.REV.items():
        std = f"{ROOT}/{path}.json"
        if not os.path.exists(std):
            continue
        pts = lambda d: [{"budget": q["budget_frac"], "summary": r3(q["irreversible_acc"]),
                          "raw": r3(q["reversible_acc"])} for q in d["points"]]
        e = {"tag": tag, "name": V.MODELS[tag][0].rstrip("*"), "gen": V.MODELS[tag][3],
             "earlierHarness": path.startswith("scale/"),
             "standard": pts(json.load(open(std)))}
        dtag = {"qwen3b-fp16": "qwen3b-nf4"}.get(tag, tag)
        dp = f"{ROOT}/v3/rd-{dtag}.json"
        if os.path.exists(dp):
            e["dense"] = pts(json.load(open(dp)))
            e["denseNote"] = "dense run in NF4" if dtag != tag else None
        rev.append(e)

    data = {"keep": KEEP, "methods": METHODS, "frontiers": frontiers, "mechanisms": mech,
            "questionAware": qa, "audit": audit, "reversibility": rev,
            "denseFactShare": 0.42}
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        f.write("// Generated by scripts/export_site_data.py from the run records. Do not edit.\n")
        f.write("window.COMPACT_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print(f"wrote {OUT}: {len(frontiers)} frontiers, {len(mech)} mechanism runs, "
          f"{len(qa)} question-aware, {len(audit)} audits, {len(rev)} reversibility")


if __name__ == "__main__":
    main()
