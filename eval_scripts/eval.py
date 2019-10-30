#!/usr/bin/env python

import argparse
from subprocess import check_output
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyps")
    parser.add_argument("refs")
    args = parser.parse_args()

    e2e_eval_script = (
        Path(__file__).parent / "e2e-metrics" / "measure_scores.py"
    )

    with open(os.devnull, "w") as DEVNULL:
        output = check_output(
            "python {} {} {}".format(
                e2e_eval_script, args.refs, args.hyps),
            shell=True,
            stderr=DEVNULL).decode("utf8")

    results = {}
    for line in output.strip().split("\n")[2:]:
        metric, score = line.split(": ")
        score = float(score)
        results[metric] = score
    print(json.dumps(results))

if __name__ == "__main__":
    main()
