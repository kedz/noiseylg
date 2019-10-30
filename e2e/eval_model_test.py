import argparse
from pathlib import Path
import json
from subprocess import check_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("system", type=Path)
    parser.add_argument("refs", type=Path)
    args = parser.parse_args()
    
    with args.system.open("r") as fp, \
            open("hyp.txt", "w") as hyp_fp:
        for line in fp:
            data = json.loads(line)
#            print(data['text']) 
            print(data["text"], file=hyp_fp)
    out = check_output(["../eval_scripts/eval.py", "hyp.txt", args.refs])
    print(out.decode("utf8"))
if __name__ == "__main__":
    main()
