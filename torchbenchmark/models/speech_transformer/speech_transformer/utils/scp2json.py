# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import sys
from builtins import str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", "-k", type=str, help="key")
    args = parser.parse_args()

    l = {}
    line = sys.stdin.readline()
    while line:
        x = str(line).rstrip().split()
        v = {args.key: " ".join(x[1:]).encode("utf_8")}
        l[x[0].encode("utf_8")] = v
        line = sys.stdin.readline()

    all_l = {"utts": l}

    # ensure "ensure_ascii=False", which is a bug
    jsonstring = json.dumps(all_l, indent=4, ensure_ascii=False)
    print(jsonstring)
