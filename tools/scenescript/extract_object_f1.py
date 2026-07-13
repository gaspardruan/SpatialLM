#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    args = parser.parse_args()

    matches = re.findall(
        r"^\|\s*avg\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*\|$",
        Path(args.log).read_text(),
        flags=re.MULTILINE,
    )
    if not matches:
        raise RuntimeError(f"No object average row found in {args.log}")
    f1_25, f1_50 = matches[-1]
    print(f"{float(f1_25):.10f}\t{float(f1_50):.10f}")


if __name__ == "__main__":
    main()
