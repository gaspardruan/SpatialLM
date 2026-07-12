#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")
    inference_checkpoint = {
        "model_state_dict": checkpoint["model_state_dict"],
        "cfg": checkpoint["cfg"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(inference_checkpoint, output)
    print(output)


if __name__ == "__main__":
    main()
