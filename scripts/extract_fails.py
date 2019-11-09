import os
from os.path import join as pjoin
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser(description="Extract failed games from console_out file.")
    parser.add_argument("console_out", help="console output from score.py")
    parser.add_argument("--copy_to", default=None)
    parser.add_argument("--copy_from", default="train")
    args = parser.parse_args()

    if args.copy_to:
        if not os.path.exists(args.copy_to):
            os.mkdir(args.copy_to)
    with open(args.console_out) as f:
        fail_msgs = filter(lambda l: l.startswith('[0] '), f)
        for line in fail_msgs:
            _, score, game = line.split()
            print(game, score)
            if args.copy_to:
                shutil.copy2(pjoin(args.copy_from, game), args.copy_to)
                shutil.copy2(pjoin(args.copy_from, os.path.splitext(game)[0]+".json"), args.copy_to)


if __name__ == "__main__":
    main()
