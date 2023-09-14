import sys

file = sys.argv[1]

with open(file, "r") as f:
    header = f.readline()

    for i, h in enumerate(header.split(",")):
        print(f"{i}: {h}")
