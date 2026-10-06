#!/usr/bin/env python3
"""Write a small test set for batched Boltz-2 inference: single-sequence YAMLs of DIFFERENT lengths cut from barnase / barstar (PDB 1BRS),
so one bucket holds several distinct proteins with padding. usage: make_inputs.py <out_dir> [n=8]"""
import sys, os

BARNASE = "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"
BARSTAR = "KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDALTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGADITIILS"


def yaml(chains):
    out = ["version: 1", "sequences:"]
    for cid, seq in chains:
        out += ["  - protein:", f"      id: {cid}", f"      sequence: {seq}", "      msa: empty"]
    return "\n".join(out) + "\n"


def main(out_dir, n=8):
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for k in range(n):                                   # complexes of 199, 196, 193, ... tokens: barnase trimmed at its C-terminus, barstar at its N-terminus
        a = BARNASE[: len(BARNASE) - 2 * k]; b = BARSTAR[k:]
        name = f"t{k:02d}_n{len(a) + len(b)}"
        open(os.path.join(out_dir, name + ".yaml"), "w").write(yaml([("A", a), ("B", b)]))
        made.append((name, len(a) + len(b)))
    for name, ntok in made:
        print(name, ntok)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
