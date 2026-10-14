#!/usr/bin/env python3
"""Write a small test set for batched Boltz-2 inference: single-sequence YAMLs of DIFFERENT lengths cut from barnase / barstar (PDB 1BRS),
so one bucket holds several distinct proteins with padding. usage: make_inputs.py <out_dir> [n=8] [1brs|top7]"""
import sys, os

BARNASE = "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"
BARSTAR = "KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDALTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGADITIILS"


def yaml(chains):
    out = ["version: 1", "sequences:"]
    for cid, seq in chains:
        out += ["  - protein:", f"      id: {cid}", f"      sequence: {seq}", "      msa: empty"]
    return "\n".join(out) + "\n"


TOP7 = "DIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQLEGGSLEHHHHHH"   # PDB 1QYS: a de novo designed fold, predicted confidently without an MSA


def main(out_dir, n=8, kind="1brs"):
    os.makedirs(out_dir, exist_ok=True)
    made = []
    if kind == "top7":                                   # monomers of 106, 105, ... residues: the C-terminal tag trimmed one residue at a time
        for k in range(n):
            a = TOP7[: len(TOP7) - k]
            name = f"top7_{k:02d}_n{len(a)}"
            open(os.path.join(out_dir, name + ".yaml"), "w").write(yaml([("A", a)]))
            print(name, len(a))
        return
    for k in range(n):                                   # complexes of 199, 196, 193, ... tokens: barnase trimmed at its C-terminus, barstar at its N-terminus
        a = BARNASE[: len(BARNASE) - 2 * k]; b = BARSTAR[k:]
        name = f"t{k:02d}_n{len(a) + len(b)}"
        open(os.path.join(out_dir, name + ".yaml"), "w").write(yaml([("A", a), ("B", b)]))
        made.append((name, len(a) + len(b)))
    for name, ntok in made:
        print(name, ntok)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8, sys.argv[3] if len(sys.argv) > 3 else "1brs")
