"""
Amino acid substitution rate matrix from the genetic code + transition/transversion bias.

    W = aa_substitution_matrix(kappa=2.0)  # human/E. coli default
    W[i, j] = P(mutate from AA i to AA j | single-nucleotide mutation occurs)
"""
import numpy as np
from Bio.Data.CodonTable import unambiguous_dna_by_id

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

_table = unambiguous_dna_by_id[1]
CODON_TABLE = {**_table.forward_table, **{c: "*" for c in _table.stop_codons}}
PURINES = {"A", "G"}


def _is_transition(a, b):
    return {a, b} <= PURINES or {a, b} <= {"C", "T"}


def _codons_for_aa():
    d = {}
    for codon, aa in CODON_TABLE.items():
        if aa != "*":
            d.setdefault(aa, []).append(codon)
    return d


def aa_substitution_matrix(kappa=2.0, normalize=True):
    """20x20 substitution rate matrix. kappa = transition/transversion ratio."""
    codons = _codons_for_aa()
    W = np.zeros((20, 20))
    for aa_from in AMINO_ACIDS:
        i = AA_TO_IDX[aa_from]
        cw = 1.0 / len(codons[aa_from])
        for cf in codons[aa_from]:
            for pos in range(3):
                for nt in "ACGT":
                    if nt == cf[pos]:
                        continue
                    ct = cf[:pos] + nt + cf[pos + 1:]
                    aa_to = CODON_TABLE[ct]
                    if aa_to == "*" or aa_to == aa_from:
                        continue
                    W[i, AA_TO_IDX[aa_to]] += cw * (kappa if _is_transition(cf[pos], nt) else 1.0)
    if normalize:
        s = W.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        W /= s
    return W


def accessible_substitutions(source_aa):
    """Set of AAs reachable from source_aa by single-nucleotide mutation."""
    result = set()
    for cf in _codons_for_aa()[source_aa]:
        for pos in range(3):
            for nt in "ACGT":
                if nt == cf[pos]:
                    continue
                aa = CODON_TABLE[cf[:pos] + nt + cf[pos + 1:]]
                if aa != "*" and aa != source_aa:
                    result.add(aa)
    return result


if __name__ == "__main__":
    W = aa_substitution_matrix(kappa=1.0, normalize=False)
    n_acc = int(np.sum(W > 0))
    print(f"Accessible AA pairs: {n_acc}/{20*19} ({100*n_acc/(20*19):.1f}%)")
    for aa in AMINO_ACIDS:
        nb = sorted(accessible_substitutions(aa))
        print(f"  {aa}: {len(nb):2d}  {','.join(nb)}")
