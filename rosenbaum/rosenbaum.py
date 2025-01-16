import numpy as np
from scipy.special import comb
from collections import Counter
from graph_tool.all import Graph, max_cardinality_matching
from math import comb, factorial


def cross_match_test(groups, matching):


    n = sum(1 for g in groups if g == "A")
    N = len(groups)  # Total number of samples (must be even)
    I = N // 2
    
    pairs = [(groups[i], groups[j]) for (i, j) in matching]
    normalized_pairs = [tuple(sorted(pair)) for pair in pairs]
    pair_counts = Counter(normalized_pairs)
    
    a1 = pair_counts.get(("A", "B"), 0)

    p_value = 0
    for A1 in range(a1 + 1):  # For A1 <= a1
        if (n - A1) % 2 != 0:
            continue

        if (I % 2) != (((n + A1) / 2) % 2):
            continue

        A2 = (n - A1) / 2
        A0 = I - (n + A1) / 2 # Remaining pairs are B-B

        if A0 < 0 or A2 < 0:
            continue  # Skip invalid configurations

        numerator = np.power(2, A1) * factorial(I)
        denominator = comb(N, n) * factorial(A0) * factorial(A1) * factorial(A2)

        p_value += numerator / denominator

    return p_value, A1
