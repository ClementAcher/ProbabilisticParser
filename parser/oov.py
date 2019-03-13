import numpy as np


def levenshtein_distance(s1, s2):
    """
    Compute the Levenshtein distance between two strings

    Args:
    - s1, s2 (str)

    Returns:
    - distance (int)
    """

    d = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)
    cost = 0

    for i in range(len(s1) + 1):
        d[i, 0] = i
    for j in range(len(s2) + 1):
        d[0, j] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + cost)
    return d[-1, -1]
