# Copyright (c) 2019 Eric Steinberger


"""
A bet-set is a list of pot-fractions resembling legal bet-sizes. Pass these to a DiscretizedPokerEnv.
The provided bet-sets are just examples that make benchmarking easier and helps avoid typing and copy-pasting lists.

Bet-set 56 for LBR is chosen to be similar to the one used in the original LBR paper (https://arxiv.org/abs/1612.07547).
All other bet-sets are just arbitrarily selected to make sense.
"""

ALL_IN_ONLY = [
    100000.0
]

POT_ONLY = [
    1.0,
]

B_2 = [
    1.0,
    100000.0
]

B_3 = [
    0.5, 1.0,
    100000.0
]

B_3_SMALL = [
    0.5, 1.0, 2.0
]

B_4 = [
    0.5, 1.0, 2.0,
    100000.0
]

B_5_SMALL = [
    0.5, 1.0, 2.0, 2.3,
    100000.0
]

B_5 = [
    0.5, 0.75, 1.0, 2.0,
    100000.0
]

B_8 = [
    0.5, 0.7, 1.25, 1.5, 2.0, 2.5,
    100000.0
]

B_16 = [
    0.10, 0.25, 0.40, 0.55, 0.75,
    0.90, 1.10, 1.30, 1.50, 1.80,
    2.25, 3.30, 4.50, 6.00, 8.00,
    100000.0
]

B_21 = [
    0.10, 0.22, 0.33, 0.44, 0.55,
    0.75, 0.88, 1.00, 1.10, 1.25,
    1.40, 1.60, 1.80, 2.00, 2.25,
    2.75, 3.75, 4.00, 5.50, 8.00,
    100000.0
]

B_41 = [
    0.10, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60,
    0.65, 0.70, 0.75, 0.80, 0.85,
    0.90, 1.00, 1.10, 1.20, 1.30,
    1.40, 1.50, 1.60, 1.70, 1.80,
    1.90, 2.00, 2.10, 2.25, 2.45,
    2.60, 2.75, 3.00, 3.30, 3.80,
    4.50, 5.0, 6.00, 7.00, 8.00,
    100000.0
]

B_76 = [
    0.10, 0.15, 0.17, 0.20, 0.23,
    0.26, 0.29, 0.33, 0.37, 0.40,
    0.43, 0.47, 0.50, 0.53, 0.56,
    0.60, 0.63, 0.67, 0.70, 0.73,
    0.78, 0.81, 0.84, 0.87, 0.90,
    0.95, 1.00, 1.05, 1.10, 1.05,
    1.15, 1.20, 1.25, 1.30, 1.35,
    1.40, 1.45, 1.50, 1.55, 1.60,
    1.65, 1.70, 1.75, 1.80, 1.85,
    1.90, 1.95, 2.00, 2.05, 2.10,
    2.15, 2.20, 2.30, 2.40, 2.50,
    2.60, 2.70, 2.80, 2.90, 3.00,
    3.10, 3.20, 3.30, 3.50, 3.70,
    4.00, 4.30, 4.60, 5.00, 5.50,
    6.00, 6.50, 7.00, 7.50, 8.00,
    100000.0
]

PL_6 = [
    0.1, 0.22, 0.3, 0.50, 0.73,
    1.0,
]

PL_10 = [
    0.1, 0.15, 0.22, 0.3, 0.39,
    0.50, 0.61, 0.73, 0.86, 1.0,
]

# ________________________ OFF-TREE BET-SETS FOR EVALUATION (off-tree w.r.t. bet-sets above) ___________________________
OFF_TREE_1 = [
    0.7,
    100000.0
]

OFF_TREE_5 = [
    0.38, 0.63, 0.93, 1.73,
    100000.0
]

OFF_TREE_11 = [
    0.20, 0.42, 0.52, 0.86, 1.23,
    1.65, 2.05, 3.40, 5.00, 7.0,
    100000.0
]

OFF_TREE_56 = [
    0.10, 0.15, 0.17, 0.20, 0.23,
    0.25, 0.27, 0.30, 0.33, 0.36,
    0.39, 0.42, 0.45, 0.48, 0.51,
    0.54, 0.57, 0.60, 0.64, 0.68,
    0.72, 0.76, 0.80, 0.86, 0.92,
    0.98, 1.04, 1.10, 1.16, 1.23,
    1.30, 1.37, 1.44, 1.51, 1.58,
    1.65, 1.74, 1.84, 1.94, 2.05,
    2.20, 2.35, 2.50, 2.65, 2.80,
    2.95, 3.10, 3.30, 3.50, 3.80,
    4.20, 5.10, 6.10, 7.10, 8.10,
    100000.0
]

OFF_TREE_PL_20 = [
    0.05, 0.11, 0.16, 0.21, 0.26,
    0.31, 0.36, 0.41, 0.46, 0.51,
    0.56, 0.61, 0.66, 0.71, 0.76,
    0.81, 0.86, 0.91, 0.95, 1.0,
]
