# Batched release probe, K4 (sim, scripts/eval/k8_release_probe.py)

Start cell real_ho, rig spot, real-like plant, seed 8700, n = 1024 worlds per row. Counts are over the worlds fully in at rest whose release finished.

| Plug | Port mu | Jaw target | sim seat | full at rest | release done | in after open | in after retreat | in at end | out % | finger at retreat mm p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| loose | 0.10-0.20 | past full open | 429/1024 | 902 | 893 | 781 | 781 | 781 | 12.5 | 26.96 |
| loose | 0.30 | past full open | 397/1024 | 767 | 708 | 547 | 539 | 535 | 24.4 | 26.96 |
| loose | 0.10-0.20 | 0.024 (chain) | 442/1024 | 902 | 896 | 791 | 458 | 457 | 49.0 | 23.88 |
| loose | 0.30 | 0.024 (chain) | 386/1024 | 753 | 701 | 537 | 391 | 391 | 44.2 | 23.88 |
| nom | 0.10-0.20 | past full open | 981/1024 | 940 | 925 | 898 | 897 | 897 | 3.0 |  |
| nom | 0.30 | past full open | 810/1024 | 793 | 715 | 707 | 707 | 707 | 1.1 |  |
| nom | 0.10-0.20 | 0.024 (chain) | 982/1024 | 945 | 933 | 906 | 649 | 649 | 30.4 | 23.88 |
| nom | 0.30 | 0.024 (chain) | 776/1024 | 770 | 711 | 701 | 659 | 659 | 7.3 | 23.88 |
| tight | 0.10-0.20 | past full open | 702/1024 | 701 | 645 | 636 | 635 | 635 | 1.6 |  |
| tight | 0.30 | past full open | 409/1024 | 407 | 264 | 262 | 262 | 262 | 0.8 |  |
| tight | 0.10-0.20 | 0.024 (chain) | 711/1024 | 689 | 644 | 638 | 600 | 600 | 6.8 | 23.88 |
| tight | 0.30 | 0.024 (chain) | 410/1024 | 420 | 265 | 260 | 258 | 258 | 2.6 | 23.88 |
