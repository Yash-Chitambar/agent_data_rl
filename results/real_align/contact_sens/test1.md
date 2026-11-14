| Policy | Place | Pads | Plug | Variant | In training | n | Seat % [95 % CI] | Full depth % | Seat time p50 / p99 s | In after release | Failures |
|---|---|---|---|---|---|---|---|---|---|---|---|
| K4 | box | 4x1.2 | nom | base: training contact (port mu 0.10-0.20, port solref 3-4 ms) | yes | 1024 | 74.3 [71.6, 76.9] | 68.7 | 1.42 / 6.54 | 670/803 | lateral_miss 202, grasp_slip 33, full_no_seat_test 18, tilt 9, jam 1 |
| K4 | box | 4x1.2 | nom | pmu_lo: port mu 0.05-0.08 | no | 1024 | 81.5 [79.1, 83.8] | 72.8 | 1.32 / 5.16 | 715/869 | lateral_miss 141, full_no_seat_test 23, grasp_slip 19, tilt 6 |
| K4 | box | 4x1.2 | nom | pmu_hi: port mu 0.35-0.60 | no | 1024 | 23.1 [20.7, 25.8] | 22.0 | 1.92 / 8.42 | 218/294 | lateral_miss 657, grasp_slip 107, tilt 18, jam 3, full_no_seat_test 2 |
| K4 | box | 4x1.2 | nom | pmu_vhi: port mu 0.80-1.00 | no | 1024 | 2.5 [1.7, 3.7] | 2.4 | 4.26 / 9.55 | 25/38 | lateral_miss 804, grasp_slip 186, tilt 6, jam 2 |
| K4 | real | 4x1.2 | nom | base: training contact (port mu 0.10-0.20, port solref 3-4 ms) | yes | 1024 | 94.6 [93.1, 95.9] | 89.5 | 1.32 / 5.26 | 884/998 | grasp_slip 22, lateral_miss 22, full_no_seat_test 9, deep_unseated 1, tilt 1 |
| K4 | real | 4x1.2 | nom | pmu_lo: port mu 0.05-0.08 | no | 1024 | 98.4 [97.5, 99.0] | 91.1 | 1.24 / 3.52 | 885/1013 | lateral_miss 8, full_no_seat_test 5, grasp_slip 3 |
| K4 | real | 4x1.2 | nom | pmu_hi: port mu 0.35-0.60 | no | 1024 | 33.9 [31.1, 36.8] | 34.6 | 2.12 / 9.14 | 349/410 | lateral_miss 583, grasp_slip 76, full_no_seat_test 9, jam 3, deep_unseated 3, tilt 3 |
| K4 | real | 4x1.2 | nom | pmu_vhi: port mu 0.80-1.00 | no | 1024 | 0.9 [0.5, 1.7] | 0.9 | 2.64 / 3.10 | 9/19 | lateral_miss 755, grasp_slip 260 |
| K4 | real | 4x1.2 | nom | ptc_2ms: port solref 2 ms (stiffer) | no | 1024 | 94.1 [92.5, 95.4] | 87.3 | 1.32 / 6.04 | 864/1002 | grasp_slip 33, lateral_miss 20, full_no_seat_test 7 |
| K4 | real | 4x1.2 | nom | ptc_8ms: port solref 8 ms (softer) | no | 1024 | 76.5 [73.8, 79.0] | 93.8 | 1.66 / 6.25 | 929/995 | full_no_seat_test 209, lateral_miss 23, grasp_slip 7, tilt 1, deep_unseated 1 |
| K4 | real | 4x1.2 | nom | pdamp_05: port damping ratio 0.5 | no | 1024 | 92.8 [91.0, 94.2] | 87.8 | 1.28 / 4.91 | 859/999 | grasp_slip 45, lateral_miss 19, full_no_seat_test 8, jam 1, tilt 1 |
| K4 | real | 4x1.2 | nom | psolimp_soft: port solimp (0.90, 0.95) | no | 1024 | 54.3 [51.2, 57.3] | 95.4 | 1.66 / 6.98 | 966/1002 | full_no_seat_test 441, lateral_miss 16, grasp_slip 10, tilt 1 |
| K4 | real | 4x1.2 | nom | padmu_08: pad mu 0.8 | no | 1024 | 96.0 [94.6, 97.0] | 91.2 | 1.30 / 6.13 | 902/1001 | lateral_miss 22, grasp_slip 15, full_no_seat_test 4 |
| K4 | real | 4x1.2 | nom | padmu_20: pad mu 2.0 | no | 1024 | 88.6 [86.5, 90.4] | 83.2 | 1.30 / 5.62 | 823/945 | grasp_slip 66, lateral_miss 44, full_no_seat_test 6, deep_unseated 1 |
| K4 | real | 4x1.2 | nom | padtc_6ms: pad solref 6 ms | no | 1024 | 86.9 [84.7, 88.8] | 83.6 | 1.28 / 4.51 | 817/947 | lateral_miss 63, grasp_slip 55, full_no_seat_test 16 |
| K4 | real | 4x1.2 | nom | grip_06: squeeze x0.6 | no | 1024 | 83.8 [81.4, 85.9] | 81.6 | 1.32 / 5.16 | 811/924 | lateral_miss 91, grasp_slip 60, full_no_seat_test 14, tilt 1 |
| K4 | real | 4x1.2 | nom | grip_14: squeeze x1.4 | no | 1024 | 95.7 [94.3, 96.8] | 90.9 | 1.32 / 7.07 | 899/991 | lateral_miss 27, grasp_slip 13, tilt 2, full_no_seat_test 2 |
| K4 | real | 4x1.2 | loose | base: training contact (port mu 0.10-0.20, port solref 3-4 ms) | no | 1024 | 45.5 [42.5, 48.6] | 84.9 | 1.40 / 5.50 | 747/1002 | full_no_seat_test 468, grasp_slip 70, lateral_miss 17, tilt 2, deep_unseated 1 |
| K4 | real | 4x1.2 | loose | pmu_hi: port mu 0.35-0.60 | no | 1024 | 18.6 [16.3, 21.1] | 30.9 | 1.85 / 8.49 | 260/414 | lateral_miss 584, full_no_seat_test 140, grasp_slip 102, tilt 5, deep_unseated 2, jam 1 |
| K4 | real | 4x1.2 | tight | base: training contact (port mu 0.10-0.20, port solref 3-4 ms) | no | 1024 | 65.7 [62.8, 68.6] | 64.5 | 2.38 / 8.18 | 654/941 | grasp_slip 285, lateral_miss 46, full_no_seat_test 12, jam 7, tilt 1 |
| K4 | real | 4x1.2 | tight | pmu_hi: port mu 0.35-0.60 | no | 1024 | 8.7 [7.1, 10.6] | 8.6 | 4.16 / 9.92 | 88/202 | lateral_miss 645, grasp_slip 260, jam 21, tilt 8, full_no_seat_test 1 |
| K4 | real | 2x1.2 | nom | base: training contact (port mu 0.10-0.20, port solref 3-4 ms) | no | 1024 | 89.3 [87.2, 91.0] | 85.5 | 1.31 / 5.14 | 848/934 | lateral_miss 82, grasp_slip 21, full_no_seat_test 5, jam 1, tilt 1 |
| K4 | real | 2x1.2 | nom | pmu_lo: port mu 0.05-0.08 | no | 1024 | 92.7 [90.9, 94.1] | 85.7 | 1.24 / 4.25 | 836/952 | lateral_miss 66, grasp_slip 5, full_no_seat_test 3, jam 1 |
| K4 | real | 2x1.2 | nom | pmu_hi: port mu 0.35-0.60 | no | 1024 | 28.3 [25.6, 31.2] | 28.9 | 2.03 / 8.66 | 291/373 | lateral_miss 599, grasp_slip 115, full_no_seat_test 9, tilt 7, deep_unseated 2, jam 2 |
| K4 | real | 6x0.8 | nom | base: training contact (port mu 0.10-0.20, port solref 3-4 ms) | no | 1024 | 96.6 [95.3, 97.5] | 90.8 | 1.32 / 5.78 | 894/1006 | lateral_miss 17, grasp_slip 10, full_no_seat_test 8 |
| K4 | real | 6x0.8 | nom | pmu_lo: port mu 0.05-0.08 | no | 1024 | 98.5 [97.6, 99.1] | 91.9 | 1.24 / 3.88 | 885/1017 | full_no_seat_test 7, lateral_miss 6, grasp_slip 2 |
| K4 | real | 6x0.8 | nom | pmu_hi: port mu 0.35-0.60 | no | 1024 | 34.7 [31.8, 37.6] | 34.9 | 2.18 / 7.98 | 348/402 | lateral_miss 596, grasp_slip 54, full_no_seat_test 8, tilt 7, jam 3, deep_unseated 1 |
| K7 | box | 4x1.2 | nom | base: training contact (port mu 0.10-0.20, port solref 3-4 ms) | yes | 1024 | 98.7 [97.8, 99.3] | 85.9 | 1.22 / 3.95 | 862/1011 | lateral_miss 9, grasp_slip 4 |
| K7 | box | 4x1.2 | nom | pmu_lo: port mu 0.05-0.08 | no | 1024 | 98.7 [97.8, 99.3] | 83.8 | 1.14 / 3.39 | 831/1015 | grasp_slip 7, lateral_miss 5, full_no_seat_test 1 |
| K7 | box | 4x1.2 | nom | pmu_hi: port mu 0.35-0.60 | no | 1024 | 55.6 [52.5, 58.6] | 52.2 | 1.90 / 8.70 | 527/614 | lateral_miss 352, grasp_slip 76, tilt 22, full_no_seat_test 3, jam 2 |
| K7 | box | 4x1.2 | nom | pmu_vhi: port mu 0.80-1.00 | no | 1024 | 8.7 [7.1, 10.6] | 8.5 | 3.38 / 8.43 | 87/118 | lateral_miss 743, grasp_slip 147, tilt 42, deep_unseated 2, full_no_seat_test 1 |
| K7 | box | 4x1.2 | nom | ptc_2ms: port solref 2 ms (stiffer) | no | 1024 | 98.2 [97.2, 98.9] | 83.6 | 1.22 / 4.36 | 842/1008 | lateral_miss 11, grasp_slip 6, full_no_seat_test 1 |
| K7 | box | 4x1.2 | nom | ptc_8ms: port solref 8 ms (softer) | no | 1024 | 86.2 [84.0, 88.2] | 88.7 | 1.40 / 4.81 | 886/1012 | full_no_seat_test 129, lateral_miss 8, grasp_slip 4 |
| K7 | box | 4x1.2 | nom | pdamp_05: port damping ratio 0.5 | no | 1024 | 98.7 [97.8, 99.3] | 84.1 | 1.20 / 3.99 | 845/1015 | lateral_miss 7, grasp_slip 5, full_no_seat_test 1 |
| K7 | box | 4x1.2 | nom | psolimp_soft: port solimp (0.90, 0.95) | no | 1024 | 46.2 [43.2, 49.3] | 88.0 | 1.28 / 3.41 | 896/1008 | full_no_seat_test 501, grasp_slip 40, lateral_miss 10 |
| K7 | box | 4x1.2 | nom | padmu_08: pad mu 0.8 | no | 1024 | 99.0 [98.2, 99.5] | 85.9 | 1.21 / 3.46 | 864/1014 | lateral_miss 9, grasp_slip 1 |
| K7 | box | 4x1.2 | nom | padmu_20: pad mu 2.0 | no | 1024 | 94.3 [92.7, 95.6] | 83.0 | 1.20 / 4.07 | 836/971 | grasp_slip 35, lateral_miss 22, full_no_seat_test 1 |
| K7 | box | 4x1.2 | nom | padtc_6ms: pad solref 6 ms | no | 1024 | 94.6 [93.1, 95.9] | 82.3 | 1.22 / 3.90 | 830/975 | lateral_miss 28, grasp_slip 26, full_no_seat_test 1 |
| K7 | box | 4x1.2 | nom | grip_06: squeeze x0.6 | no | 1024 | 88.5 [86.4, 90.3] | 79.1 | 1.18 / 3.66 | 803/915 | lateral_miss 74, grasp_slip 41, tilt 2, full_no_seat_test 1 |
| K7 | box | 4x1.2 | nom | grip_14: squeeze x1.4 | no | 1024 | 98.9 [98.1, 99.4] | 86.3 | 1.20 / 5.13 | 866/1013 | lateral_miss 10, grasp_slip 1 |
| K7 | box | 4x1.2 | loose | base: training contact (port mu 0.10-0.20, port solref 3-4 ms) | no | 1024 | 38.2 [35.3, 41.2] | 81.7 | 1.18 / 3.42 | 729/1013 | full_no_seat_test 533, grasp_slip 91, lateral_miss 9 |
| K7 | box | 4x1.2 | loose | pmu_hi: port mu 0.35-0.60 | no | 1024 | 26.0 [23.4, 28.7] | 51.4 | 1.60 / 9.18 | 460/641 | lateral_miss 346, full_no_seat_test 287, grasp_slip 112, tilt 13 |
| K7 | box | 4x1.2 | tight | base: training contact (port mu 0.10-0.20, port solref 3-4 ms) | no | 1024 | 90.0 [88.1, 91.7] | 84.8 | 1.38 / 5.12 | 865/1004 | grasp_slip 82, lateral_miss 9, full_no_seat_test 8, jam 2, tilt 1 |
| K7 | box | 4x1.2 | tight | pmu_hi: port mu 0.35-0.60 | no | 1024 | 34.4 [31.5, 37.3] | 34.1 | 2.39 / 9.32 | 349/458 | lateral_miss 420, grasp_slip 166, tilt 60, jam 20, full_no_seat_test 5, deep_unseated 1 |
| K7 | box | 2x1.2 | nom | base: training contact (port mu 0.10-0.20, port solref 3-4 ms) | no | 1024 | 92.6 [90.8, 94.0] | 82.0 | 1.22 / 3.76 | 823/950 | lateral_miss 72, grasp_slip 4 |
| K7 | box | 2x1.2 | nom | pmu_lo: port mu 0.05-0.08 | no | 1024 | 92.6 [90.8, 94.0] | 79.4 | 1.16 / 3.30 | 796/953 | lateral_miss 61, grasp_slip 10, full_no_seat_test 4, tilt 1 |
| K7 | box | 2x1.2 | nom | pmu_hi: port mu 0.35-0.60 | no | 1024 | 49.8 [46.7, 52.9] | 47.3 | 1.94 / 8.63 | 475/575 | lateral_miss 310, grasp_slip 157, tilt 35, full_no_seat_test 5, jam 5, deep_unseated 2 |
| K7 | box | 6x0.8 | nom | base: training contact (port mu 0.10-0.20, port solref 3-4 ms) | no | 1024 | 99.1 [98.3, 99.5] | 85.0 | 1.20 / 4.16 | 859/1016 | lateral_miss 6, grasp_slip 3 |
| K7 | box | 6x0.8 | nom | pmu_lo: port mu 0.05-0.08 | no | 1024 | 98.6 [97.7, 99.2] | 82.8 | 1.16 / 3.24 | 831/1019 | grasp_slip 9, full_no_seat_test 3, lateral_miss 1, tilt 1 |
| K7 | box | 6x0.8 | nom | pmu_hi: port mu 0.35-0.60 | no | 1024 | 52.1 [49.1, 55.2] | 48.7 | 1.78 / 7.65 | 490/567 | lateral_miss 425, grasp_slip 45, tilt 13, full_no_seat_test 5, jam 2 |
| K7 | real | 4x1.2 | nom | base: training contact (port mu 0.10-0.20, port solref 3-4 ms) | yes | 1024 | 98.7 [97.8, 99.3] | 92.0 | 1.10 / 3.91 | 921/1012 | lateral_miss 9, grasp_slip 4 |
| K7 | real | 4x1.2 | nom | pmu_lo: port mu 0.05-0.08 | no | 1024 | 98.3 [97.4, 99.0] | 90.3 | 1.06 / 3.12 | 904/1012 | grasp_slip 7, lateral_miss 6, full_no_seat_test 3, tilt 1 |
| K7 | real | 4x1.2 | nom | pmu_hi: port mu 0.35-0.60 | no | 1024 | 53.0 [50.0, 56.1] | 52.7 | 2.04 / 6.85 | 537/574 | lateral_miss 408, grasp_slip 59, tilt 11, full_no_seat_test 2, jam 1 |
| K7 | real | 4x1.2 | nom | pmu_vhi: port mu 0.80-1.00 | no | 1024 | 7.1 [5.7, 8.9] | 6.7 | 2.72 / 6.74 | 68/84 | lateral_miss 808, grasp_slip 104, tilt 34, full_no_seat_test 2, jam 2, deep_unseated 1 |
| K7 | real | 4x1.2 | nom | ptc_2ms: port solref 2 ms (stiffer) | no | 1024 | 98.5 [97.6, 99.1] | 91.3 | 1.10 / 4.00 | 918/1010 | lateral_miss 9, grasp_slip 6 |
| K7 | real | 4x1.2 | nom | ptc_8ms: port solref 8 ms (softer) | no | 1024 | 88.2 [86.1, 90.0] | 93.1 | 1.46 / 5.58 | 939/1003 | full_no_seat_test 100, lateral_miss 16, grasp_slip 5 |
| K7 | real | 4x1.2 | nom | pdamp_05: port damping ratio 0.5 | no | 1024 | 98.5 [97.6, 99.1] | 92.4 | 1.08 / 4.16 | 923/1015 | lateral_miss 8, full_no_seat_test 5, grasp_slip 2 |
| K7 | real | 4x1.2 | nom | psolimp_soft: port solimp (0.90, 0.95) | no | 1024 | 51.1 [48.0, 54.1] | 93.5 | 1.12 / 3.71 | 945/1010 | full_no_seat_test 468, grasp_slip 24, lateral_miss 9 |
| K7 | real | 4x1.2 | nom | padmu_08: pad mu 0.8 | no | 1024 | 99.0 [98.2, 99.5] | 92.3 | 1.10 / 3.99 | 927/1014 | lateral_miss 9, grasp_slip 1 |
| K7 | real | 4x1.2 | nom | padmu_20: pad mu 2.0 | no | 1024 | 94.9 [93.4, 96.1] | 88.8 | 1.10 / 3.83 | 890/974 | grasp_slip 32, lateral_miss 19, full_no_seat_test 1 |
| K7 | real | 4x1.2 | nom | padtc_6ms: pad solref 6 ms | no | 1024 | 95.6 [94.2, 96.7] | 89.8 | 1.08 / 3.99 | 911/982 | grasp_slip 23, lateral_miss 21, jam 1 |
| K7 | real | 4x1.2 | nom | grip_06: squeeze x0.6 | no | 1024 | 88.9 [86.8, 90.7] | 84.5 | 1.10 / 3.39 | 848/916 | lateral_miss 66, grasp_slip 47, tilt 1 |
| K7 | real | 4x1.2 | nom | grip_14: squeeze x1.4 | no | 1024 | 99.0 [98.2, 99.5] | 93.6 | 1.10 / 3.98 | 939/1014 | lateral_miss 9, grasp_slip 1 |
| K7 | real | 4x1.2 | loose | base: training contact (port mu 0.10-0.20, port solref 3-4 ms) | no | 1024 | 42.2 [39.2, 45.2] | 88.0 | 1.08 / 3.79 | 787/1011 | full_no_seat_test 523, grasp_slip 58, lateral_miss 11 |
| K7 | real | 4x1.2 | loose | pmu_hi: port mu 0.35-0.60 | no | 1024 | 23.8 [21.3, 26.5] | 48.7 | 1.67 / 9.12 | 423/606 | lateral_miss 384, full_no_seat_test 263, grasp_slip 120, tilt 12, jam 1 |
| K7 | real | 4x1.2 | tight | base: training contact (port mu 0.10-0.20, port solref 3-4 ms) | no | 1024 | 92.1 [90.3, 93.6] | 88.8 | 1.20 / 4.39 | 902/1004 | grasp_slip 66, lateral_miss 9, full_no_seat_test 5, jam 1 |
| K7 | real | 4x1.2 | tight | pmu_hi: port mu 0.35-0.60 | no | 1024 | 34.9 [32.0, 37.8] | 34.6 | 2.30 / 9.07 | 354/475 | lateral_miss 434, grasp_slip 179, tilt 39, jam 12, full_no_seat_test 3 |
| K7 | real | 2x1.2 | nom | base: training contact (port mu 0.10-0.20, port solref 3-4 ms) | no | 1024 | 93.4 [91.7, 94.7] | 87.8 | 1.10 / 5.24 | 877/956 | lateral_miss 63, grasp_slip 3, tilt 1, full_no_seat_test 1 |
| K7 | real | 2x1.2 | nom | pmu_lo: port mu 0.05-0.08 | no | 1024 | 92.6 [90.8, 94.0] | 84.5 | 1.06 / 3.38 | 843/954 | lateral_miss 63, grasp_slip 9, full_no_seat_test 4 |
| K7 | real | 2x1.2 | nom | pmu_hi: port mu 0.35-0.60 | no | 1024 | 49.9 [46.8, 53.0] | 49.5 | 2.06 / 7.89 | 504/566 | lateral_miss 326, grasp_slip 139, tilt 36, full_no_seat_test 7, jam 4, deep_unseated 1 |
| K7 | real | 6x0.8 | nom | base: training contact (port mu 0.10-0.20, port solref 3-4 ms) | no | 1024 | 99.7 [99.1, 99.9] | 93.1 | 1.10 / 3.77 | 936/1021 | lateral_miss 2, grasp_slip 1 |
| K7 | real | 6x0.8 | nom | pmu_lo: port mu 0.05-0.08 | no | 1024 | 99.1 [98.3, 99.5] | 89.8 | 1.06 / 3.64 | 904/1020 | grasp_slip 7, lateral_miss 1, full_no_seat_test 1 |
| K7 | real | 6x0.8 | nom | pmu_hi: port mu 0.35-0.60 | no | 1024 | 53.0 [50.0, 56.1] | 52.5 | 1.90 / 7.39 | 532/567 | lateral_miss 426, grasp_slip 41, tilt 11, full_no_seat_test 2, jam 1 |
