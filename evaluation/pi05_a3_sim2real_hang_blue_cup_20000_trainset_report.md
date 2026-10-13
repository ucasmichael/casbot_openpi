# pi05_a3_sim2real_hang_blue_cup checkpoint evaluation

- Frames: 5089 (all 10 training episodes)
- Checkpoint: `pi05_a3_sim2real_hang_blue_cup_h50/20000`
- Prediction compared: `policy.infer(obs)["actions"][0]` against the same frame `action`
- `value_rel`: `abs(pred-gt)/abs(gt)`; `n/a` when `|gt| < 1e-3`
- `range_rel`: absolute error divided by checkpoint action `q99-q01`; `n/a` for constant dimensions

|#| dimension | gt mean | pred mean | abs mean | abs p95 | abs max | value rel % | range rel % |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `left_shoulder_pitch_joint` | 0.00000 | -0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a | n/a |
| 1 | `left_shoulder_roll_joint` | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a | n/a |
| 2 | `left_shoulder_yaw_joint` | 0.00000 | -0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a | n/a |
| 3 | `left_elbow_joint` | 0.00000 | -0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a | n/a |
| 4 | `left_wrist_roll_joint` | 0.00000 | -0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a | n/a |
| 5 | `left_wrist_pitch_joint` | 0.00000 | -0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a | n/a |
| 6 | `left_wrist_yaw_joint` | 0.00000 | -0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a | n/a |
| 7 | `right_shoulder_pitch_joint` | -0.53123 | -0.53399 | 0.00405 | 0.00999 | 0.03094 | 0.80312 | 0.63961 |
| 8 | `right_shoulder_roll_joint` | -0.34402 | -0.34370 | 0.00402 | 0.01016 | 0.03225 | 3.56605 | 0.58733 |
| 9 | `right_shoulder_yaw_joint` | 0.27436 | 0.27544 | 0.00437 | 0.01138 | 0.04908 | 3.47661 | 0.52359 |
| 10 | `right_elbow_joint` | 0.53846 | 0.54306 | 0.00984 | 0.02412 | 0.05155 | 8.84279 | 0.51366 |
| 11 | `right_wrist_roll_joint` | -0.74207 | -0.74797 | 0.00814 | 0.01956 | 0.05741 | 1.15243 | 0.62107 |
| 12 | `right_wrist_pitch_joint` | -0.36948 | -0.36728 | 0.00802 | 0.02028 | 0.06092 | 5.96265 | 0.64678 |
| 13 | `right_wrist_yaw_joint` | -0.18749 | -0.18156 | 0.01174 | 0.03312 | 0.08998 | 10.53934 | 0.53149 |
| 14 | `left_thumb_rotation_0` | 3988.99976 | 3988.99976 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a |
| 15 | `left_thumb_wiggles_1` | 3975.00024 | 3975.00024 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a |
| 16 | `left_thumb_bent_2` | 4092.00000 | 4092.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a |
| 17 | `left_index_wiggles_0` | 4095.00024 | 4095.00024 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a |
| 18 | `left_index_bent_1` | 4068.99976 | 4068.99976 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a |
| 19 | `left_middle_bent` | 4068.99976 | 4068.99976 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a |
| 20 | `left_ring_wiggles_0` | 4095.00024 | 4095.00024 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a |
| 21 | `left_ring_bent_1` | 4068.99976 | 4068.99976 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a |
| 22 | `left_pinky_wiggles_0` | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a | n/a |
| 23 | `left_pinky_bent_1` | 4068.99976 | 4068.99976 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | n/a |
| 24 | `right_thumb_rotation_0` | 4051.26831 | 4052.26416 | 3.41164 | 10.56016 | 81.89282 | 0.08607 | 0.54583 |
| 25 | `right_thumb_wiggles_1` | 2833.13916 | 2842.56958 | 24.20471 | 61.16045 | 443.48792 | 10.34951 | 0.64145 |
| 26 | `right_thumb_bent_2` | 2306.39160 | 2309.38013 | 8.59228 | 23.20669 | 154.28210 | 0.37627 | 0.62407 |
| 27 | `right_index_wiggles_0` | 1545.49500 | 1535.98413 | 28.40496 | 72.04488 | 269.24487 | 14.78782 | 0.80982 |
| 28 | `right_index_bent_1` | 3618.74902 | 3620.12744 | 6.17805 | 16.06680 | 51.21338 | 0.17328 | 0.69196 |
| 29 | `right_middle_bent` | 3496.23413 | 3495.64111 | 6.05104 | 15.93545 | 56.58594 | 0.17394 | 0.61201 |
| 30 | `right_ring_wiggles_0` | 981.25702 | 981.70270 | 15.22327 | 40.62483 | 137.34570 | 3.90141 | 0.70736 |
| 31 | `right_ring_bent_1` | 3369.28833 | 3370.84619 | 6.06191 | 15.59189 | 58.12354 | 0.17951 | 0.63847 |
| 32 | `right_pinky_wiggles_0` | 3542.14893 | 3563.38452 | 26.20132 | 63.16476 | 320.85620 | 1.02734 | 0.81353 |
| 33 | `right_pinky_bent_1` | 2752.86353 | 2751.60815 | 6.22243 | 17.25513 | 59.27319 | 0.22199 | 0.76826 |
| 34 | `waist_roll` | 0.00050 | 0.00033 | 0.00039 | 0.00094 | 0.01167 | 17.99624 | 0.35746 |
| 35 | `waist_pitch` | 0.00012 | 0.00013 | 0.00028 | 0.00081 | 0.00784 | 16.07890 | 0.30042 |
| 36 | `waist_yaw` | -0.00009 | -0.00016 | 0.00024 | 0.00062 | 0.00971 | 14.36557 | 0.30627 |

## Safety audit

- All 5089x37 predictions are finite.
- Predicted hand values outside `[0,4095]`: 7,841 values across 4,142 frames. Most are tiny negative/over-4095 extrapolations, but maxima reach -118.82 and 4,224.73.
- Predicted hand values are not integer actuator counts: 50,797 values differ from nearest integer by >1e-3.
- Arm hard-limit violations in the first action: 135 values across 134 frames (right shoulder roll 90, right elbow 2, right wrist yaw 43); waist hard-limit violations: 0.
- The policy output is still in OpenPI training order (arms, hands, waist roll/pitch/yaw), not A3 wire order (waist yaw/roll/pitch, arms, hands).
- To form a wire command, reorder as `[training[36], training[34], training[35], training[0:34]]`; then mask inactive sides, clip/round hand counts, and apply actor safety limits.
- The converter writes the same absolute qpose to `observation.state` and `action`; this is a training-set, same-frame/teacher-forced check, not a held-out generalization score.
