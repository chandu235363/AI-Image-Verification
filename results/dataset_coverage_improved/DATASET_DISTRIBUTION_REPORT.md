# Dataset Distribution Report

- Seed: 20261002
- Per-class target: 2000
- Total prepared images: 6000

## Split and class counts

| Split | Total | Real | AI-Generated | Manipulated |
|---|---:|---:|---:|---:|
| train | 4198 | 1400 | 1400 | 1398 |
| val | 901 | 300 | 300 | 301 |
| test | 901 | 300 | 300 | 301 |

## CASIA Tp parsed group counts

| Group | Train | Validation | Test |
|---|---:|---:|---:|
| Tp_D_CND | 2 | 1 | 1 |
| Tp_D_CNN | 17 | 4 | 4 |
| Tp_D_CRD | 9 | 2 | 2 |
| Tp_D_CRN | 68 | 15 | 15 |
| Tp_D_NND | 9 | 2 | 2 |
| Tp_D_NNN | 104 | 22 | 22 |
| Tp_D_NRD | 22 | 5 | 5 |
| Tp_D_NRN | 275 | 59 | 59 |
| Tp_S_CND | 12 | 2 | 2 |
| Tp_S_CNN | 138 | 30 | 30 |
| Tp_S_CRD | 17 | 3 | 3 |
| Tp_S_CRN | 78 | 16 | 16 |
| Tp_S_NND | 46 | 10 | 10 |
| Tp_S_NNN | 394 | 85 | 85 |
| Tp_S_NRD | 37 | 8 | 8 |
| Tp_S_NRN | 170 | 37 | 37 |

## Raw CASIA filename group tokens

Case variants are retained here for provenance and coalesced in the canonical table above.

| Raw token | Train | Validation | Test |
|---|---:|---:|---:|
| Tp_D_CND | 2 | 1 | 1 |
| Tp_D_CNN | 17 | 4 | 4 |
| Tp_D_CRD | 9 | 2 | 2 |
| Tp_D_CRN | 68 | 15 | 15 |
| Tp_D_NND | 9 | 2 | 2 |
| Tp_D_NNN | 104 | 22 | 22 |
| Tp_D_NRD | 22 | 5 | 5 |
| Tp_D_NRN | 275 | 59 | 59 |
| Tp_S_CND | 12 | 2 | 2 |
| Tp_S_CNN | 137 | 30 | 30 |
| Tp_S_CRD | 17 | 3 | 3 |
| Tp_S_CRN | 78 | 16 | 16 |
| Tp_S_CnN | 1 | 0 | 0 |
| Tp_S_NND | 46 | 10 | 10 |
| Tp_S_NNN | 394 | 85 | 85 |
| Tp_S_NRD | 37 | 8 | 8 |
| Tp_S_NRN | 170 | 37 | 37 |

## Source formats

| Format | Count |
|---|---:|
| jpg | 4803 |
| tif | 1197 |

| Class/source | jpg | tif |
|---|---:|---:|
| Real | 2000 | 0 |
| AI-Generated | 2000 | 0 |
| Manipulated | 803 | 1197 |

Selection strategy: seeded random Real/AI sample; CASIA Tp proportional allocation by parsed filename group with up to 3-item floor; deterministic within-group sample; per-group 70/15/15 split; case-only variants are coalesced in canonical group reporting while raw filename tokens are retained.
Raw sources were read only; prepared images are stored under this report directory's dataset folder.

## Available CASIA Tp source groups

Programmatic filename inspection found 5,123 readable files across 16 case-normalized groups (0 unreadable supported-format files). Counts below precede the diagnostic-only path exclusions.

| Group | Available | Selected | Test |
|---|---:|---:|---:|
| Tp_D_CND | 6 | 4 | 1 |
| Tp_D_CNN | 59 | 25 | 4 |
| Tp_D_CRD | 29 | 13 | 2 |
| Tp_D_CRN | 251 | 98 | 15 |
| Tp_D_NND | 29 | 13 | 2 |
| Tp_D_NNN | 380 | 148 | 22 |
| Tp_D_NRD | 79 | 32 | 5 |
| Tp_D_NRN | 1,016 | 393 | 59 |
| Tp_S_CND | 36 | 16 | 2 |
| Tp_S_CNN | 509 | 198 | 30 |
| Tp_S_CRD | 54 | 23 | 3 |
| Tp_S_CRN | 281 | 110 | 16 |
| Tp_S_NND | 167 | 66 | 10 |
| Tp_S_NNN | 1,463 | 564 | 85 |
| Tp_S_NRD | 132 | 53 | 8 |
| Tp_S_NRN | 632 | 244 | 37 |
