# Data contract

Every fact below was observed on 2026-09-30 by reading `../MTK_dataset`, an exploration notebook and a bounded probe of raw packets. Items marked **Assumed** or **Open** were not verified. Nothing in the raw dataset is modified.

## Sources

| Item | Location |
|---|---|
| Collection A (6 GHz / 5 GHz, 160 MHz) | `../MTK_dataset/CSI_handoff_20260903_123743/data/<date>/<session>/` (dates 2026-08-25, 08-26, 08-28, `handoff_0901` = 2026-09-01) |
| Collection B (5 GHz, 80 MHz) | `../MTK_dataset/OneDrive_2026-09-29/5G 80 MHz/<date>/<session>/` (2026-09-11, 09-14, 09-17) |
| Vendor manifest with `split` column | `CSI_handoff_20260903_123743/dataset_manifest.csv` (10 sessions only) |
| Dataset description (English translation) | `docs/DATASET.md` in this repository |

The raw folders total 59 GB. Two zip files sit beside them and are not used.

## Session inventory

28 session folders: 13 in Collection A, 15 in Collection B. 22 are annotated. Six sessions from 2026-09-14 have no annotations (`1p_1`, `1p_2`, `3p_kid_3`, `3p_kid_5`, `3p_kid_6`, `door`), and `2026-09-14/3p_kid_2` has 9 s of annotation. Each session folder holds:

- `csi_*.bin`: raw TLV frames, several files per session;
- `meta.json`, `timestamps.csv`: one row per CSI record (`file, offset, chip_us, epoch, iso_ms, rssi, snr, bw, tx, rx`);
- `annotations_v2.json` (ground truth), `annotations_v2.csv`;
- `video.avi`, `frames.csv`, `video_meta.json` (used for label sanity checks only, never as model input).

## Signal geometry (observed)

- Four links per record group: `(tx, rx)` in `{(0,0), (0,1), (1,0), (1,1)}`. Row counts per link in a session differ by a few dozen records.
- Collection A packets carry **512 subcarriers** (bandwidth code 3, 160 MHz). Collection B packets carry **256 subcarriers** (code 2, 80 MHz). A probe of 300 packets per date found one width per collection, plus about 0.7% packets with 64 subcarriers and code 0 in `2026-08-26/2p_dandr`.
- Rate per link: 46.8 to 87.5 Hz depending on session (about 54 Hz on 2026-08-25, 87 Hz on 08-26 and 08-28, 64 Hz mean on 09-01, 53 Hz on 09-11, 84 Hz on 09-14, 74 Hz on 09-17). Longest gap seen in the probed sessions: 0.23 s.
- The exploration notebook shows RSSI and SNR distributions that differ between collections and overlap between classes.

## Labels (observed)

`annotations_v2.json` holds `annotations` with `label`, `start`, `end` (Unix epoch seconds), `stable`, `adult_count`, `child_count`, `total_count`, `child_type`, `synthetic_child`, `respiration`, `occupants`.

235 segments over 22 sessions. Labeled seconds before exclusions:

| Label | Segments | Seconds | Median segment (s) |
|---|---:|---:|---:|
| empty | 48 | 7,020 | 158 |
| multiple | 29 | 3,868 | 161 |
| single-child | 30 | 3,329 | 135 |
| transition | 122 | 1,187 | 10 |
| single-adult | 6 | 1,088 | 193 |

All non-transition segments are `stable`. Gaps between consecutive segments are 0 to 1 s.

### Target

Positive: `single-child`. Negative: `empty`, `single-adult`, `multiple`. Excluded: `transition`, unannotated time, and every segment with `child_type == "unknown"`.

The dataset holds roughly one long `single-child` segment per session. After the exclusions, the positive class has 19 segments in total across all roles (11 in the pool, 2 in the primary test, 6 in the shift test), so the effective sample size is far smaller than the window count.

### The child is a dummy

`child_type` on `single-child` is `synthetic-infant` (2,672 s) or `unknown` (657 s). No real child appears. The handoff README calls the class "the weak-breathing dummy infant". The breathing rate and amplitude fields in `respiration` are empty in the segments I read. A dummy with the breathing device off is labeled `empty` (190 s).

## Roles after exclusions

Seconds per class after removing `transition` and `unknown`.

| Role | Sessions | Child | Multiple | Adult | Empty | Non-overlapping 5 s windows |
|---|---:|---:|---:|---:|---:|---:|
| Train and validation pool (A: 08-25, 08-26, 09-01) | 11 | 1,638 | 2,290 | 385 | 3,930 | 1,648 |
| Primary test (A: 08-28) | 2 | 390 | 365 | 0 | 770 | 305 |
| Secondary shift test (B: 09-11, 09-17) | 8 | 644 | 652 | 703 | 2,320 | 851 |

Pool by date, in seconds:

| Date | Sessions | Empty | Adult | Child | Multiple |
|---|---:|---:|---:|---:|---:|
| 2026-08-25 | 5 | 2,320 | 200 | 830 | 1,340 |
| 2026-08-26 | 3 | 935 | 185 | 295 | 605 |
| 2026-09-01 | 3 | 675 | 0 | 518 | 345 |

Per-session seconds are recomputed by the data-index stage from the annotation files and become the authoritative table (`cache/<id>/index.csv`).

Session facts that shape the design:

- `single-adult` exists only on 08-25 and 08-26 in the pool: two sessions, 385 s.
- `2026-09-01/2p_kid_no_empty` has no empty class. `2026-08-26/valid` lasts 307 s.
- Three Collection B sessions lose all child time (`09-11/2p_kid_4`, `09-11/3p_kid_2`, `09-17/2p_1`) and become negative-only.
- The primary test contains no `single-adult` seconds, so it cannot measure false alarms on a lone adult.
- Class mix, sample rate and collection all change with the date, so a date-level split confounds them.

## Provenance and manifest gaps

- The vendor manifest covers 10 sessions. The three 2026-09-01 sessions and all Collection B sessions have no `split`. This project assigns them roles by date (see `EVALUATION.md`).
- `MANIFEST.sha256` exists for the Collection A handoff. Hashing the full 59 GB per run is too slow. The cache hashes each session's `timestamps.csv`, `meta.json`, `annotations_v2.json`, and the size and mtime of each `.bin` file.
- The 2026-08-25 `meta.json` has no bandwidth value, yet its packets show code 3 and 512 subcarriers. **Assumed** to be 160 MHz as `dataset.md` states.

## Hazards

1. **The vendor parser fails silently under NumPy 2.** `tools/CSIPacket.py` calls `np.complex_`, removed in NumPy 2. A bare `except: pass` swallows the error, so `csi_packets()` yields no packets. The `CPD_Demo3` environment has NumPy 2.4.6. I reproduced the failure and confirmed that aliasing `np.complex_ = np.complex128` restores parsing. Demo5 reads records itself with `numpy.frombuffer` and checks equality against the vendor parser on sampled packets.
2. **Leakage through overlapping windows.** Windows a second apart share four seconds of signal. Every window carries its `session` and `segment` identifiers, and splits act on sessions.
3. **Confounds.** Date, collection, rate and class mix move together. A model can learn the day instead of the child. The date-level validation folds and the Collection B test measure this, and they cannot separate the causes.
4. **Few positive segments.** 19 positive segments after exclusions. Window-level intervals would overstate certainty.
5. **Filter memory at segment edges.** The 0.1 Hz high-pass has a memory of about 20 s, so windows near an annotation-segment edge partly reflect the neighbouring segment when the signal is read with context (as it is, 5 s each side). Measured: 26% feature change at the edge, 10% at 10 s, 1% at 20 s. The cache stores each window's `edge_distance_s`, and the report has an interior-only table.
6. **Dummy only.** Any result describes the dummy infant.
7. **Video.** Frames verify labels. They are never a model input.

## Validation rules (each becomes an automated check)

- The data index reproduces the notebook's inventory: 28 sessions, 22 labeled, and the class seconds above.
- The `unknown` exclusion removes exactly 657 s of child time and 561 s of `multiple`.
- No eligible window overlaps a transition, an unknown segment, an unlabeled interval, or a label boundary.
- Every window has exactly one session and one segment. No session appears in two roles.
- Own CSI reader equals the vendor parser on at least 200 sampled packets per collection.
- Cache arrays are finite, shaped `(N, 50, 4, 64)`, and match the recorded hashes.

## Open questions

Questions for the data owner that could change the design: the dummy's breathing rate (sets the pass band), whether geometry is constant across days, and what `child_type: unknown` means.
