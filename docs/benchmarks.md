# Benchmarks

Baseline device measurements taken before any Spark workload, so later phases have numbers to
compare against. Tutorial 1's Phase 7 compares Spark stage times against these.

Measured 30 Oct 2026 by the owner with `fio`, sequential write (1M blocks, iodepth 32) and 4k
random read (iodepth 64), `direct=1`. The ZFS numbers come from a throwaway dataset with
`primarycache=metadata`, so the host's 1 TB of RAM does not hide the disk. The exact commands are
in `docs/tutorials/0-setup.md`, Step 1, Explore.

| Device | Where | Sequential write | 4K random read |
| --- | --- | --- | --- |
| 931 GB Samsung 860 SSD, XFS | `/mnt/scratch` (scratch tier) | 533 MB/s | 73.6k IOPS |
| 477 GB PNY SSD, ZFS 16K records | measured on the old `hot` pool; now the `lake` read cache | 72 MB/s sustained | 4.9k IOPS |
| 18 TB WDC HDD, ZFS 1M records | `/lake/lakehouse` (lake tier) | 203 MB/s | 108 IOPS |

## What the numbers decided

- The PNY slows to about 70 MB/s on sustained writes even after a full TRIM — too slow for
  Postgres and Kafka. The `hot` pool on it was dropped: those services moved to the root NVMe
  under `/srv/pensieve` (`PG_ROOT`, `KAFKA_DATA`), and the PNY became an L2ARC read cache for
  `lake`, where it is about 45x faster than the HDD at random reads and losing it costs nothing.
- The root NVMe (Samsung 980 PRO 2TB, ext4 on LVM) holds Postgres, Kafka logs and, from
  Tutorial 3, training checkpoints. No fio number recorded for it yet.
- Scratch (`/mnt/scratch`) is disposable: Spark shuffle and spill land there.

## Spark, Phase 7: executor granularity on the silver rebuild

Timed 2026-11-01 on branch `phase/t1-p07`, small shape (2 workers × 8 cores / 48g, 16-core app
cap). Job: `make submit JOB=transform/trips_silver.py ARGS="--month <m>"` for 2024-01…06 — the
tutorial's suggested benchmark job; each run rebuilds one month's silver partitions
(`overwritePartitions`), so runs are idempotent and comparable. Wall = `make submit` (includes the
~10 s container + JVM boot floor); app duration, task time, GC and shuffle from the history server
REST API (`:18080/api/v1/applications/<id>`).

| Setting | Executors | Wall, 6-mo mean | App duration, mean | Σ task time, mean | GC share | Spill |
| --- | --- | --- | --- | --- | --- | --- |
| A: shape defaults | 4 × 4c / 20g + 3g overhead | 23.5 s | 20.2 s | 62.2 s | 0.8% | 0 B |
| B: 8-core executors | 2 × 8c / 40g + 6g overhead | 22.2 s | 18.9 s | 52.5 s | 1.3% | 0 B |

Per-month wall: A 22.3 / 22.9 / 24.5 / 23.8 / 23.9 / 23.9 s; B 21.3 / 21.4 / 22.9 / 22.3 / 22.8 /
22.4 s (2024-01…06). Shuffle write per month is deterministic: 153.5 / 155.2 / 186.3 / 185.5 /
196.7 / 186.6 MB.

What it says so far:

- At this data size both settings sit on the submit floor; executor granularity moves wall time
  ~5% but task time ~15% in favour of fewer, bigger executors. Re-run when the workload is
  shuffle- or spill-bound, where the trade-off should be sharper.
- GC is negligible at 20–46 GB heaps (0.4–1.1 s per month run, ≤ 1.3% of task time) — headroom
  for the tutorial's 40 g executor suggestion; the GC-time dashboard panel will barely twitch at
  this scale.
- No spill anywhere: the scratch SSD's write rate only matters on much bigger shuffles.

