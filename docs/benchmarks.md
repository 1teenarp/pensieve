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
