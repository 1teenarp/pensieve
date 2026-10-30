# Pensieve

A homelab AI platform on one EPYC server ("pensive"): a lakehouse, streaming, an ML platform, and
deep learning from first principles up to 7B models, built phase by phase from a series of tutorials.

The name comes from J.K. Rowling's Pensieve, the basin that holds memories so they can be revisited.
The platform is the vessel, the lakehouse is the pool of memories (the `pensieve.lake` package), and
the models draw on it to learn, recall and decide.

## The tutorials (the source of truth)

| # | Tutorial | Phases | Markdown copy |
| --- | --- | --- | --- |
| 0 | Set up Pensieve: host, repo and working agreement | Steps 1-4 | docs/tutorials/0-setup.md |
| 1 | Build a Spark lakehouse on one box | 1-10 | docs/tutorials/1-lakehouse.md |
| 2 | Streaming and MLOps on the lakehouse | 11-21 | docs/tutorials/2-streaming-mlops.md |
| 3 | Deep learning from neurons to 7B models | 22-35 | docs/tutorials/3-deep-learning.md |

Every phase has the same parts: Brief (in tutorial 3), Build, Verify, Commit, and optional Explore.
Tutorial 0 must be complete before any other phase starts.

## Where things are

| Folder | What lives there | Tutorials |
| --- | --- | --- |
| `platform/` | Compose stacks, service config, one folder per Docker image | all |
| `pipelines/` | Spark jobs, Airflow DAGs, event contracts, stream tools | 1, 2 |
| `libs/pensieve/` | Shared Python package (`pensieve.lake`: Spark session, Iceberg catalog) | all |
| `ml/` | Training code, configs, sweeps, registry and serving helpers | 2 |
| `dl/` | Post-training, decision models, computer use, from-scratch models, kernels | 3 |
| `docs/tutorials/` | Markdown copies of the tutorials | all |
| `docs/journal/` | One entry per phase: what was built, verify results, deviations, findings | all |

Versions are pinned in `versions.env`. Secrets and machine paths go in `.env` (copy `.env.example`).
Datasets, checkpoints, model weights and screen recordings live outside the repo.

## Running it

    make up                  # core: storage, catalog, Spark, Trino, sample source DB
    make up-with S=airflow   # add optional stacks: notebook, airflow
    make up-stream           # tutorial 2, streaming
    make up-ml               # tutorial 2, ML platform
    make up-dl               # tutorial 3, GPU workstation
    ./dc ps                  # any docker compose command, with env loaded

## Catching up on the project

1. `git log --oneline --decorate` shows one merge and one tag per phase (`t1-p03` is tutorial 1, phase 3).
2. `docs/journal/<tag>.md` says what was built, the verify numbers, and anything that differed from the tutorial.
3. The pull request for that phase holds the diff and the review discussion.

## How work happens

One phase is one branch (`phase/t1-p03`), one pull request and one tag. Whoever builds it, a person or
a coding agent, follows the tutorial's Build steps, runs every Verify check, writes the journal entry
and opens the pull request. The owner reviews, merges and tags. Agents follow `AGENTS.md`
(`CLAUDE.md` links to it). Host-level and destructive steps are always done by the owner.
