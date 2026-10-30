# Set up Pensieve: Host, Repo and Working Agreement

Oct 29, 2026 · @Praneet

## Start here

Tutorial 0 is the prerequisite for every other tutorial. When it is done, the host is ready, the `pensieve` repo exists on GitHub with its working agreement, and a coding agent can build any later phase as a pull request you review. Tutorials 1, 2 and 3 then cover only their own phases.

**Who does what.** You (the owner) do anything that touches the host outside Docker or the GitHub account. The agent does everything inside the repo. That split is the same in every later tutorial.

| Step | Who | What you have afterwards |
| --- | --- | --- |
| 1. Prepare the host | Owner | Docker, tools, NUMA layout, storage tiers mounted |
| 2. Create and seed the repo | Owner | GitHub repo with README, AGENTS.md, templates and the tutorials as Markdown |
| 3. Build the repo skeleton | Agent, owner reviews | versions.env, ./dc, Makefile, env template, shapes, pre-commit; tag `t0` |
| 4. Run phases with the harness | Both | The loop you repeat for every phase from `t1-p01` to `t3-p35` |

**Files you need.** Four starter files come with this tutorial: `README.md`, `AGENTS.md`, `docs/journal/TEMPLATE.md` and `.github/pull_request_template.md`. Keep them next to you for Step 2. Everything else is written in Step 3.

**How Tutorial 0 is checked.** Steps 1 and 2 end with a Verify you run yourself. Step 3 is a normal phase: tag `t0`, branch `phase/t0`, journal entry `docs/journal/t0.md`, one pull request.

## Repo layout and conventions

All tutorials build into one repository, `pensieve`, grouped by domain rather than by file type. One repo suits a single shared platform: one `.env`, one Docker network, one catalog. A single commit pins infrastructure, jobs and training code together, which is what makes runs reproducible later.

```plain
pensieve/
  README.md                 # what this is and how to read it
  AGENTS.md                 # working agreement for coding agents (CLAUDE.md links here)
  versions.env              # every pinned image tag and tool version (committed)
  .env.example              # secrets and paths (copy to .env, never commit .env)
  shapes/                   # cluster sizes: small.env, large.env, numa.env
  dc                        # ./dc <compose command>: loads env, selects stacks
  Makefile                  # entry point; includes make/*.mk
  make/                     # core.mk here; stream.mk, ml.mk, dl.mk in later tutorials
  .pre-commit-config.yaml   # secret scanning, notebook output stripping, lint
  .github/                  # PR template; workflows/ with path-filtered CI (Tutorial 1, Phase 10)
  platform/                 # everything that runs as a service
    compose.<stack>.yaml    # one file per stack: storage, catalog, spark, query, sources, ...
    conf/                   # service config + templates (rendered files are gitignored)
    docker/                 # one folder per image, each with a pinned requirements.lock
  pipelines/                # Spark jobs (ingest/, transform/, maintain/), Airflow DAGs
  libs/pensieve/            # shared package; pensieve.lake = Spark session, Iceberg catalog
  ml/                       # Tutorial 2: training, configs, registry, serving
  dl/                       # Tutorial 3: post-training, decision models, from-scratch models, kernels
  notebooks/                # committed without outputs
  scripts/                  # bootstrap, downloads, generators
  docs/
    tutorials/              # every tutorial, exported as Markdown
    journal/                # one entry per phase: results, deviations, findings
    benchmarks.md, upgrades.md
```

These rules keep it manageable as it grows:

| Rule | Why |
| --- | --- |
| Services live in `platform/`, data code in `pipelines/`, shared Python in `libs/pensieve/`; Tutorials 2 and 3 add `ml/` and `dl/` | You can find, change and upgrade one area without touching the others |
| Paths in Compose snippets are relative to `platform/`; `../` reaches the repo root | One consistent rule; snippets stay short |
| Every image tag and tool version is pinned in `versions.env`; every image installs Python packages from a committed lockfile | A rebuild next year produces the same image |
| All Compose commands go through `./dc`; stacks start selectively with `make up` and `make up-with` | No accidental start of everything; one place loads env files |
| Make targets live in `make/<stack>.mk` | The Makefile stays readable as targets pile up |
| Never commit `.env`, rendered configs, data, checkpoints, weights, notebook outputs or screen recordings | Secrets and private data stay out of history; pre-commit enforces it |
| One phase = one branch `phase/<tag>`, one PR, one journal entry, one tag (`t1-p03` = tutorial 1, phase 3) | You can jump back to any phase's working state and read why it looks the way it does |

When a later phase says "add to the Makefile", the targets go in that stack's `.mk` file. When it names a path like `jobs/ingest/...` or `dags/...`, that folder is under `pipelines/`.

## Step 1 (owner): Prepare the host

**Goal:** Docker and the host tools installed, a NUMA layout chosen, and the storage tiers formatted and mounted. These are the steps `AGENTS.md` forbids an agent to run, so you do them by hand. GPU drivers and the NVIDIA container toolkit come later, in Tutorial 2, also as owner steps.

### Storage tiers

Each kind of data goes to the cheapest device that is fast enough for it, behind a path in `.env`. Only one NVMe is detected while both GPUs are installed, so two old SATA SSDs stand in for the fast tiers for now:

| Tier | Device today | Mounted at | Holds | `.env` variable |
| --- | --- | --- | --- | --- |
| OS | 1.8 TB NVMe (`nvme0n1`), ext4 on LVM | `/` | OS, Docker images; Tutorial 3's training checkpoints under `/srv/pensieve` | `DL_CKPT` |
| hot | 477 GB SATA SSD (`sdc`), ZFS pool `hot` | `/hot` | Postgres (catalog, source DB, Airflow, MLflow), Kafka logs | `PG_ROOT`, `KAFKA_DATA` |
| scratch | 931 GB SATA SSD (`sdb`), XFS | `/mnt/scratch` | Spark shuffle and spill; disposable | `SCRATCH_DIR` |
| lake | 18 TB HDD (`sda`), ZFS pool `lake` | `/lake` | Iceberg tables, landing zone, Spark event logs, stream checkpoints, training datasets | `DATA_ROOT`, `DL_DATA` |
| trunk | 18 TB HDD (`sdd`), ZFS pool `trunk` | `/trunk` | Model weights (the existing `/trunk/ai` library) | `MODEL_CACHE` |

No tier is redundant yet. That is acceptable while the lake holds sample data you can regenerate and models you can re-download.

**The ideal setup, once more NVMe works** (check the board manual for M.2 slots that share lanes with a GPU slot, or add a 4-slot M.2 card in a free x16 slot set to x4x4x4x4 in the BIOS):

| Tier | Ideal hardware | Move from today's setup |
| --- | --- | --- |
| OS | The current NVMe | Nothing; training checkpoints move to scratch |
| hot | 2 NVMe drives, ZFS mirror | `zpool attach hot <sdc> <nvme-a>`, wait for the resilver, `zpool detach hot <sdc>`, then attach `<nvme-b>` as the mirror. Paths don't change. |
| scratch | 1 dedicated 4 TB NVMe, XFS | Format it, point the `/mnt/scratch` line in `/etc/fstab` at it. Paths don't change. |
| lake | 3 or more 18 TB HDDs as RAID-Z1 or mirrors, plus a mirrored SSD pair as a special vdev for metadata | Short term: `zpool attach lake <sda> <new-hdd>` makes it a mirror. RAID-Z means a new pool and a `zfs send` |
| trunk | A second 18 TB HDD as its mirror | `zpool attach trunk <sdd> <new-hdd>` |

Because every service reads its location from `.env`, moving to the ideal setup touches the host and `.env` only, never the repo.

### Build

1. **Install Docker Engine and the Compose plugin** from Docker's official apt repository. You need Compose 2.20 or newer. Add yourself to the `docker` group and log back in.
2. **Install the host tools:** `git`, `gh` (GitHub CLI), `uv`, `pre-commit`, `gettext-base` (for `envsubst`), `jq` and `numactl`. Run `gh auth login` so you and the agent can push branches and open pull requests.
3. **Pick a NUMA layout in the BIOS.** Set NPS (nodes per socket) to 4 and leave "L3 as NUMA" (ACPI SRAT L3 Cache as NUMA Domain) off. Leave memory interleaving and the other Memory Addressing options on Auto. Nothing needs NPS4 until Tutorial 1's Phase 7, but changing it later means a reboot in the middle of experiments.
4. **Find the two SATA SSDs by their stable names.** Everything below erases them, so check the sizes match: about 931 GB for scratch (`sdb` today) and 477 GB for hot (`sdc`).

```bash
lsblk -o NAME,SIZE,MODEL,MOUNTPOINT
ls -l /dev/disk/by-id/ | grep -E 'sd[bc]$'      # copy the two ata-... names
SCRATCH_DEV=/dev/disk/by-id/ata-<931GB-ssd>
HOT_DEV=/dev/disk/by-id/ata-<477GB-ssd>
```

5. **Scratch: XFS on the 931 GB SSD** for Spark shuffle and spill.

```bash
sudo wipefs -a ${SCRATCH_DEV}-part* $SCRATCH_DEV    # removes old partitions and signatures
sudo mkfs.xfs -f $SCRATCH_DEV
sudo mkdir -p /mnt/scratch
echo "UUID=$(sudo blkid -s UUID -o value $SCRATCH_DEV) /mnt/scratch xfs noatime,nofail 0 0" | sudo tee -a /etc/fstab
sudo systemctl daemon-reload && sudo mount -a
sudo mkdir -p /mnt/scratch/spark && sudo chown 185:185 /mnt/scratch/spark   # 185 = spark user in the image
```

6. **Hot: a ZFS pool on the 477 GB SSD** for Postgres and Kafka. Small records suit Postgres's 8 KB pages; `autotrim` keeps the SSD fast.

```bash
findmnt /hot; ls -A /hot 2>/dev/null              # both must print nothing
sudo wipefs -a ${HOT_DEV}-part* $HOT_DEV
sudo zpool create -f -o ashift=12 -o autotrim=on \
  -O compression=lz4 -O atime=off -O xattr=sa -O mountpoint=/hot hot $HOT_DEV
sudo zfs create -o recordsize=16K hot/lakehouse
```

The pool is called `hot`, not `buffer`, so it can't collide with the old `buffer` pool if the missing NVMe comes back, or with a leftover `/buffer` folder on the root drive.

7. **Lake: a ZFS pool on the empty 18 TB HDD**, kept apart from `trunk` so a failure in one can't take the other with it. Large records suit the big files object stores write.

```bash
sudo wipefs -n /dev/sda                           # must print nothing
sudo zpool create -o ashift=12 -O compression=lz4 -O atime=off -O xattr=sa \
  -O mountpoint=/lake lake /dev/disk/by-id/ata-<18TB-hdd>
sudo zfs create -o recordsize=1M lake/lakehouse
sudo mkdir -p /lake/lakehouse/{seaweedfs,landing,spark-events}
sudo chown 185:185 /lake/lakehouse/spark-events
```

8. **Root folder for later tutorials:** `sudo mkdir -p /srv/pensieve`. Tutorial 3 puts training checkpoints here, on the fastest drive.

### Verify

- `docker compose version` shows 2.20 or newer, and `docker run --rm hello-world` works without `sudo`.
- `numactl -H` lists 4 nodes, each with 14 cores plus their SMT siblings and about 250 GB of memory.
- `df -h /mnt/scratch` shows about 930 GB, and `findmnt /mnt/scratch` shows `xfs`.
- `zpool list` shows `hot`, `lake` and `trunk` ONLINE; `zfs get recordsize hot/lakehouse lake/lakehouse` shows `16K` and `1M`.
- `gh auth status` shows you logged in.

### Explore (optional)

- **Measure each tier before Spark does.** Run the same two `fio` tests with `--filename` on `/mnt/scratch`, `/hot/lakehouse` and `/lake/lakehouse`, then delete the test files. The agent records the numbers in `docs/benchmarks.md` during Step 3; Tutorial 1's Phase 7 compares Spark stage times against them.

```bash
fio --name=seqwrite --filename=/mnt/scratch/fio.test --size=20G --bs=1M --rw=write --direct=1 --ioengine=libaio --iodepth=32
fio --name=randread --filename=/mnt/scratch/fio.test --size=20G --bs=4k --rw=randread --direct=1 --ioengine=libaio --iodepth=64
rm /mnt/scratch/fio.test
```

- **See your NUMA layout.** Compare `lscpu` and `numactl -H`. Each logical CPU N has its SMT sibling at N+56, so node 0 is `0-13,56-69`, node 1 `14-27,70-83`, node 2 `28-41,84-97` and node 3 `42-55,98-111`. Phase 7 pins workers to these.

## Step 2 (owner): Create and seed the repo

**Goal:** a private GitHub repo whose first commit already holds the working agreement and the plan, so the agent's very first session can follow the rules instead of being told them in a prompt.

The agent cannot open the tutorial pages (they are private claude.ai links), so each one is exported to Markdown and committed under `docs/tutorials/`. Re-export when a tutorial changes; the commit history then shows how the plan changed too.

### Build

1. **Create the repo and clone it** on the host:

```bash
gh repo create pensieve --private --clone && cd pensieve
mkdir -p docs/tutorials docs/journal .github
```

2. **Add the four starter files** at these paths: `README.md`, `AGENTS.md`, `docs/journal/TEMPLATE.md`, `.github/pull_request_template.md`. Then link `CLAUDE.md` to `AGENTS.md`, because Claude Code reads `CLAUDE.md`:

```bash
ln -s AGENTS.md CLAUDE.md
```

3. **Export every tutorial to Markdown** from its page and save them as:

| Tutorial | File |
| --- | --- |
| 0. Set up Pensieve (this one) | `docs/tutorials/0-setup.md` |
| 1. Build a Spark lakehouse on one box | `docs/tutorials/1-lakehouse.md` |
| 2. Streaming and MLOps on the lakehouse | `docs/tutorials/2-streaming-mlops.md` |
| 3. Deep learning from neurons to 7B models | `docs/tutorials/3-deep-learning.md` |

4. **Commit straight to `main`.** This is the only commit that skips a pull request, because the PR template does not exist until it lands.

```bash
git add -A && git commit -m "docs: README, working agreement, templates and tutorials"
git push -u origin main
```

5. **Protect `main`** if your GitHub plan allows it: add a branch rule that requires a pull request before merging. Even without the rule, `AGENTS.md` forbids the agent to merge.

### Verify

- `gh repo view --web` shows the README rendered on the repo page.
- `ls -l CLAUDE.md` shows the link to `AGENTS.md`.
- `ls docs/tutorials` lists four files.

## Step 3 (agent): Build the repo skeleton

**Goal:** the shared plumbing every later phase uses: pinned versions, the `./dc` Compose wrapper, the Makefile, the env template, cluster shapes and pre-commit. This is the first phase run the normal way, tagged `t0`.

### Kickoff prompt

Start the agent (Claude Code, or a smaller model such as Haiku) inside the repo on the host and send:

```text
Read AGENTS.md and README.md. Then do Step 3 of docs/tutorials/0-setup.md as phase t0.
Steps 1 and 2 are done by me; record their Verify commands and output in the journal
by running them read-only. Do not run any owner-only step.
```

If you ran the `fio` tests in Step 1, paste their output into the prompt and ask the agent to save it in `docs/benchmarks.md`.

### Build

1. **Create the folders and empty package.**

```bash
mkdir -p shapes make platform/{conf/{spark,seaweedfs,trino,source-db},docker/{spark,jupyter,airflow}} \
         pipelines/{jobs/{ingest,transform,maintain},dags} libs/pensieve/lake notebooks scripts .github/workflows
touch libs/pensieve/__init__.py libs/pensieve/lake/__init__.py
```

2. **Write `versions.env`** and fill every `<pin>` with the newest stable release on the day you start, from each project's release page or `gh release list -R <owner>/<repo>`. Record each chosen version under Findings in the journal.

```bash
# Every image and toolchain version in one place. Upgrade one line at a time; log it in docs/upgrades.md.
SPARK_BASE=spark:4.1.2-scala2.13-java17-python3-ubuntu
ICEBERG_VERSION=1.11.0
POSTGRES_TAG=17
SEAWEEDFS_TAG=<pin>
POLARIS_TAG=<pin a 1.x release>
TRINO_TAG=<pin>
AIRFLOW_VERSION=3.3.2
JUPYTER_TAG=<pin a dated tag>
AWS_CLI_TAG=<pin>
# tutorial 2
KAFKA_TAG=4.3.0
KAFKA_UI_TAG=<pin>
KAFKA_EXPORTER_TAG=<pin>
MLFLOW_TAG=<pin a v3.x release>
VLLM_TAG=<pin>
# tutorials 2 and 3
UV_VERSION=<pin>
```

3. **Write `./dc`**, the one Compose entry point. It loads `versions.env`, `.env` and the shape file, then passes Compose the stack files you ask for (all that exist by default). Make it executable with `chmod +x dc`.

```bash
#!/usr/bin/env bash
# usage: ./dc <any docker compose command>   e.g. ./dc ps, ./dc logs -f trino, STACKS="stream" ./dc up -d
set -euo pipefail
cd "$(dirname "$0")"
set -a; . ./versions.env; [ -f .env ] && . ./.env; . "shapes/${SHAPE:-small}.env"; set +a
stacks="${STACKS:-}"
if [ -z "$stacks" ]; then                       # default: every stack file that exists
  for f in platform/compose.*.yaml; do [ -e "$f" ] && stacks="$stacks $(basename "$f" .yaml | cut -d. -f2)"; done
fi
args=()
for s in $stacks; do [ -f "platform/compose.$s.yaml" ] && args+=(-f "platform/compose.$s.yaml"); done
exec docker compose -p homelab --project-directory platform "${args[@]}" "$@"
```

4. **Write `.env.example`.** Versions are deliberately not here: `.env` holds secrets and machine paths, `versions.env` holds what you commit. The owner copies it to `.env` and sets real secrets.

```bash
HOST_NAME=pensive
DATA_ROOT=/lake/lakehouse
PG_ROOT=/hot/lakehouse
SCRATCH_DIR=/mnt/scratch/spark

S3_ACCESS_KEY=lakeadmin
S3_SECRET_KEY=change-me
WAREHOUSE_BUCKET=warehouse

POLARIS_CLIENT_ID=root
POLARIS_CLIENT_SECRET=change-me
CATALOG_DB_PASSWORD=change-me
SOURCE_DB_PASSWORD=change-me
AIRFLOW_DB_PASSWORD=change-me
```

5. **Write `shapes/small.env`.** Cluster size lives here, not in `.env`; Tutorial 1 adds `large.env` and `numa.env`.

```bash
SPARK_WORKERS=2
SPARK_WORKER_CORES=8
SPARK_WORKER_MEMORY=48g
SPARK_WORKER_MEM_LIMIT=56g
EXECUTOR_CORES=4
EXECUTOR_MEMORY=20g
APP_MAX_CORES=16
```

6. **Write the `Makefile`.** It loads `versions.env`, `.env` and a shape file and exports every variable, so `make up SHAPE=large` resizes the cluster without editing anything. `make up` starts only the core stacks; optional ones start with `make up-with` or the shortcuts. Stack-specific targets live in `make/*.mk`.

```makefile
SHAPE ?= small
include versions.env
-include .env
include shapes/$(SHAPE).env
export

DC     := ./dc
CORE   := storage catalog spark query sources
STACKS ?= $(CORE)
SCALE   = $(if $(filter spark,$(STACKS)),--scale spark-worker=$(SPARK_WORKERS))

net:
	docker network inspect lake >/dev/null 2>&1 || docker network create lake

up: net conf                  # starts only the selected stacks (default: the core)
	STACKS="$(STACKS)" $(DC) up -d --build $(SCALE)

up-with:                      # usage: make up-with S="notebook airflow"
	$(MAKE) up STACKS="$(CORE) $(S)"

up-stream: ; $(MAKE) up-with S=stream        # tutorial 2
up-ml:     ; $(MAKE) up-with S=ml            # tutorial 2
up-dl:     ; $(MAKE) up-with S="ml dl"       # tutorial 3

down:                         # stops the selected stacks
	STACKS="$(STACKS)" $(DC) down
down-all:
	$(DC) down
ps:
	$(DC) ps
logs:
	$(DC) logs -f $(S)
versions-check:               # fails while any version is still a placeholder
	@! grep -n '<' versions.env

-include make/*.mk
```

7. **Write `make/core.mk`.** `conf` is a placeholder until Tutorial 1 renders config templates. `lock` regenerates every image's lockfile; each image folder holds a hand-written `requirements.in` (top-level packages only) and a `python-version` file. The GPU images in Tutorials 2 and 3 use a full `uv.lock` instead.

```makefile
conf:
	@echo "nothing to render yet"

lock:                         # regenerate every image's requirements.lock
	for d in platform/docker/*/; do \
	  [ -f $$d/requirements.in ] && uv pip compile $$d/requirements.in -o $$d/requirements.lock \
	    --python-version $$(cat $$d/python-version); \
	done
```

8. **Write `.gitignore`.**

```text
.env
platform/conf/**/*.generated
platform/conf/spark/spark-defaults.conf
platform/conf/seaweedfs/s3.json
__pycache__/
.venv/
mlruns/
.ipynb_checkpoints/
recordings/
checkpoints/
*.pt
*.safetensors
*.gguf
```

9. **Write `.pre-commit-config.yaml`**, then run `pre-commit install` and `pre-commit autoupdate` once to pin real versions.

```yaml
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.0.0           # replaced by autoupdate
    hooks: [{ id: gitleaks }]          # blocks commits containing keys or passwords
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks: [{ id: nbstripout }]        # commits notebooks without outputs
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks: [{ id: ruff }]
```

### Verify

- `make versions-check` passes: no `<pin>` left in `versions.env`.
- `make net` creates the `lake` network, and running it again is a no-op.
- `cp .env.example .env && git check-ignore .env` prints `.env`.
- `bash -n dc` reports no syntax errors, and `./dc version` prints the Compose version.
- `pre-commit run --all-files` passes.
- `docs/journal/t0.md` records the Step 1 and Step 2 Verify outputs and every version chosen.

### Commit

`chore: repo skeleton, versions.env, dc wrapper, env template, cluster shapes, Makefile, pre-commit`

The agent opens the pull request using the template. You review, merge, and tag: `git tag t0 && git push --tags`.

## Step 4: Run phases with the harness

**Goal:** a loop you repeat for every phase from `t1-p01` to `t3-p35`, with you deciding what merges. Once `t0` is tagged, the rules live in `AGENTS.md`, so the prompt for each phase is one line:

```text
Read AGENTS.md, then do phase t1-p01 from docs/tutorials/1-lakehouse.md.
```

### The loop

1. **Start the phase.** Send the one-line prompt in a fresh agent session, so each phase starts from the repo and its journal rather than from a long chat.
2. **Do owner steps when asked.** If the phase needs something on the host (GPU drivers and the NVIDIA toolkit in Tutorial 2, a reboot, deleting data), the agent stops and lists the exact commands under "Needs owner action" in the journal. Run them, then tell the agent to continue.
3. **Review the pull request.** Read the journal entry first: Verify results, Deviations, Findings. Then the diff. Ask for changes in PR comments, or re-run any Verify command yourself.
4. **Merge and tag.** `git tag t1-p01 && git push --tags`. Only you merge and tag.
5. **Next phase.** Back to 1 with the next tag.

### When a phase is blocked

A PR titled `BLOCKED: <tag>` means a Verify step failed after one reasonable fix. Its journal entry says what failed and the error. Fix it yourself, or hand that phase to a stronger model with the journal entry as context. Either way the fix lands in the same branch, and the deviation stays recorded.

A smaller model such as Haiku follows clearly written steps well and struggles with environment problems it hasn't seen: driver mismatches, permission errors, a tag that no longer exists. The stop-and-block rule keeps those from turning into silent workarounds.

### Keeping the plan and the repo in step

- **A tutorial changes:** re-export it to `docs/tutorials/` and commit it in its own small PR (`docs: refresh tutorial 2`) before the next phase starts.
- **Catching up after a break:** `git log --oneline --decorate` shows one merge and one tag per phase; `docs/journal/<tag>.md` gives the numbers and deviations; the PR holds the review discussion.
- **Going back:** `git checkout t1-p05` gives the repo exactly as that phase left it.

Tutorial 0 is done when `t0` is tagged and you have sent the first one-line prompt. Next: Tutorial 1, Phase 1.

