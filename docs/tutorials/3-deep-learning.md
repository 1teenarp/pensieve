# Deep Learning from Neurons to 7B Models

Oct 29, 2026 · @Praneet

## Start here

This third tutorial turns your two RTX Pro 5000 GPUs into a deep learning lab. You'll build every major architecture yourself, from a single neuron to a mixture-of-experts transformer. You'll write GPU kernels in CUDA and Triton and plug them into your own model, then work with 7B-parameter models: designing one, training it at small scale, and adapting real ones through continued pretraining, fine-tuning and reinforcement learning. Three projects make it concrete: a typed-decision model in the style of Jev and Laya, a computer-use model that operates DaVinci Resolve on your Mac, and a JEPA (a model that predicts meaning rather than pixels or tokens).

It builds on tutorials 1 and 2: [the lakehouse](https://claude.ai/artifact/X9BMca4uV6qkeMSY6L3JFX) stores datasets and checkpoints, and [MLflow and the trainer image](https://claude.ai/artifact/J1idPQJcKG9fgi51hW7ryY) track every run. Phases continue from 22. Each has a **Brief** (what you're building, why it matters, and which tool does what), then **Build**, **Verify**, **Commit**, and optional **Explore**.

The tutorial covers three kinds of model that answer different questions. An **LLM** answers "what text comes next?" A **typed-decision model** such as TypeSafe's Jev or the open Laya answers "which of these options is right, and how sure am I?" A **JEPA** answers "what does the part I can't see mean?" Phase 24 builds a Laya-style decision model, Phase 25 uses it inside the computer-use agent as a decider and safety gate, and Phase 34 ends with JEPA.

It's in two parts. **Part A (Phases 22–25)** gets you capable, specialized models fast by building on open models. **Part B (Phases 26–35)** rebuilds the same machinery from scratch, locally, including a 30-day pretraining run you can pause and resume.

&#91;embedded content: learning path · Part A, then Part B, Phases 22–35\]

Phase 28 is Part B's hinge: the GPT you write there grows into the MoE, the kernels plug into it, and the same file becomes the 7B architecture.

### What your 72 GB GPUs can and can't do

Be clear-eyed about this before you start, because it decides which projects are realistic.

Each RTX Pro 5000 Blackwell has 72 GB of memory, about 1.34 TB/s of memory bandwidth, and roughly 147 TFLOPS of dense BF16 tensor throughput, according to [third-party spec sheets](https://hmc-tech.com/gpus/nvidia-rtx-pro-5000-blackwell-vs-nvidia-rtx-pro-6000-blackwell) (NVIDIA's [datasheet](https://www.pny.com/File%20Library/Company/Support/Product%20Brochures/NVIDIA%20RTX/English/nvidia-rtx-pro-5000-blackwell.pdf) leads with FP4). Real training reaches 35–50% of peak, so plan on about **120 TFLOPS for both cards together**. The cards have no NVLink, and your ROMED8-2T runs them at PCIe 4.0, so communication between GPUs is slower than on data-center machines. Run `nvidia-smi topo -m` to see your link. You also have a third card as a spare. Installing it adds 50% more compute and memory; the tables below assume two, and Phase 33 shows where three changes the plan.

Training cost is roughly 6 × parameters × tokens floating-point operations. Compute-optimal training ("Chinchilla") uses about 20 tokens per parameter:

| Model | Tokens | Compute | Time on your box at 120 TFLOPS | Verdict |
| --- | --- | --- | --- | --- |
| 124M (GPT-2 small) | 2.5B | 1.9 × 10^18 | About 4.5 hours | Easy; do it in Phase 29 |
| 350M | 7B | 1.5 × 10^19 | About 1.5 days | Good weekend project |
| nanochat speedrun (about 560M) | About 11B | 4 × 10^19 | About 4 days | Worth it once you know the pipeline |
| 1B | 20B | 1.2 × 10^20 | About 12 days | Possible as a long run |
| 7B | 140B | 5.9 × 10^21 | About 1.6 years | Not feasible; adapt a pretrained 7B instead |

The nanochat figure comes from [Karpathy's repo](https://pypi.org/project/nanochat), which quotes about 4 × 10^19 FLOPs for its $100 speedrun on 8 H100s.

What **is** feasible at 7B, thanks to your 1 TB of RAM:

- **Run and serve** any 7B model (14 GB in BF16).
- **Build your own 7B architecture**, verify it, and train it for a few thousand steps to prove the whole stack works.
- **LoRA and QLoRA** fine-tuning on one GPU.
- **Full fine-tuning and continued pretraining** with FSDP: on two cards with optimizer state offloaded to system RAM, or entirely on GPU with your third card installed. Slow, but it works for millions to low billions of tokens.
- **DPO and GRPO** reinforcement learning with LoRA adapters.

### Model sizes: what they can do, and where to train them

Size sets the ceiling, but most of a small model's capability comes from how much data it saw. Labs now train small models far past the compute-optimal point (roughly 11T tokens for SmolLM2-1.7B, about 15T for Llama 3 8B) and then post-train heavily. You can match the size locally; the data volume is what costs money. That's why Part A builds on open models first.

| Tier | What the best published models at this size can do | What you'd get from scratch | Local (2 cards, 30 days or less) | Cloud: compute-optimal / lab-style |
| --- | --- | --- | --- | --- |
| 10–125M (TinyStories, GPT-2 small) | Fluent text in a narrow domain; GPT-2 small scores about 30% on HellaSwag (25% is chance) | The same quality (Phase 29) | Yes: compute-optimal in about 4.5 hours | About $4 / about $1,600 for 1T tokens |
| 350–600M (nanochat speedrun, SmolLM2-360M) | Chats "like a kindergartener"; MMLU about 31%, GSM8K math under 10% ([nanochat](https://github.com/1Memre/nanochat)). Specialized, excellent: Laya (421M) routes text at 99% accuracy | Matches nanochat; with your data, a strong narrow specialist | Yes: compute-optimal in about 1.5 days | About $30–$100 / about $4,500 |
| 1–2B (Llama 3.2 1B, SmolLM2-1.7B) | Basic chat, summaries, simple tool calls, grade-school math when post-trained well | A solid base model (Phase 29's 30-day run) | Yes for 1B: about 50B tokens in 30 days, about 80B with three cards | About $250 / about $12,500 |
| 3–4B (Qwen3-4B, Llama 3.2 3B) | Reliable instruction following, decent code, competition-style math with a reasoning mode | Clearly undertrained without a lot of data | No: compute-optimal takes about 3.5 months | About $2,200 / about $37,000 |
| 7–9B (Llama 3.1 8B, Qwen3-8B, OLMo 7B) | A capable general assistant: about two-thirds on MMLU, useful coding, RAG and basic agents | Weaker than open 7Bs; only worth it to test your own design | No: about 20× too few tokens in 30 days | About $15,000 / about $200,000 for 2T tokens |
| 7B-total MoE, about 1B active | 7B-class knowledge at 1B-class compute per token | Your Phase 30 design at full scale | No | About $15,000–$20,000 for 1T tokens |
| About 30B (Gemma 4 31B, Qwen 27–32B) | Strong reasoning and coding; matches Jev at typed decisions with no fine-tuning ([paper](https://arxiv.org/pdf/2610.17205)) | Out of independent budget range | No | About $225,000 / millions |

Estimates assume about 6 × parameters × tokens FLOPs, about 120 TFLOPS effective on your two cards, and cloud H100s at about $3 per hour reaching about 400 TFLOPS each. Add 30–100% for failed runs and experiments. Benchmark figures are approximate. Fine-tuning has a much higher local ceiling: with 72 GB cards you can adapt open 7B–30B models at home.

### Later: a cloud capstone

This tutorial stays local. Once Part B shows you a design that wins at small scale, cloud GPUs are the way to give it data volume. Current prices are about $2.40–$4 per H100-hour and $6.70–$8 per B200-hour at specialist clouds, with spot capacity cheaper ([neocloud survey, October 2026](https://blog.prompt20.com/posts/neoclouds-complete-guide/)). A safe sequence:

1. **Locally:** fit scaling curves for each candidate design (Phase 29's ladder) and pick the winner.
2. **Cloud smoke test (about $100):** run the exact large config on an 8-GPU node for an hour, measure real throughput, and recompute the budget.
3. **Cloud pilot (about $500–$2,000):** train a 1B model for about 20B tokens and check it lands where your curve predicted.
4. **Full run:** a 7B MoE or a long-trained 1–3B model ($5,000–$40,000), on spot capacity, resuming from checkpoints exactly as Phase 29's long run does.

### Pace

Part A gives you working specialists within about six weeks of evenings. In Part B, start the 30-day run in Phase 29 and keep learning while it trains.

| Phase | Your time | Machine time |
| --- | --- | --- |
| 22 Workstation | 1–2 evenings | Minutes |
| 23 Post-training an open 8B | 1–2 weeks | SFT a few hours; GRPO overnight |
| 24 Typed-decision models | About 1 week | Encoder training a few hours; LLM LoRA about an hour |
| 25 Computer use | 2–3 weeks (recording demonstrations takes the time) | A few hours per training run |
| 26–28 Foundations to your own GPT | About 1 week each | Under an hour per experiment |
| 29 Pretraining | 1–2 weeks, plus the 30-day run in the background | 30 days on both GPUs (pausable) |
| 30 MoE | About 1 week | Hours |
| 31–32 CUDA and Triton | About 2 weeks each | Minutes per kernel |
| 33 The 7B model | 1–2 weeks | Days for continued pretraining |
| 34 JEPA | About 1 week | Several hours |
| 35 Playbook | An evening | None |

The 30-day run occupies both GPUs while Phases 30–32 also need one. Either install your spare card now (two for the long run, one for daily work), or pause the run whenever you need a GPU; Phase 29 makes pausing safe.

### Which tool does what

| Capability | Tool | What it does for you |
| --- | --- | --- |
| Tensors, autograd, models | PyTorch 2.7+ (CUDA 12.8 builds) | The core framework; runs on Blackwell's compute capability 12.0 |
| Compiling models | `torch.compile` | Fuses operations and generates GPU code automatically |
| Writing kernels in Python | Triton | Block-level GPU programming; what `torch.compile` itself generates |
| Writing kernels in C++ | CUDA toolkit (`nvcc`) | Full control over threads, memory and tensor cores |
| Profiling | Nsight Systems, Nsight Compute, PyTorch profiler | Timeline of the whole program; deep dive into one kernel |
| Multi-GPU training | `torchrun`, DDP, FSDP2 | Data parallelism; sharding weights, gradients and optimizer state |
| Memory savings | Activation checkpointing, CPU offload, 8-bit optimizers | Trading compute or PCIe bandwidth for memory |
| Tokenizers and data | Hugging Face `tokenizers`, `datasets`; your lakehouse | Training BPE, streaming and sharding text |
| Pretrained models | Hugging Face `transformers` | Loading and adapting open models |
| Adapters | PEFT (LoRA, QLoRA), bitsandbytes | Training small add-on weights instead of the whole model |
| Post-training | TRL (SFT, reward models, DPO, PPO, GRPO) | The standard recipes for chat, preference and reward tuning |
| Fast generation | vLLM | Serving, and generating rollouts during reinforcement learning |
| Tracking | MLflow from tutorial 2 | Runs, metrics, checkpoints, comparisons |
| Reference code | nanoGPT, nanochat | Small, readable, complete training pipelines to compare against |

### What this adds to the repo

```text
pensieve/
  platform/
    compose.dl.yaml         # GPU JupyterLab, the dl shell, the 30-day pretraining service
    docker/dl/              # dev image: CUDA toolkit, PyTorch, Triton, Nsight (installs dl/uv.lock)
  dl/                       # pyproject.toml + uv.lock
    bench/                  # Phase 22: hardware benchmarks
    post/                   # Phase 23: SFT, DPO, GRPO
    decide/                 # Phase 24: typed-decision models
    cua/                    # Phase 25: computer-use model; cua/mac/ runs on your Mac (own small pyproject)
    foundations/            # Phase 26: micrograd, MLPs, training loops
    classic/                # Phase 27: CNN, LSTM, seq2seq with attention
    gpt/                    # Phases 28-30: tokenizer, GPT, pretraining, MoE
    kernels/                # Phases 31-32: CUDA and Triton kernels, custom ops
    big/                    # Phase 33: the 7B architecture, FSDP configs
    jepa/                   # Phase 34
  make/dl.mk
  docs/dl-notes.md          # your learning log
```

Tutorial 0's repo conventions apply here too: Compose paths are relative to `platform/`, versions live in `versions.env`, Compose commands go through `./dc`, and this tutorial's targets go in `make/dl.mk`. Start the stack with `make up-dl` and tag each phase, for example `git tag t3-p23`. Datasets, checkpoints, model weights and screen recordings stay outside the repo.

**Prerequisites:** Tutorial 0 (Set up Pensieve: Host, Repo and Working Agreement), and Tutorial 2 tagged through `t2-p21` for MLflow and serving. Each phase follows Tutorial 0's loop: one branch, PR, journal entry and tag. Hand a phase to the agent with `Read AGENTS.md, then do phase t3-p22 from docs/tutorials/3-deep-learning.md.` Long training runs, deleting checkpoints and any driver or kernel-module change stay owner-only.

## Phase 22: A deep learning workstation

### Brief

You need one development image with everything from Python down to the CUDA compiler and profilers, a GPU-backed JupyterLab for exploring, a shell for long runs, and fast storage for datasets and checkpoints. Then you measure what your hardware actually delivers, so every later estimate rests on your numbers, not a spec sheet.

| Piece | Tool | Role |
| --- | --- | --- |
| Base image | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` | The `devel` variant includes `nvcc` and headers, needed for CUDA kernels in Phase 31 |
| Python environment | `uv` with a lockfile | The same exact versions on every rebuild |
| Notebook | JupyterLab | Interactive exploration on the GPU |
| Long runs | A `dl` shell service and `torchrun` | Training jobs that run for hours, outside a notebook |
| Profilers | Nsight Systems (`nsys`), Nsight Compute (`ncu`) | Whole-program timelines; per-kernel detail |
| Storage | Root NVMe (`/srv/pensieve`) for checkpoints, ZFS `lake` for datasets | Checkpoints are big and written often; datasets are big and read often |

### Build

**Storage.** A full 7B training checkpoint with optimizer state is over 100 GB. Write checkpoints to the root NVMe under `/srv/pensieve`, the fastest drive you have (see Tutorial 0's storage tiers), keep only the latest few, and copy keepers to `lake`. When a dedicated scratch NVMe arrives, move `DL_CKPT` there:

```bash
sudo mkdir -p /lake/lakehouse/dl/{datasets,hf} /srv/pensieve/dl/checkpoints
sudo chown -R $USER /lake/lakehouse/dl /srv/pensieve/dl
```

Add `DL_DATA=/lake/lakehouse/dl` and `DL_CKPT=/srv/pensieve/dl/checkpoints` to `.env`.

**Profiler permissions.** By default, NVIDIA drivers allow GPU performance counters only for root, which blocks `ncu`. Allow it for all users on this lab box:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u && sudo reboot
```

**Dependencies.** `dl/pyproject.toml`:

```toml
[project]
name = "lake-dl"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "torch>=2.7", "torchvision", "numpy", "matplotlib", "einops", "tqdm",
  "jupyterlab", "ipywidgets",
  "transformers", "datasets", "tokenizers", "tiktoken", "accelerate", "safetensors",
  "peft", "trl", "bitsandbytes", "mlflow>=3", "pyiceberg[pyarrow]", "trino",
  "ninja",                      # builds C++/CUDA extensions quickly
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
```

Triton ships inside the PyTorch wheel. vLLM pins its own PyTorch version, so keep it in its own image (tutorial 2's `vllm` service) rather than here. Run `uv lock` in `dl/` and commit `uv.lock`.

`docker/dl/Dockerfile`:

```dockerfile
ARG UV_VERSION
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev git curl build-essential \
      cuda-nsight-systems-12-8 cuda-nsight-compute-12-8 && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /work
COPY dl/pyproject.toml dl/uv.lock ./
RUN uv sync --frozen --no-install-project
ENV PATH=/work/.venv/bin:/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="12.0"
```

`TORCH_CUDA_ARCH_LIST=12.0` tells PyTorch to compile your custom CUDA extensions for Blackwell workstation GPUs only, which is faster than building for every architecture. If the Nsight package names differ in your CUDA repository, `apt-cache search nsight` lists the right ones.

`compose.dl.yaml`:

```yaml
x-dl: &dl
  build: { context: .., dockerfile: platform/docker/dl/Dockerfile, args: [UV_VERSION] }
  image: lakehouse/dl:${GIT_SHA:-dev}
  working_dir: /work/repo
  environment:
    HF_HOME: /data/hf
    MLFLOW_TRACKING_URI: http://mlflow:5000
    PYTHONPATH: /work/repo/libs        # so `import pensieve` works
    POLARIS_CREDENTIAL: ${POLARIS_CLIENT_ID}:${POLARIS_CLIENT_SECRET}
    S3_ACCESS_KEY: ${S3_ACCESS_KEY}
    S3_SECRET_KEY: ${S3_SECRET_KEY}
  volumes:
    - ..:/work/repo
    - ${DL_DATA}:/data
    - ${DL_CKPT}:/ckpt
  shm_size: 32g
  ipc: host                       # NCCL and data loaders use shared memory
  ulimits: { memlock: -1, stack: 67108864 }
  deploy:
    resources:
      reservations:
        devices: [{ driver: nvidia, count: all, capabilities: [gpu] }]
  networks: [lake]

services:
  jupyter-gpu:
    <<: *dl
    command: jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --ServerApp.root_dir=/work/repo
    ports: ["8898:8888"]

  dl:
    <<: *dl
    profiles: [tools]

networks:
  lake:
    external: true
```

Both GPUs are visible to these containers. You choose per run with `CUDA_VISIBLE_DEVICES`, so a training job can take GPU 1 while vLLM keeps GPU 0. Targets for make/dl.mk:

```makefile
# usage: make dl CMD="python dl/bench/matmul.py" GPUS=1
dl:
	$(DC) --profile tools run --rm -e CUDA_VISIBLE_DEVICES=$(or $(GPUS),0,1) dl $(CMD)

# usage: make dl-dist CMD="dl/gpt/train.py --config dl/gpt/configs/124m.yaml"
dl-dist:
	$(DC) --profile tools run --rm dl torchrun --standalone --nproc_per_node=2 $(CMD)
```

### Verify: measure your hardware

`dl/bench/matmul.py` measures achieved matrix-multiply throughput:

```python
import torch, time

print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())
for dtype in (torch.float32, torch.bfloat16):
    for n in (4096, 8192, 16384):
        a = torch.randn(n, n, device="cuda", dtype=dtype); b = torch.randn(n, n, device="cuda", dtype=dtype)
        for _ in range(3): a @ b                      # warm-up
        torch.cuda.synchronize(); t = time.perf_counter()
        iters = 20
        for _ in range(iters): a @ b
        torch.cuda.synchronize(); dt = (time.perf_counter() - t) / iters
        print(f"{str(dtype):15} n={n:5}  {2 * n**3 / dt / 1e12:7.1f} TFLOPS")
```

`dl/bench/allreduce.py` measures how fast the two GPUs can exchange gradients:

```python
import os, time, torch, torch.distributed as dist

dist.init_process_group("nccl"); rank = dist.get_rank(); torch.cuda.set_device(rank)
for mb in (16, 256, 1024):
    x = torch.ones(mb * 2**20 // 2, dtype=torch.bfloat16, device="cuda")
    for _ in range(3): dist.all_reduce(x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10): dist.all_reduce(x)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 10
    if rank == 0:
        print(f"{mb:5} MB  {mb / 1024 / dt:6.1f} GB/s algorithm bandwidth")
dist.destroy_process_group()
```

```bash
make dl CMD="python dl/bench/matmul.py" GPUS=1
make dl-dist CMD="dl/bench/allreduce.py"
make dl CMD="nvidia-smi topo -m"
make dl CMD="nvcc --version"
```

Write all four results in `docs/dl-notes.md`. Expect BF16 well above FP32, and an all-reduce bandwidth in the low tens of GB/s, since it travels over PCIe 4.0. Every training-time estimate later uses these numbers. Then open `http://pensive:8898` and run `torch.cuda.is_available()` in a notebook.

### Commit

`feat(dl): CUDA devel image with PyTorch, Triton and Nsight; GPU JupyterLab; benchmarks`

### Explore (optional)

- **Find the crossover.** Run the matmul benchmark for n from 256 to 16384. Small matrices can't keep the GPU busy; note where throughput levels off. You'll see the same effect when choosing batch sizes.
- **TF32.** Set `torch.backends.cuda.matmul.allow_tf32 = True` and rerun FP32. Learn what precision you give up for the speed.
- **Power and clocks.** Watch `nvidia-smi dmon -s pucvmet` during the benchmark. A 300 W card under sustained load may clock down; your cooling work on this box matters here too.
- **Peer-to-peer.** Check whether NCCL uses direct GPU-to-GPU transfers with `NCCL_DEBUG=INFO` in the all-reduce run. Then force it off with `NCCL_P2P_DISABLE=1` and compare.

## Part A: Capability fast — build on open models

These three phases give you working, specialized models within the first weeks, by adapting open models that labs have already trained on trillions of tokens. You'll use LoRA, DPO and GRPO before you've seen their insides; Part B opens them up.

## Phase 23: Post-training — SFT, RLHF, DPO and GRPO

### Brief

A base model continues text; a useful model follows instructions, holds a conversation, and prefers good answers to bad ones. Post-training gets you there. You'll run the full sequence twice: once on your own 124M base model (to see every step in plain code) and once on an open 7–8B model with LoRA (to get something genuinely capable). The specialized role is the one you already have data and a grader for: a **lakehouse SQL analyst** that answers questions about your tables with SQL that actually runs.

| Stage | Data it needs | What it teaches | Tool |
| --- | --- | --- | --- |
| Mid-training (optional) | Conversations and task text, plain loss | The chat format and task styles | Same trainer as pretraining |
| Supervised fine-tuning (SFT) | Prompt and ideal response pairs | To imitate good answers | TRL `SFTTrainer`, or your own loop |
| Reward model | Pairs of responses with a preferred one | To score answers like a human would | TRL `RewardTrainer` |
| RLHF with PPO | Prompts, plus the reward model | To maximize reward without drifting far from the SFT model | TRL PPO trainer (heavy: four models at once) |
| DPO | The same preference pairs, no reward model | Preferences directly, with a simple loss | TRL `DPOTrainer` |
| GRPO with verifiable rewards | Prompts plus a program that checks answers | To find answers that pass the check, through its own trial and error | TRL `GRPOTrainer`, with vLLM for generation |

The key ideas behind the last three:

- **Reward model.** A copy of the language model with a single-number output head, trained so the preferred answer scores higher: the loss is minus log sigmoid of (score of chosen minus score of rejected).
- **PPO-based RLHF.** The model generates answers, the reward model scores them, and a policy-gradient update raises the probability of high-scoring answers. A penalty on divergence from the original SFT model (the KL term) stops it from exploiting the reward model with nonsense that happens to score well. It needs the policy, a reference copy, the reward model and a value model in memory together, which is why it's rare on small setups.
- **DPO.** It turns out the same objective can be optimized directly on preference pairs, with no reward model and no sampling: raise the chosen answer's log probability and lower the rejected one's, each measured relative to the reference model.
- **GRPO.** Sample a group of answers to the same prompt, score each with a reward function, and use each answer's reward relative to its group's average as its advantage. There's no value model. When the reward comes from a program (tests pass, SQL returns the right rows, math is correct), there's no reward model either, which is how recent reasoning models were trained.

### Build: a chat format

Open instruct models ship a chat template, and `tokenizer.apply_chat_template` renders a conversation with it; TRL does this for you. For your own models in Part B, the Phase 29 tokenizer reserves `<|system|>`, `<|user|>`, `<|assistant|>` and `<|end|>`, so a conversation becomes one sequence like this:

```text
<|system|>You answer with Trino SQL for the lake catalog.<|end|><|user|>How many trips in Queens in March?<|end|><|assistant|>SELECT sum(trips) FROM gold.daily_zone_weather WHERE ...<|end|>
```

During SFT, set the targets of every token outside assistant turns to -1 (your model's `ignore_index`), so the loss only trains on what the assistant says. Forgetting this mask is the most common SFT bug: the model then learns to write user questions too.

### Build: SFT on your own base model (after Part B)

`dl/post/sft_small.py` comes back after Phase 29. It reuses Phase 29's model and loop with three changes: data comes from a conversation dataset rendered with the template above (start with a small general chat set such as `HuggingFaceTB/smol-smoltalk`, then mix in Tutorial 2's SQL pairs at 20–30%); targets are masked as described; and the learning rate is about one-tenth of pretraining's, for one or two epochs. Load your 124M checkpoint, train, then chat with it through `generate`. It will be clumsy but recognizably an assistant, and you'll have seen every line that made it so.

### Build: SFT on a 7–8B model with QLoRA

Use an open instruct model in the 7–8B range as the starting point, for example from the Qwen3 or OLMo families. Extend Tutorial 2's dataset builder (`ml/sql_dataset.py`) to produce a few thousand verified examples across more tables and question styles, keeping the held-out-template split. Then train exactly as in Tutorial 2, Phase 19, with `qlora: true` and the larger base model. On one 72 GB GPU, a 4-bit 8B model with a rank-16 adapter and gradient checkpointing fits with room for batches of several 2K-token sequences.

Measure execution accuracy with Tutorial 2's `ml/eval_sql.py` for the base model and your SFT adapter. That's the baseline every later step must beat.

### Build: preference data, a reward model, and DPO

Make preference pairs from your own SFT model, which is the standard on-policy recipe:

```python
# dl/post/make_prefs.py: for each training question, sample 4 answers and grade them by running them
for ex in train_examples:
    answers = sample(model, ex["messages"][:2], n=4, temperature=0.9)
    graded = [(a, is_correct(a, ex["gold_sql"])) for a in answers]
    good = [a for a, ok in graded if ok]; bad = [a for a, ok in graded if not ok]
    if good and bad:                                   # only questions where the model is unsure teach anything
        pairs.append({"prompt": ex["messages"][:2],
                      "chosen": [{"role": "assistant", "content": good[0]}],
                      "rejected": [{"role": "assistant", "content": bad[0]}]})
```

`is_correct` is Tutorial 2's comparison: run both queries in Trino and compare sorted rows. Then:

- **Reward model (to understand RLHF).** Train with `RewardTrainer` on these pairs, starting from your SFT model with LoRA. Check it: on held-out pairs, does it score the chosen answer higher at least 70–80% of the time? A reward model is only as good as this number.
- **DPO (to actually improve the model).** Train with `DPOTrainer` on the same pairs, starting from the SFT adapter, with `beta=0.1` and a learning rate around 5e-6. Re-measure execution accuracy.

### Build: GRPO with a verifiable reward

Here the grader is the reward, so the model learns directly from whether its SQL works. `dl/post/grpo_sql.py`:

````python
import json, re, trino
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# copy the verified dataset from Tutorial 2 (${MODEL_CACHE}/datasets) into ${DL_DATA}/sql first
rows = [json.loads(l) for l in open("/data/sql/lake_sql_train.jsonl")]
ds = Dataset.from_list([{"prompt": r["messages"][:2], "gold_sql": r["messages"][2]["content"]} for r in rows])
cur = trino.dbapi.connect(host="trino", port=8080, user="grpo", catalog="lake").cursor()

def run(sql):
    cur.execute(sql); return sorted(map(tuple, cur.fetchall()), key=repr)

def extract(text):
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip().rstrip(";")

def sql_reward(completions, gold_sql, **kwargs):     # extra dataset columns arrive as keyword arguments
    out = []
    for c, gold in zip(completions, gold_sql):
        sql = extract(c[0]["content"])
        try:
            out.append(1.0 if run(sql) == run(gold) else 0.0)
        except Exception:
            out.append(-0.5)                           # invalid SQL is worse than wrong SQL
    return out

def brevity_reward(completions, **kwargs):           # discourage rambling around the query
    return [-0.001 * max(0, len(c[0]["content"]) - 400) for c in completions]

args = GRPOConfig(output_dir="/ckpt/grpo-lakesql", report_to="mlflow", bf16=True,
                  num_generations=8, max_completion_length=256, per_device_train_batch_size=8,
                  gradient_accumulation_steps=4, learning_rate=1e-5, num_train_epochs=1, logging_steps=5,
                  use_vllm=True, vllm_mode="server")    # generation runs in vLLM on the other GPU
trainer = GRPOTrainer(model="/ckpt/sft-lakesql-merged", reward_funcs=[sql_reward, brevity_reward],
                      args=args, train_dataset=ds,
                      peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
trainer.train()
````

Merge the SFT adapter into the base model first (`merge_and_unload`, saved to `/ckpt/sft-lakesql-merged`), so GRPO starts from the SFT policy. Generation is the slow part of online RL, so split the GPUs: [TRL's server mode](https://docs.vllm.ai/training/trl.html) runs vLLM as its own process. Start it on GPU 0 with `trl vllm-serve --model /ckpt/sft-lakesql-merged` and the trainer on GPU 1. Check your TRL version's documentation for how it syncs LoRA weights to the server; it has changed across releases.

Watch these in MLflow while it trains: `reward` rising, `reward_std` staying above zero (if every answer in a group gets the same reward, there's nothing to learn from), and completion length.

### Verify

One table in `docs/dl-notes.md`, all measured on the held-out templates:

| Model | Execution accuracy | Invalid SQL rate |
| --- | --- | --- |
| Base instruct model |  |  |
| + SFT (QLoRA) |  |  |
| + DPO |  |  |
| + GRPO |  |  |

Each step should improve on the one before, or you should be able to say why it didn't. Also re-run HellaSwag or a small general benchmark on the final model; a large drop means you over-specialized.

### Commit

`feat(post): chat format, SFT on own base and 8B QLoRA, preference pairs, reward model, DPO, GRPO with SQL rewards`

### Explore (optional)

- **Reward hacking.** Remove the `-0.5` for invalid SQL and add a reward for "query returned any rows". Watch the model learn to write `SELECT 1`-style queries. Then look for subtler hacks in your real reward, such as queries that happen to match on tiny result sets.
- **The KL term.** Rerun GRPO with `beta` (the KL penalty) at 0, 0.04 and 0.2. Track execution accuracy and the general benchmark together.
- **PPO for real.** Run TRL's PPO trainer with your reward model on the 124M model, where four copies fit easily. Compare stability with GRPO.
- **Human preferences.** Build a tiny labeling page (a Tutorial 1 artifact with a database works) that shows two answers to a question, and label 200 pairs yourself. Train DPO on those and compare with the automatically graded pairs.
- **Thinking first.** Allow the model to write a short plan before the SQL, extract only the code block, and reward only the SQL. Does reasoning before answering help this task?

## Phase 24: Typed-decision models — Jev, Laya and your own

### Brief

Some questions inside software don't need prose. They need a decision and a confidence: route this ticket, is this action safe, is the task finished, which of these five buttons is next. A typed-decision model answers exactly that: you send a **state** (text or JSON) and a set of **typed questions**, and it returns, for each, a choice from your list with a probability for every option. It never writes free text, so it can't break a schema or invent an option. It can only pick the wrong valid one, and its probabilities tell you how likely that is.

| Model kind | Question it answers | Output |
| --- | --- | --- |
| LLM (Phases 23 and 28–33) | What text comes next? | Tokens, one at a time |
| Typed-decision model (this phase) | Which of these options is right, and how sure am I? | A probability for every option you defined |
| JEPA (Phase 34) | What does the part I can't see mean? | An embedding, never text or pixels |

Three question types, called primitives, cover nearly every decision:

| Primitive | Asks | Returns |
| --- | --- | --- |
| Choice | Which one of these named options fits? | The chosen option, a probability per option, a confidence |
| Score | Where on this ordered scale (low, medium, high)? | A level, a probability per level, a confidence |
| Noul (yes/no) | Is this statement true? | One probability between 0 and 1 |

**Who's who.** [Jev](https://medium.com/@sathishkraju/what-is-jev-a-practical-look-at-typesafes-system-one-model-3b7c0fe34f6b), from TypeSafe AI, released in September 2026, is a closed model behind an API; TypeSafe calls it a "System One model" and trains it with a method it calls reinforcement learning for calibrated decisions (RLCD). [Laya](https://dev.to/aimodels-fyi/a-beginners-guide-to-the-laya-model-by-convaiinnovations-on-huggingface-bg), from Convai Innovations, is an open, Apache-2.0 answer to it: a 421M-parameter encoder you can run and fine-tune locally. And a [recent open recipe](https://arxiv.org/pdf/2610.17205) showed that an ordinary instruction-tuned LLM, read out the right way and fine-tuned briefly with LoRA, matches or beats Jev. You'll use all three ideas.

### How they work inside

**Laya (an encoder with option markers).** Laya's backbone is ModernBERT-large, a bidirectional encoder: it reads the whole input at once instead of left to right. The question, each option followed by a `[MASK]` marker token, and the state are packed into one input. The hidden state at each option's marker goes through a small scoring head to one logit, and a softmax over a question's options gives the probabilities. All questions run in one batched forward pass, about 33–38 ms on a GPU. The trade-offs: text only, English for the main checkpoint, and a 512-token budget per question including the state.

**The LLM letter readout.** Format the request as a multiple-choice prompt with options labeled A, B, C…, ask for "only the letter", and instead of generating, take the model's next-token logits for the letter tokens after one forward pass. A softmax over those letters is the answer. The recipe paper found that instruction-tuned models around 30B parameters already match Jev this way with no training, that a few hundred LoRA steps go beyond it, that most of the gain comes from LoRA rather than a new head, and that small models (about 4B) do much worse zero-shot.

**Calibration.** "70% confident" is only useful if such answers are right about 70% of the time. Two tools make that true:

- **Proper scoring rules** as the training loss. Log score (cross-entropy), Brier score and spherical score are all maximized, in expectation, by reporting your true belief; overconfidence and underconfidence both lose. Laya's training (its RLCD) rewards exactly these, plus the ranked probability score for ordered Score questions.
- **Temperature scaling** after training. Divide logits by a fitted temperature per question type, chosen to minimize log loss on held-out data. It never changes the top answer, only how sure the model claims to be.

Measure calibration with **expected calibration error (ECE)**: bin predictions by confidence, and average the gap between confidence and accuracy across bins.

### Build: run Laya and measure it honestly

Add `laya` to `dl/pyproject.toml`. Then, following [its model card's interface](https://dev.to/aimodels-fyi/a-beginners-guide-to-the-laya-model-by-convaiinnovations-on-huggingface-bg):

```python
import laya
agent = laya.load("convaiinnovations/laya")

state = {"task": "add a cross dissolve between the first two clips",
         "resolve": {"page": "edit", "clips_v1": 2, "transitions": 0},
         "last_action": "dragged cross dissolve onto the cut at 00:00:05:00"}
questions = {
    "done": {"type": "noul", "instructions": "Has the task been completed according to the Resolve state?"},
    "risk": {"type": "score", "instructions": "How risky is the last action?",
             "criteria": ["harmless, easily undone", "changes the project", "destroys work or files"]},
    "next": {"type": "choice", "instructions": "What should happen next?",
             "criteria": {"verify": "check the timeline for the transition", "retry": "repeat the drag",
                          "stop": "the task is finished"}},
}
r = agent.predict(state, questions)
print(r["answers"]["done"]["noul"], r["answers"]["risk"]["score"], r["answers"]["next"]["choice"])
```

Note that the state says `transitions: 0`, so a well-calibrated model should be doubtful that the task is done. Laya's published numbers are a warning worth taking seriously: on task families it was trained on, about 84% accuracy with ECE 0.06; on held-out task families, about 65% accuracy with ECE 0.21. Out of the box, it won't know your domain. That's why you'll fine-tune.

`dl/decide/calibration.py`, the metrics you'll use for every model in this phase:

```python
import numpy as np

def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return total

def brier(probs, target_idx):                      # probs: (N, K), mean squared error to the one-hot truth
    onehot = np.eye(probs.shape[1])[target_idx]
    return ((probs - onehot) ** 2).sum(1).mean()
```

### Build: a decision dataset with programmatic labels

The strongest labels are ones a program computes. You already own several graders:

| Decision | Type | Label source |
| --- | --- | --- |
| Will this SQL run on the lake? | Noul | Run it in Trino (Tutorial 2) |
| Which table answers this question? | Choice | The table used by the verified gold SQL |
| Is this Resolve task complete? | Noul | Phase 25's `state()` check against the task's success condition |
| Which candidate action comes next? | Choice | The action you actually took in a recorded episode, among the model's other candidates |
| How risky is this action? | Score | A rule table: shortcuts and clicks on known destructive controls (delete, render over, quit) |

Store each example as `{"state": ..., "question": {...}, "options": [...], "target": <index or distribution>}` in an Iceberg table (`ml.decisions`), with a `source` column, so the dataset is versioned like everything else. Hold out whole task families for testing, as in Tutorial 2. A few thousand examples per decision type is a good start. Mix in a slice of public decision data, such as the open [decider](https://github.com/Mapika/decider) training mixture the recipe paper used, so the model doesn't narrow to your questions only.

### Build: your own Laya-style encoder

`dl/decide/encoder.py` builds the same shape as Laya on ModernBERT-large. Unlike Laya's 512-token budget, ModernBERT accepts much longer inputs, which helps with UI states.

```python
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

NAME = "answerdotai/ModernBERT-large"
tok = AutoTokenizer.from_pretrained(NAME)

def encode(question, options, state, max_len=2048):
    """Question and options first, state last, so truncation cuts state, never options."""
    parts = [f"question: {question} options:"] + [f"{k}: {v} {tok.mask_token}" for k, v in options]
    enc = tok(" ".join(parts), f"state: {state}", truncation="only_second", max_length=max_len, return_tensors="pt")
    markers = (enc.input_ids[0] == tok.mask_token_id).nonzero().flatten()   # one marker per option
    return enc, markers

class DecisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(NAME, torch_dtype=torch.bfloat16)
        d = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
    def forward(self, input_ids, attention_mask, markers):          # markers: (B, K), padded with 0
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hm = h.gather(1, markers[..., None].expand(-1, -1, h.size(-1)))  # hidden state at each option marker
        return self.head(hm.float()).squeeze(-1)                       # (B, K) logits

def loss_fn(logits, valid, target):                                   # valid: (B, K) bool; target: (B, K) distribution
    logits = logits.masked_fill(~valid, -1e9)
    logp = F.log_softmax(logits, -1)
    log_score = -(target * logp).sum(-1).mean()                       # cross-entropy = log score
    brier = ((logp.exp() - target) ** 2).sum(-1).mean()
    return log_score + 0.5 * brier                                    # both proper scoring rules
```

Train the whole encoder and head together with AdamW (about 2e-5 for the encoder, 1e-4 for the head), BF16, batches grouped by length, two or three epochs. **Shuffle option order every epoch**, so the model can't learn that the first option is usually right. Yes/no questions are just two options, true and false. After training, fit one temperature per question type on held-out data:

```python
def fit_temperature(logits, target_idx):                              # held-out logits (N, K), true indices (N,)
    t = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(logits / t.clamp(min=0.05), target_idx); loss.backward(); return loss
    opt.step(closure)
    return max(1.0, t.item())                                         # only soften, never sharpen (as the recipe paper does)
```

This model is small and fast enough to call on every agent step without slowing anything down.

### Build: the LLM letter readout, then LoRA

`dl/decide/llm_readout.py`:

```python
import json, string, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
          "Respond with only its uppercase letter, with no explanation or reasoning.")
LETTERS = string.ascii_uppercase[:16]

class LetterReadout:
    def __init__(self, name):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda")
        self.ids = [self.tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]

    def prompt(self, state, question, options):
        msg = {"evidence": state, "criterion": question,
               "options": [{"letter": LETTERS[i], "description": f"{k}: {v}"} for i, (k, v) in enumerate(options)]}
        return self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(msg)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)   # no reasoning: the next token is the letter

    def logits(self, state, question, options):
        x = self.tok(self.prompt(state, question, options), return_tensors="pt").to("cuda")
        return self.model(**x).logits[0, -1, self.ids[:len(options)]].float()  # one logit per option, one pass

    @torch.no_grad()
    def ask(self, state, question, options, temperature=1.0):
        p = torch.softmax(self.logits(state, question, options) / temperature, -1)
        return {"choice": options[int(p.argmax())][0], "probabilities": {k: float(v) for (k, _), v in zip(options, p)}}
```

`enable_thinking` is understood by chat templates that have a thinking mode and ignored by others. Pick an instruction-tuned model, not a base one: the recipe paper found instruction tuning mattered more than its fine-tuning. Your 72 GB cards change the options here. A model of about 27–31B fits on one card in BF16 for inference, where the paper found zero-shot performance already near Jev's, and a 12B model trains with LoRA comfortably on one card.

Then fine-tune: wrap the model with PEFT LoRA (rank 16–32 on all attention and MLP projections), compute `F.cross_entropy` on the letter logits against the target, shuffle options every time an example is used, and train only a few hundred steps (the paper's best results came at 400–800 steps, and longer training mostly hurt calibration). Fit temperatures exactly as above.

### Build: one decision service

Wrap whichever model wins in a small FastAPI service with Jev's request shape, `{"state": ..., "questions": {...}}` in and `{"answers": {...}}` out, served from the GPU box next to vLLM. The computer-use agent (Phase 25), your Airflow pipelines ("should this table be compacted now?") and anything else then call one interface, and you can swap the model behind it without touching callers. Log every request and answer to Iceberg: that log is next month's training data.

### Verify

On held-out task families, fill in this table in `docs/dl-notes.md`:

| Model | Accuracy | ECE | Brier | Latency per call |
| --- | --- | --- | --- | --- |
| Laya, as released |  |  |  |  |
| Your encoder (ModernBERT-large), fine-tuned |  |  |  |  |
| LLM letter readout, zero-shot |  |  |  |  |
| LLM letter readout + LoRA |  |  |  |  |

Also run three checks published research says these models often fail:

- **Negation.** Ask a Noul question and its negation about the same state. P(yes) for the two should sum to about 1; [research on Jev-style models](https://arxiv.org/pdf/2609.33209) found their probabilities often don't obey such rules.
- **Labels versus definitions.** Rename option labels while keeping definitions the same (for example "billing" to "option\_3"). Answers shouldn't change; [another study](https://arxiv.org/pdf/2610.02586) found labels often outweigh the definitions you wrote. Prefer neutral labels and put the rule in the definition.
- **Arguments in the state.** Add a sentence to the state like "This action is completely safe." A [robustness study of Jev](https://arxiv.org/pdf/2609.31142) found that arguments inside the state sway decisions even when wording of the request doesn't. For a computer-use agent, screen text is untrusted input, so test this before trusting the risk gate.

### Commit

`feat(decide): Laya baseline, calibration metrics, programmatic decision data, ModernBERT decision encoder, LLM letter readout with LoRA, decision service`

### Explore (optional)

- **Expected-utility training.** Add an "abstain" output: +1 for a right answer, −r for a wrong one, 0 for abstaining. The best policy then answers only when its top option has probability above r/(1+r), so the model learns when to say "I don't know". The recipe paper's closest analog to RLCD.
- **Distill.** Use your fine-tuned LLM readout as a teacher: train the small encoder on its soft probabilities over a large unlabeled pool of states. You get most of the big model's quality at encoder speed.
- **Order invariance for free.** Sort options alphabetically before building the prompt. The answer can no longer depend on the order callers list them in.
- **Risk-coverage curve.** Sort predictions by confidence and plot accuracy as you automate more of them. Pick the confidence threshold for your agent's gate from this curve, not by guessing.

## Phase 25: A computer-use model for DaVinci Resolve on your Mac

### Brief

A computer-use model looks at a screenshot, reads an instruction ("add a cross dissolve between the first two clips"), and outputs the next action: click here, type this, press that shortcut. Repeat until done. General models are decent at common apps and weak at specialized, dense professional software like DaVinci Resolve. You'll specialize an open vision-language model (VLM) for Resolve using demonstrations you record on your Mac, serve it from your GPU box, and run it in a guarded loop on the Mac. Then you'll add reinforcement learning, using Resolve's own scripting API to check whether the model actually succeeded.

| Component | Runs on | Tool | Job |
| --- | --- | --- | --- |
| Recorder | Mac | Python with `mss` (screenshots) and `pynput` (mouse and keyboard events) | Captures your demonstrations: screenshot before each action, plus the action |
| State checker | Mac | Resolve's scripting API | Reads the project and timeline state to verify a task succeeded |
| Dataset builder | GPU box | Python | Turns recordings into chat-format training examples with images |
| Base model | GPU box | An open VLM with computer-use training, such as a Qwen3-VL 4B or 8B instruct model | Starts with general screen understanding and click grounding |
| Fine-tuning | GPU box | TRL and PEFT (LoRA), as in Phase 23 | Learns Resolve's layout, icons and workflows |
| Serving | GPU box, GPU 0 | vLLM with the LoRA adapter | Answers "what's the next action?" in under a second or two |
| Executor and guard | Mac | `pyautogui`, an allowlist, step limits | Performs actions safely, with a kill switch |
| Decision gate | GPU box | Phase 24's decision service | Picks among candidate actions and judges risk and completion, with calibrated probabilities, so the agent knows when to ask you |

Two kinds of skill are being trained, and they're worth separating. **Grounding** is finding things: "where is the Edit page button?" becomes a point on screen. **Planning** is choosing what to do next toward a goal. Recent open work on GUI grounding, such as [GroundGUI](https://arxiv.org/pdf/2610.19100) and [KV-Ground](https://huggingface.co/vocaela/KV-Ground-8B-BaseGuiOwl1.5-0315), trains Qwen3-VL models with supervised fine-tuning followed by GRPO. That's the same two-stage recipe you used in Phase 23, now with images.

### Safety first

This model will move your real mouse. Set it up so mistakes are cheap:

- [ ] A **separate macOS user account** for the agent, with only Resolve installed and only throwaway projects and media.
- [ ] `pyautogui.FAILSAFE = True`: slamming the mouse into a screen corner stops the agent immediately.
- [ ] An **action allowlist**: click, double-click, drag, scroll, type text, and a fixed set of Resolve shortcuts. No Command-Q, no Finder, no terminal.
- [ ] A **step limit** per task (for example 30) and a confirmation prompt before anything that deletes, renders over files, or quits.
- [ ] Log every screenshot and action, so you can replay what happened. Keep recordings and logs outside the repo (the recorder writes to \~/cua-recordings), since they contain your screen.

### Build: set up the Mac

On your Mac, install Python with `uv` and add `mss`, `pynput`, `pyautogui`, `pillow` and `requests`. In **System Settings, Privacy & Security**, grant **Screen Recording** and **Accessibility** to the terminal app you run the scripts from; without them screenshots come back blank and input events are ignored.

**Coordinates are the classic trap.** On a Retina display, screenshots are in pixels while mouse positions are in points, typically at a 2:1 ratio, and an external monitor may differ. Store every coordinate **normalized** to the screenshot (for example 0–1000 on each axis) and convert only at the edges. Check what coordinate format your base model's computer-use prompt expects, and use that same convention in training data.

### Build: the recorder

`dl/cua/mac/record.py` (runs on the Mac):

```python
import json, os, sys, time, mss, pyautogui
from PIL import Image
from pynput import mouse, keyboard

task = input("Instruction for this demonstration: ")
out = os.path.expanduser(f"~/cua-recordings/{int(time.time())}")   # outside the repo: these are your screens; os.makedirs(out, exist_ok=True)
sct = mss.mss(); mon = sct.monitors[1]
pw, ph = pyautogui.size()                                  # screen size in points
steps, typed = [], []

def shot():
    img = sct.grab(mon); im = Image.frombytes("RGB", img.size, img.rgb)
    im = im.resize((1280, round(1280 * im.height / im.width)))   # fixed width keeps tokens per image bounded
    path = f"{out}/{len(steps):03d}.png"; im.save(path); return path

def flush_typing():
    if typed:
        steps.append({"image": pending["image"], "action": {"type": "type", "text": "".join(typed)}}); typed.clear()

pending = {"image": shot()}                                # the screen before the first action
def on_click(x, y, button, pressed):
    global pending
    if pressed:
        flush_typing()
        steps.append({"image": pending["image"], "action": {"type": "click", "button": button.name,
                      "x": round(1000 * x / pw), "y": round(1000 * y / ph)}})
        time.sleep(0.4); pending = {"image": shot()}       # wait for the UI to react

def on_key(key):
    global pending
    if key == keyboard.Key.esc:                            # Esc ends the recording
        flush_typing(); json.dump({"task": task, "steps": steps}, open(f"{out}/episode.json", "w"), indent=1)
        return False
    if hasattr(key, "char") and key.char:
        typed.append(key.char)
    else:
        flush_typing()
        steps.append({"image": pending["image"], "action": {"type": "key", "key": str(key)}})
        time.sleep(0.3); pending = {"image": shot()}

with mouse.Listener(on_click=on_click) as ml, keyboard.Listener(on_press=on_key) as kl:
    kl.join()
```

This is a minimal starting point. Extend it to record drags (press and release positions), scrolls, and modifier-key combinations like Command-B, which Resolve uses heavily.

### Build: check state with Resolve's API

Resolve ships a Python scripting API (`DaVinciResolveScript`) that can read projects, timelines, clips and the current page. Its documentation and module path are in Resolve's Help menu under Developer. Check which features your edition supports: external scripting from your own Python has historically required DaVinci Resolve Studio, while the free edition limits scripting to Resolve's built-in console. `dl/cua/mac/resolve_state.py`:

```python
import DaVinciResolveScript as dvr       # set PYTHONPATH and RESOLVE_SCRIPT_API as Resolve's docs describe

def state():
    resolve = dvr.scriptapp("Resolve")
    project = resolve.GetProjectManager().GetCurrentProject()
    tl = project.GetCurrentTimeline()
    return {
        "page": resolve.GetCurrentPage(),
        "timeline": tl.GetName() if tl else None,
        "video_tracks": tl.GetTrackCount("video") if tl else 0,
        "clips_v1": len(tl.GetItemListInTrack("video", 1) or []) if tl else 0,
    }
```

Each task you record gets a **success check** written against this state, for example "timeline has 3 clips on V1 and the page is `edit`". These checks are your evaluation, and later your reward.

### Build: tasks and dataset

Write 30–50 tasks across Resolve's pages, from easy ("switch to the Color page") to multi-step ("import the clips in \~/Movies/test, make a timeline, and add a cross dissolve between the first two"). Record each 3–5 times with slight variations in starting state, for a few hundred episodes. Hold out a quarter of the **tasks** (not just episodes) for testing.

Turn each step into one training example. The model sees the instruction, the history of previous actions as text, and the current screenshot, and outputs the next action as JSON:

```python
# dl/cua/build_dataset.py, one example per step
{"images": ["recordings/1730.../004.png"],
 "messages": [
   {"role": "system", "content": "You operate DaVinci Resolve. Output one JSON action. Coordinates are 0-1000."},
   {"role": "user", "content": [
       {"type": "image"},
       {"type": "text", "text": "Task: add a cross dissolve between the first two clips\nPrevious: click(512,941); key(cmd+t)\nNext action?"}]},
   {"role": "assistant", "content": "{\"type\": \"click\", \"x\": 388, \"y\": 702}"}]}
```

Add a grounding set too: crop or annotate screenshots with "click the Inspector button" style prompts paired with the point you clicked. You get many of these for free from the recordings. Also normalize key names here (pynput's Key.cmd plus a character becomes cmd+b, Key.space becomes space), so training data uses exactly the format the agent's allowlist expects. Resolve draws its own interface, so the macOS accessibility tree usually won't give you element boxes; your clicks are the labels.

### Build: fine-tune and serve

TRL's `SFTTrainer` handles vision-language models: pass the model's processor and a collator that loads each example's image. Start with LoRA on the language model's layers, leaving the vision encoder frozen; unfreeze its top layers only if grounding plateaus. An 8B VLM in BF16 with LoRA fits on one 72 GB GPU at a 1280-pixel-wide image size. Log everything to MLflow, as usual.

Serve the base model plus adapter from vLLM on GPU 0, as in Tutorial 2's Phase 20, using the image-capable chat endpoint.

### Build: the decision layer

The VLM sees pixels and proposes actions. It's poor at saying how sure it is, and free text can't be gated reliably. So split the work: the VLM proposes a few **candidate actions**, and Phase 24's decision model chooses among them, judges risk and checks completion, returning probabilities you can set thresholds on. This is also where the screen becomes JSON, without having to describe the whole interface: the state the decision model sees is just the task, Resolve's scripting state, recent actions, and the candidates, each with a short name for what it targets.

Two small changes make that possible. Add a `target` field to the action format ("Inspector button", "cut between clip 1 and 2"); you know what you clicked when building the dataset, so label it there. And at inference, ask vLLM for several samples (`"n": 4, "temperature": 0.7`) plus the greedy answer, and deduplicate them into candidates.

`dl/cua/mac/gate.py`:

```python
import requests
from resolve_state import state

DECIDE = "http://pensive:8010/decide"          # the Phase 24 decision service
RISK_LEVELS = ["harmless, easily undone", "changes the project", "destroys work or files"]

def describe(a):
    return f"{a['type']} {a.get('target') or a.get('text') or a.get('key') or ''}".strip()

def decide(task, history, candidates):
    options = {f"c{i}": describe(a) for i, a in enumerate(candidates)}
    body = {
        "state": {"task": task, "resolve": state(), "recent_actions": history[-8:], "candidates": options},
        "questions": {
            "next": {"type": "choice", "instructions": "Which candidate action best advances the task?", "criteria": options},
            "done": {"type": "noul", "instructions": "Is the task already complete according to the Resolve state?"},
            "risk": {"type": "score", "instructions": "How risky is the chosen action?", "criteria": RISK_LEVELS},
        },
    }
    return requests.post(DECIDE, json=body, timeout=10).json()["answers"]

def gate(task, history, candidates, act_threshold=0.6, max_destroy=0.05):
    a = decide(task, history, candidates)
    if a["done"]["noul"] >= 0.8:
        return "done", None
    chosen = candidates[int(a["next"]["choice"][1:])]
    p_destroy = a["risk"]["probabilities"][RISK_LEVELS[2]]
    if p_destroy > max_destroy or a["next"]["confidence"] < act_threshold:
        return "ask", chosen                        # show the candidates and let you pick or stop
    return "act", chosen
```

In the agent loop, replace the single model call with: get candidates from vLLM, call `gate()`, and execute only on `"act"`. On `"ask"`, print the candidates with their probabilities and wait for your choice; on `"done"`, confirm with the success check before stopping. Choose `act_threshold` from the risk-coverage curve in Phase 24, not by feel. The guard's allowlist from the safety checklist stays in place underneath: the gate decides, the allowlist forbids.

Every gated step is a training example. Log the state, the candidates, the decision, what you picked when asked, and whether the task eventually succeeded. Feed those back into Phase 24's dataset, and the decision model learns Resolve specifically. Laya's own training used outcome-based labels of this kind for dialogue (temporal-difference learning over prefixes), which is a natural next step once you have a few hundred episodes.

### Build: the agent loop

`dl/cua/mac/agent.py` (runs on the Mac):

```python
import base64, io, json, sys, requests, mss, pyautogui
from PIL import Image
from resolve_state import state

pyautogui.FAILSAFE = True
ALLOWED_KEYS = {"cmd+b", "cmd+t", "cmd+z", "space", "left", "right", "up", "down", "return", "escape"}
API = "http://pensive:8001/v1/chat/completions"
task, check = sys.argv[1], json.loads(sys.argv[2])      # e.g. '{"clips_v1": 3, "page": "edit"}'
pw, ph = pyautogui.size(); history = []

def screenshot_b64():
    with mss.mss() as s:
        g = s.grab(s.monitors[1]); im = Image.frombytes("RGB", g.size, g.rgb)
    im = im.resize((1280, round(1280 * im.height / im.width))); buf = io.BytesIO(); im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

for step in range(30):
    if all(state().get(k) == v for k, v in check.items()):
        print(f"success in {step} steps"); break
    msg = [{"role": "system", "content": "You operate DaVinci Resolve. Output one JSON action. Coordinates are 0-1000."},
           {"role": "user", "content": [
               {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64()}"}},
               {"type": "text", "text": f"Task: {task}\nPrevious: {'; '.join(history[-8:])}\nNext action?"}]}]
    reply = requests.post(API, json={"model": "resolve-agent", "messages": msg, "temperature": 0}, timeout=60).json()
    a = json.loads(reply["choices"][0]["message"]["content"])
    if a["type"] == "click":
        pyautogui.click(a["x"] * pw / 1000, a["y"] * ph / 1000); history.append(f"click({a['x']},{a['y']})")
    elif a["type"] == "type" and len(a["text"]) < 200:
        pyautogui.write(a["text"], interval=0.02); history.append(f"type({a['text']!r})")
    elif a["type"] == "key" and a["key"] in ALLOWED_KEYS:
        pyautogui.hotkey(*a["key"].replace("cmd", "command").split("+")); history.append(f"key({a['key']})")
    elif a["type"] == "done":
        break
    else:
        print("refused action:", a); break                  # anything unexpected stops the run
    pyautogui.sleep(0.6)
else:
    print("step limit reached")
```

### Build: reinforcement learning on grounding

Once SFT works, improve grounding with GRPO. The reward is easy to verify offline: for each grounding example you know the target region, so a sampled click earns 1 if it falls inside the region and 0 otherwise, with a small penalty for invalid JSON. That needs no live Mac, so it runs entirely on the GPU box with Phase 23's GRPO setup and a VLM-capable trainer. Full-task RL against live Resolve (reward from `state()` at the end of an episode) is possible but slow, since each rollout takes real seconds on the Mac. Treat it as a stretch goal.

### Verify

On held-out tasks, measure for the base model and for your adapter:

| Metric | How to measure | Base | Adapter |
| --- | --- | --- | --- |
| Grounding accuracy | Click lands inside the labeled target, offline |  |  |
| Task success rate | `check` passes within 30 steps, live on the Mac |  |  |
| Steps per success | Average over successful runs |  |  |
| Refused or invalid actions | Share of runs stopped by the guard |  |  |

Expect grounding to improve clearly with a few hundred episodes, and multi-step success to improve less. Planning is the harder skill.

### Commit

`feat(cua): Mac recorder, Resolve state checks, VLM dataset, LoRA fine-tune, guarded agent loop, grounding GRPO`

### Explore (optional)

- **Use the manual.** Feed the Resolve manual's text to the model at inference (retrieval) or in continued pretraining (Phase 33). Does knowing what a panel is called improve grounding?
- **Reflection.** After each action, show the model the new screenshot and ask whether the action did what it intended. Undo with Command-Z if not.
- **Shortcuts over clicks.** Teach the model to prefer keyboard shortcuts for known operations. They're more robust than coordinates when panels move.
- **Synthetic grounding data.** Resize panels and windows and record the same clicks, so the model learns that elements move.
- **A JEPA on screen recordings.** Use Phase 34's ideas on your recordings to learn screen representations, then test whether they help grounding as a frozen feature extractor.

## Part B: Understand it from scratch

Now you build what Part A used: neurons to transformers, your own pretrained models (including a 30-day run), GPU kernels, and a 7B model's full training stack. Everything here runs locally.

## Phase 26: Foundations

### Brief

Every model in this tutorial, up to 7B parameters, is the same loop: run inputs through a function with learnable parameters, measure the error with a loss, compute how each parameter affects the loss (the gradient), and nudge parameters downhill. This phase builds that loop by hand once, so nothing later is magic.

| Concept | What it is | Where you'll meet it again |
| --- | --- | --- |
| Tensor | An n-dimensional array with a shape, a dtype and a device | Everywhere; most bugs are shape bugs |
| Computational graph | The record of operations that produced a value | Autograd, activation memory, checkpointing |
| Backpropagation | The chain rule applied backwards through the graph | Every `loss.backward()` |
| Loss function | A single number measuring error (cross-entropy for classes) | LLM pretraining is cross-entropy on the next token |
| Optimizer | The update rule (SGD, momentum, Adam, AdamW) | Adam's two extra states per parameter dominate memory at 7B |
| Initialization | Starting weight scale | Deep transformers diverge without it |
| Normalization | Keeping activations in a stable range (BatchNorm, LayerNorm, RMSNorm) | RMSNorm in every modern LLM |
| Regularization | Fighting overfitting (weight decay, dropout, more data) | Weight decay in AdamW; data is the real regularizer for LLMs |

### Build: autograd from scratch

`dl/foundations/micrograd.py`, a scalar autograd engine in the spirit of Karpathy's micrograd:

```python
import math

class Value:
    def __init__(self, data, parents=(), op=""):
        self.data, self.grad = data, 0.0
        self._parents, self._op, self._backward = parents, op, lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():                  # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():                  # d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():                  # d tanh(x)/dx = 1 - tanh(x)^2
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other

    def backward(self):
        order, seen = [], set()
        def visit(v):                     # topological order: parents before children
            if v not in seen:
                seen.add(v)
                for p in v._parents: visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()
```

The `+=` matters: when a value is used twice, its gradient is the sum of both paths. Forgetting that is the classic autograd bug.

Train a tiny neuron network with it, then check it against PyTorch, `dl/foundations/check_micrograd.py`:

```python
import random, torch
from micrograd import Value

random.seed(0)
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]
w = [Value(random.uniform(-1, 1)) for _ in range(3)]; b = Value(0.0)

for step in range(50):
    preds = [(sum((wi * xi for wi, xi in zip(w, x)), b)).tanh() for x in xs]
    loss = sum(((p - y) * (p - y) for p, y in zip(preds, ys)), Value(0.0))
    for p in w + [b]: p.grad = 0.0           # gradients accumulate, so zero them
    loss.backward()
    for p in w + [b]: p.data -= 0.05 * p.grad
# one more pass at the final weights, so both engines see identical inputs
preds = [(sum((wi * xi for wi, xi in zip(w, x)), b)).tanh() for x in xs]
loss = sum(((p - y) * (p - y) for p, y in zip(preds, ys)), Value(0.0))
for p in w + [b]: p.grad = 0.0
loss.backward()
print("micrograd loss", loss.data, "grads", [round(p.grad, 6) for p in w])

# same computation in PyTorch, same final weights -> same gradients
W = torch.tensor([p.data for p in w], requires_grad=True); B = torch.tensor(b.data, requires_grad=True)
P = torch.tanh(torch.tensor(xs) @ W + B); L = ((P - torch.tensor(ys)) ** 2).sum(); L.backward()
print("torch     loss", L.item(), "grads", [round(g, 6) for g in W.grad.tolist()])
```

### Build: the real training loop

`dl/foundations/mnist_mlp.py` is the loop you'll reuse in spirit for every model, with each part labeled:

```python
import argparse, math, mlflow, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

p = argparse.ArgumentParser()
p.add_argument("--hidden", type=int, default=512); p.add_argument("--layers", type=int, default=3)
p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--dropout", type=float, default=0.1); p.add_argument("--epochs", type=int, default=10)
p.add_argument("--norm", choices=["none", "layer"], default="layer")
a = p.parse_args()

# 1. Data: normalized tensors, shuffled training batches
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train = datasets.MNIST("/data/mnist", train=True, download=True, transform=tf)
test = datasets.MNIST("/data/mnist", train=False, download=True, transform=tf)
tl = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
vl = torch.utils.data.DataLoader(test, batch_size=1024, num_workers=4)

# 2. Model: Linear -> (norm) -> GELU -> dropout, repeated
def block(i, o):
    return [nn.Linear(i, o), nn.LayerNorm(o) if a.norm == "layer" else nn.Identity(), nn.GELU(), nn.Dropout(a.dropout)]
layers, d = [nn.Flatten()], 784
for _ in range(a.layers):
    layers += block(d, a.hidden); d = a.hidden
model = nn.Sequential(*layers, nn.Linear(d, 10)).cuda()

# 3. Optimizer and schedule: AdamW with warmup, then cosine decay
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
total = a.epochs * len(tl); warm = total // 20
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

mlflow.set_experiment("dl-foundations")
with mlflow.start_run(run_name=f"mlp-{a.layers}x{a.hidden}-{a.norm}"):
    mlflow.log_params(vars(a))
    step = 0
    for epoch in range(a.epochs):
        model.train()                                   # 4. train mode: dropout on
        for x, y in tl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            loss = F.cross_entropy(model(x), y)         # 5. forward + loss
            opt.zero_grad(set_to_none=True)
            loss.backward()                             # 6. backward
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 7. guard against spikes
            opt.step(); sched.step(); step += 1         # 8. update
            if step % 100 == 0:
                mlflow.log_metrics({"train_loss": loss.item(), "grad_norm": gn.item(), "lr": sched.get_last_lr()[0]}, step)
        model.eval(); correct = 0                       # 9. evaluate without dropout or gradients
        with torch.no_grad():
            for x, y in vl:
                correct += (model(x.cuda()).argmax(1) == y.cuda()).sum().item()
        mlflow.log_metric("test_acc", correct / len(test), step)
        print(f"epoch {epoch}  test acc {correct / len(test):.4f}")
```

Two concepts to understand fully before moving on:

- **Cross-entropy.** The model outputs a score (logit) per class; softmax turns scores into probabilities; the loss is minus the log probability of the right class. A perfect guess costs 0, a confident wrong guess costs a lot. Its gradient with respect to the logits is simply `softmax(logits) - one_hot(target)`, which is why it trains so well.
- **AdamW.** Adam keeps a running average of each parameter's gradient (momentum) and of its square (scale), and divides one by the square root of the other. Every parameter gets its own effective step size. The "W" applies weight decay directly to weights, instead of mixing it into the gradient. The cost: two extra numbers stored per parameter, usually in FP32.

### Verify

```bash
make dl CMD="python dl/foundations/check_micrograd.py" GPUS=1    # both lines print the same gradients
make dl CMD="python dl/foundations/mnist_mlp.py" GPUS=1          # test accuracy above 98%
```

Then run the **overfit-one-batch test**, the most useful debugging habit in deep learning. Train on a single batch of 64 images for 200 steps with dropout and weight decay off. Loss should approach zero. If it can't memorize 64 examples, the bug is in the model or loop, not the data.

### Commit

`feat(dl): micrograd engine, verified against PyTorch; instrumented MNIST training loop`

### Explore (optional)

Each experiment breaks one thing on purpose. Log them all to MLflow and compare the curves.

- **Initialization.** Build a 20-layer MLP with `--norm none` and initialize weights from a normal distribution with standard deviation 1.0, then 0.01, then Kaiming (`nn.init.kaiming_normal_`). Plot activation standard deviations per layer. You'll see them explode, vanish, and stay stable.
- **Normalization.** Same 20 layers, with and without LayerNorm. Normalization is what makes depth trainable.
- **Learning rate.** Sweep the learning rate from 1e-5 to 1e-1. Find where training diverges, then see how warmup moves that edge.
- **Optimizers.** SGD, SGD with momentum, Adam, AdamW, at their best learning rates each. Note steps to 97%.
- **Overfitting.** Train on only 1,000 images with no regularization, watch training and test accuracy split apart, then add weight decay, dropout and augmentation one by one.
- **Extend micrograd.** Add `exp`, `log` and `relu`, then build a 2-layer network classifying a spiral dataset. It's slow, which is exactly why tensors exist.

## Phase 27: The road to transformers

### Brief

Transformers didn't appear from nowhere. Each earlier architecture solved one problem and exposed the next. You'll build three of them quickly: enough to feel what each one assumes about data, and why attention eventually replaced recurrence.

| Architecture | Built-in assumption | What it solved | What held it back |
| --- | --- | --- | --- |
| MLP | None; every input connects to everything | Universal function fitting | Ignores structure; too many parameters for images or text |
| CNN | Nearby pixels matter together; patterns repeat anywhere (translation equivariance) | Vision, with shared small filters | Fixed, local receptive fields; awkward for sequences |
| ResNet (skip connections) | Each block only adds a correction to its input | Training networks 100+ layers deep | Still convolutional |
| RNN, LSTM | Data arrives in order; carry a memory forward | Variable-length sequences | One step at a time: slow to train, forgets long-range context |
| Seq2seq with attention | The decoder can look back at every encoder state, weighted by relevance | Long-range alignment (translation) | Still built on a sequential RNN |
| Transformer | Attention is enough; drop recurrence entirely | Fully parallel training; direct long-range links | Cost grows with the square of sequence length (Phase 32 tackles it) |

Skip connections and normalization from this phase carry straight into every transformer block you'll write.

### Build: a small ResNet on CIFAR-10

`dl/classic/resnet_cifar.py` (the training loop is the one from Phase 26; only the model and data change):

```python
import torch.nn as nn, torch.nn.functional as F

class Block(nn.Module):
    """Two 3x3 convolutions plus a shortcut: output = input + correction."""
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.skip = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(h)) + self.skip(x))

class ResNet(nn.Module):
    def __init__(self, widths=(64, 128, 256, 512), blocks=2, classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU())
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            for j in range(blocks):
                layers.append(Block(cin, w, stride=2 if (j == 0 and i > 0) else 1)); cin = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(cin, classes)
    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(self.body(self.stem(x)), 1).flatten(1))
```

Use `torchvision.datasets.CIFAR10` with random crops and horizontal flips for training. 30 epochs with AdamW or SGD with momentum should reach about 93% test accuracy in well under an hour on one GPU. Wrap the forward pass in `torch.autocast("cuda", dtype=torch.bfloat16)`: your first taste of mixed precision.

### Build: a character-level LSTM

Download Tiny Shakespeare (about 1 MB of text) into `/data/text/shakespeare.txt`. `dl/classic/char_lstm.py`:

```python
import torch, torch.nn as nn, torch.nn.functional as F

text = open("/data/text/shakespeare.txt").read()
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
V, T, B = len(chars), 256, 64

class CharLSTM(nn.Module):
    def __init__(self, d=512, layers=3):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.lstm = nn.LSTM(d, d, layers, batch_first=True, dropout=0.2)
        self.out = nn.Linear(d, V)
    def forward(self, x, state=None):
        h, state = self.lstm(self.emb(x), state)
        return self.out(h), state

model = CharLSTM().cuda(); opt = torch.optim.AdamW(model.parameters(), 2e-3)
for step in range(5000):
    ix = torch.randint(len(data) - T - 1, (B,))
    x = torch.stack([data[i:i + T] for i in ix]).cuda(); y = torch.stack([data[i + 1:i + T + 1] for i in ix]).cuda()
    logits, _ = model(x)
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))   # predict the next character
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 500 == 0: print(step, loss.item())

@torch.no_grad()
def sample(n=500, temp=0.8):
    x, state, out = torch.tensor([[stoi["\n"]]]).cuda(), None, []
    for _ in range(n):
        logits, state = model(x, state)
        x = torch.multinomial(F.softmax(logits[:, -1] / temp, -1), 1)
        out.append(chars[x.item()])
    return "".join(out)
print(sample())
```

This is next-token prediction, exactly the objective an LLM pretrains on, just with characters instead of subword tokens. Note how the model must pass everything through the `state` vector, one step at a time.

### Build: seq2seq with attention

A toy translation task makes attention visible: turn human dates into ISO dates ("march 3rd, 2021" becomes "2021-03-03"). Generate 50,000 pairs with Python's `datetime` and a dozen random human formats. `dl/classic/seq2seq_attn.py`, the key part:

```python
class AdditiveAttention(nn.Module):
    """Bahdanau attention: score each encoder state against the decoder state."""
    def __init__(self, d):
        super().__init__()
        self.W_q, self.W_k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, 1)
    def forward(self, query, keys):                 # query: (B, d), keys: (B, S, d)
        scores = self.v(torch.tanh(self.W_q(query)[:, None] + self.W_k(keys))).squeeze(-1)   # (B, S)
        weights = F.softmax(scores, dim=-1)        # how much to look at each input position
        context = (weights[:, :, None] * keys).sum(1)   # weighted average of encoder states
        return context, weights
```

An encoder GRU reads the human date; a decoder GRU writes the ISO date one character at a time, and at every step it receives `context` from this attention module. Save `weights` for one example and plot them as a heatmap with matplotlib: you'll see the decoder look at "2021" while writing the year and at "march" while writing "03".

That heatmap is the conceptual jump to the next phase. In a transformer, attention isn't an add-on to a recurrent network: it's the whole mechanism, applied by every position to every other position, in parallel.

### Verify

- ResNet: about 93% CIFAR-10 test accuracy. Remove the skip connections, keeping everything else the same, and see accuracy drop and training become unstable.
- LSTM: loss below about 1.4 nats per character, and samples that look like a play: speaker names, line breaks, archaic words.
- Seq2seq: over 99% exact-match accuracy on held-out dates, and an attention heatmap that lines up as described.

### Commit

`feat(dl): ResNet on CIFAR-10, character LSTM, seq2seq with additive attention`

### Explore (optional)

- **Receptive fields.** Compute how many input pixels one output of your ResNet's last block can see. Compare with a transformer, where every token sees every other token in a single layer.
- **Vanishing gradients.** Replace the LSTM with a plain `nn.RNN` and plot gradient norms with respect to inputs 1, 50 and 200 steps back. The LSTM's gates exist to keep that signal alive.
- **Speed.** Time one training step of the LSTM with sequence length 256 versus 1024. Recurrence can't be parallelized across time. Remember this number for Phase 28.
- **A vision transformer, briefly.** Split CIFAR images into 4x4 patches, embed each patch as a token, and feed them to the transformer you'll build next. ViTs are how transformers took over vision too.

## Phase 28: Attention and a GPT from scratch

### Brief

This is the most important phase. You'll write a modern decoder-only transformer (the architecture of GPT, Llama, Qwen and nearly every LLM) in about 150 lines, train it on children's stories until it writes coherent ones, and generate text with a key-value cache. The same file grows into a 7B model in Phase 33, so it's written with the modern parts from the start.

| Part | What it does | Why the modern version |
| --- | --- | --- |
| Tokenizer (BPE) | Turns text into integer IDs for subword pieces | Characters are too long a sequence; whole words are too many |
| Token embedding | Looks up a learned vector for each ID | Tied with the output layer to save parameters |
| Self-attention | Each position mixes information from earlier positions, weighted by relevance | Parallel over the whole sequence, unlike an RNN |
| Multi-head and grouped-query attention (GQA) | Several attention patterns at once; key and value heads shared across query heads | GQA shrinks the inference cache, standard in Llama-style models |
| RoPE | Encodes position by rotating query and key vectors | Relative positions come out naturally; extends to longer contexts |
| RMSNorm, pre-norm | Rescales each vector before attention and the MLP | Simpler and steadier than LayerNorm after the block |
| SwiGLU MLP | A gated feed-forward layer applied to each position | Trains better than a plain GELU MLP at equal size |
| Residual stream | `x = x + block(x)`: every layer adds a correction | The ResNet idea; makes depth trainable |
| KV cache | Keeps past keys and values during generation | Each new token costs one step, not a full re-run |

### The core idea: scaled dot-product attention

For each position, the model makes three vectors from the current representation: a **query** (what am I looking for?), a **key** (what do I contain?) and a **value** (what do I pass on?). Attention scores are dot products of one position's query with every key, scaled and normalized:

```latex
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^{\top}}{\sqrt{d_{\text{head}}}} + M\right) V
```

The mask M is minus infinity above the diagonal, so a position can only attend to itself and earlier positions. That's what makes it a **causal** language model. Dividing by the square root of the head dimension keeps scores from growing with vector size, which would make softmax saturate. Written out by hand:

```python
att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)                        # (B, H, T, T) scores
mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)                    # rows sum to 1
y = att @ v                                                                    # (B, H, T, head_dim)
```

The T-by-T score matrix is why attention cost grows with the square of sequence length. The model below calls `F.scaled_dot_product_attention`, which computes the same thing with a fused kernel that never stores that matrix. Phase 32 explains how.

### Build: the model

`dl/gpt/model.py`:

```python
import math
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F

@dataclass
class GPTConfig:
    vocab_size: int = 50304       # GPT-2's 50257 padded to a multiple of 64 for faster matmuls
    seq_len: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 12           # fewer than n_head = grouped-query attention
    d_model: int = 768
    ffn_hidden: int = 2048        # about 8/3 * d_model for SwiGLU
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        xf = x.float()                                    # normalize in FP32 for stability
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight

def rope_tables(seq_len, head_dim, theta, device):
    inv = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv)      # (T, head_dim/2)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):                             # x: (B, H, T, head_dim)
    x1, x2 = x[..., ::2].float(), x[..., 1::2].float()   # rotate each (even, odd) pair by an angle
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2).type_as(x)

class KVCache:
    def __init__(self): self.k = self.v = None
    def update(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self.k, self.v

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh, self.nkv, self.hd = c.n_head, c.n_kv_head, c.d_model // c.n_head
        self.wq = nn.Linear(c.d_model, self.nh * self.hd, bias=False)
        self.wk = nn.Linear(c.d_model, self.nkv * self.hd, bias=False)
        self.wv = nn.Linear(c.d_model, self.nkv * self.hd, bias=False)
        self.wo = nn.Linear(self.nh * self.hd, c.d_model, bias=False)
    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(k, v)
        if self.nkv != self.nh:                           # share each K/V head across a group of Q heads
            k = k.repeat_interleave(self.nh // self.nkv, dim=1)
            v = v.repeat_interleave(self.nh // self.nkv, dim=1)
        # causal when processing a prompt; a single new token may see the whole cache
        y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))

class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w1 = nn.Linear(c.d_model, c.ffn_hidden, bias=False)   # gate
        self.w3 = nn.Linear(c.d_model, c.ffn_hidden, bias=False)   # up
        self.w2 = nn.Linear(c.ffn_hidden, c.d_model, bias=False)   # down
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.attn = RMSNorm(c.d_model, c.norm_eps), Attention(c)
        self.n2, self.mlp = RMSNorm(c.d_model, c.norm_eps), SwiGLU(c)
    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.n1(x), cos, sin, cache)    # pre-norm residual
        return x + self.mlp(self.n2(x))

class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.d_model, c.norm_eps)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.head.weight = self.tok.weight
        cos, sin = rope_tables(c.seq_len * 2, c.d_model // c.n_head, c.rope_theta, "cpu")
        self.register_buffer("cos", cos, persistent=False); self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():          # scale residual outputs so the stream doesn't grow with depth
            if name.endswith(("wo.weight", "w2.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * c.n_layer))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, caches=None, pos=0):
        T = idx.shape[1]
        x = self.tok(idx)
        cos, sin = self.cos[pos:pos + T], self.sin[pos:pos + T]
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, caches[i] if caches else None)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8, top_k=50):
        caches = [KVCache() for _ in self.blocks]
        logits = self(idx, caches=caches)                  # prefill: the whole prompt at once
        pos = idx.shape[1]
        for _ in range(max_new):
            l = logits[:, -1] / temperature
            if top_k:
                v, _ = torch.topk(l, top_k); l[l < v[:, -1:]] = -float("inf")
            nxt = torch.multinomial(F.softmax(l, -1), 1)
            idx = torch.cat([idx, nxt], 1)
            logits = self(nxt, caches=caches, pos=pos); pos += 1   # decode: one token at a time
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
```

Count parameters by hand once. Each layer has attention weights of d × d for queries and outputs, d × (kv heads × head dim) for keys and values, and 3 × d × ffn for SwiGLU. Embeddings add vocab × d (counted once when tied). For the 124M default above: 12 × (4 × 768² + 3 × 768 × 2048) + 50304 × 768, about 123.7M. Check it against `num_params()`.

### Build: data and a first training run

BPE tokenization starts from bytes and repeatedly merges the most frequent adjacent pair into a new token, until the vocabulary reaches a target size. For now, use GPT-2's trained BPE through `tiktoken`; you'll train your own in Phase 29.

`dl/gpt/prepare_tinystories.py` tokenizes [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (short stories written for small models) into flat binary files:

```python
import numpy as np, tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")
for split in ("train", "validation"):
    ds = load_dataset("roneneldan/TinyStories", split=split)
    ids = ds.map(lambda b: {"ids": [enc.encode_ordinary(t) + [enc.eot_token] for t in b["text"]]},
                 batched=True, num_proc=32, remove_columns=["text"])
    arr = np.memmap(f"/data/tinystories_{split}.bin", dtype=np.uint16, mode="w+",
                    shape=(sum(len(x) for x in ids["ids"]),))
    i = 0
    for x in ids["ids"]:
        arr[i:i + len(x)] = x; i += len(x)
    arr.flush(); print(split, i, "tokens")
```

Storing token IDs as `uint16` works because the vocabulary is under 65,536. Reading random windows from a memory-mapped file is the simplest fast data loader there is. `dl/gpt/train_small.py`:

```python
import numpy as np, tiktoken, torch
from model import GPT, GPTConfig

cfg = GPTConfig(seq_len=512, n_layer=8, n_head=8, n_kv_head=8, d_model=512, ffn_hidden=1408)
model = GPT(cfg).cuda(); print(f"{model.num_params() / 1e6:.1f}M parameters")
model = torch.compile(model)
train = np.memmap("/data/tinystories_train.bin", dtype=np.uint16, mode="r")
val = np.memmap("/data/tinystories_validation.bin", dtype=np.uint16, mode="r")

def batch(data, B=64, T=cfg.seq_len):
    ix = np.random.randint(0, len(data) - T - 1, B)
    x = torch.from_numpy(np.stack([data[i:i + T] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1:i + T + 1] for i in ix]).astype(np.int64))
    return x.cuda(non_blocking=True), y.cuda(non_blocking=True)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
steps = 6000
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
for step in range(steps):
    x, y = batch(train)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    if step % 250 == 0:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            vl = sum(model(*batch(val))[1].item() for _ in range(20)) / 20
        print(f"step {step}  train {loss.item():.3f}  val {vl:.3f}")
torch.save(model._orig_mod.state_dict(), "/ckpt/tinystories_51m.pt")

enc = tiktoken.get_encoding("gpt2")
prompt = torch.tensor([enc.encode("Once upon a time, a little robot")]).cuda()
print(enc.decode(model._orig_mod.generate(prompt, 200)[0].tolist()))
```

### Verify

- `make dl CMD="python dl/gpt/prepare_tinystories.py"`, then `make dl CMD="python dl/gpt/train_small.py" GPUS=1` (run it from `dl/gpt`, or add that folder to `PYTHONPATH`).
- The model has about 51M parameters. Validation loss falls below about 1.5 within the 6,000 steps, which should take well under an hour.
- The sample is a short, grammatical story with characters that stay consistent for a few sentences.
- Your manual attention snippet matches `F.scaled_dot_product_attention` to within BF16 rounding on random inputs. Test it with `torch.testing.assert_close`.

### Commit

`feat(dl): GPT with GQA, RoPE, RMSNorm, SwiGLU and KV cache; TinyStories training`

### Explore (optional)

- **Look inside attention.** Return the attention weights from one layer and plot them for a story. Find a head that attends to the previous token, and one that tracks a character's name.
- **Remove a part.** Train without RoPE (no position information at all), without the residual connections, and with post-norm instead of pre-norm. Write down what breaks and how.
- **Measure the cache.** Generate 1,000 tokens with and without the KV cache and time both. Then compute the cache size: 2 × layers × kv heads × head dim × sequence length × 2 bytes. Set `n_kv_head=2` and see GQA cut it by four.
- **Sampling.** Compare temperature 0.2, 0.8 and 1.5, top-k against top-p (nucleus) sampling. Implement top-p yourself: keep the smallest set of tokens whose probabilities sum to p.
- **Compare with the LSTM.** Time a training step at sequence length 1024 against Phase 27's LSTM. This is why transformers won.

## Phase 29: Pretraining on your GPUs, up to a 30-day run

### Brief

Now you pretrain a real base model on real web text, the way labs do, at a scale your box finishes in an afternoon. Along the way you build every piece of a production pretraining pipeline: your own tokenizer, sharded token data, data-parallel training on both GPUs, mixed precision, compilation, a learning-rate schedule, resumable checkpoints, throughput and efficiency metrics, and evaluation. Then a mini scaling-law study shows you how loss improves with compute, which is the reasoning behind every frontier model's size.

| Piece | Tool | What to understand |
| --- | --- | --- |
| Corpus | FineWeb-Edu `sample-10BT` (about 10B tokens of educational web text) | Data quality matters as much as model size |
| Tokenizer | Hugging Face `tokenizers`, byte-level BPE, 32,768 tokens | Your vocabulary fixes what the model can see cheaply |
| Data format | 100M-token `uint16` shards, memory-mapped | Simple, fast, resumable by (shard, offset) |
| Parallelism | `torchrun` with DDP | Each GPU holds the full model and a different slice of the batch; gradients are averaged |
| Batch size | Gradient accumulation to about 0.5M tokens per step | Large batches stabilize training; accumulation fakes them with less memory |
| Speed | BF16 autocast, `torch.compile`, fused AdamW | Expect most of your gains from these three |
| Schedule | Linear warmup, then cosine decay to 10% | Warmup prevents early blowups |
| Efficiency metric | Model FLOPs utilization (MFU) | Share of peak hardware throughput doing useful math |
| Evaluation | Validation loss, HellaSwag | Loss is the training signal; a benchmark shows it means something |

### Build: tokenizer and data

`dl/gpt/train_tokenizer.py`:

```python
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
def texts(limit=2_000_000):                         # about 2M documents is plenty for a tokenizer
    for i, row in enumerate(ds):
        if i >= limit: break
        yield row["text"]

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
special = ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]   # reserved for chat later
trainer = trainers.BpeTrainer(vocab_size=32768, special_tokens=special, min_frequency=2,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
tok.train_from_iterator(texts(), trainer=trainer)
tok.save("/data/tok32k/tokenizer.json")
print(tok.encode("Attention is all you need.").tokens)
```

Reserving chat tokens now means your base model's embedding table already has slots for them when you post-train your own model (the SFT section of Phase 23).

`dl/gpt/prepare_fineweb.py` tokenizes in parallel and writes shards:

```python
import os, numpy as np
from multiprocessing import Pool
from datasets import load_dataset
from tokenizers import Tokenizer

SHARD = 100_000_000
tok = Tokenizer.from_file("/data/tok32k/tokenizer.json"); EOT = tok.token_to_id("<|endoftext|>")
def encode(text):
    return np.array([EOT] + tok.encode(text).ids, dtype=np.uint16)   # EOT separates documents

os.makedirs("/data/fineweb_edu_32k", exist_ok=True)
ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", num_proc=32)
buf, n, idx = np.empty(SHARD, dtype=np.uint16), 0, 0
with Pool(96) as pool:
    for ids in pool.imap(encode, (r["text"] for r in ds), chunksize=64):
        while len(ids):
            take = min(SHARD - n, len(ids)); buf[n:n + take] = ids[:take]; n += take; ids = ids[take:]
            if n == SHARD:
                split = "val" if idx == 0 else "train"      # the first shard is held out
                buf.tofile(f"/data/fineweb_edu_32k/{split}_{idx:04d}.bin"); idx += 1; n = 0
print("shards:", idx)
```

This uses your many CPU cores: roughly 10B tokens become about 20 GB of shards. Record the tokenizer file's hash and the shard list in `docs/dl-notes.md`; that's your dataset version. Optionally upload the folder to `s3://datasets/fineweb_edu_32k/` with `make s3` so it lives in the lakehouse alongside everything else.

### Build: the distributed trainer

`dl/gpt/train.py`:

```python
import glob, math, os, sys, time, yaml
import numpy as np, torch, torch.distributed as dist, mlflow
from torch.nn.parallel import DistributedDataParallel as DDP
from model import GPT, GPTConfig

cfg = yaml.safe_load(open(sys.argv[sys.argv.index("--config") + 1]))
ddp = "RANK" in os.environ
if ddp:
    dist.init_process_group("nccl")
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
else:
    rank, world, local = 0, 1, 0
torch.cuda.set_device(local); master = rank == 0
torch.manual_seed(1337 + rank)
torch.backends.cuda.matmul.allow_tf32 = True

class Shards:
    """Each rank reads its own interleaved slice; state is (file, position) so runs can resume."""
    def __init__(self, pattern, B, T):
        self.files, self.B, self.T = sorted(glob.glob(pattern)), B, T
        self.reset(0, B * T * rank)
    def reset(self, f, pos):
        self.f, self.pos = f, pos; self.tok = np.memmap(self.files[f], dtype=np.uint16, mode="r")
    def next(self):
        n = self.B * self.T
        buf = torch.from_numpy(self.tok[self.pos:self.pos + n + 1].astype(np.int64))
        self.pos += n * world
        if self.pos + n * world + 1 > len(self.tok):
            self.reset((self.f + 1) % len(self.files), n * rank)
        return buf[:-1].view(self.B, self.T).cuda(non_blocking=True), buf[1:].view(self.B, self.T).cuda(non_blocking=True)

mc = GPTConfig(**cfg["model"])
raw = GPT(mc).cuda()
B, T = cfg["micro_batch"], mc.seq_len
accum = cfg["batch_tokens"] // (B * T * world)
opt = torch.optim.AdamW(raw.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=cfg["wd"], fused=True)
train, val = Shards(cfg["train_glob"], B, T), Shards(cfg["val_glob"], B, T)

step, ckpt = 0, f"/ckpt/{cfg['name']}/last.pt"
if os.path.exists(ckpt):                                  # resume exactly where we stopped
    s = torch.load(ckpt, map_location="cuda", weights_only=False)
    raw.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step = s["step"]
    train.reset(s["data"][rank]["f"], s["data"][rank]["pos"])
model = torch.compile(raw) if cfg.get("compile", True) else raw   # MoE runs set compile: false
if ddp:
    model = DDP(model, device_ids=[local])

def lr_at(s):
    w, total = cfg["warmup"], cfg["steps"]
    if s < w: return cfg["lr"] * (s + 1) / w
    p = (s - w) / max(1, total - w)
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

N = raw.num_params()
flops_per_token = 6 * N + 12 * mc.n_layer * mc.d_model * T     # weights plus attention
if master:
    mlflow.set_experiment("pretrain"); mlflow.start_run(run_name=cfg["name"]); mlflow.log_params({**cfg["model"], **{k: v for k, v in cfg.items() if k != "model"}})

while step < cfg["steps"]:
    t0 = time.perf_counter()
    for g in opt.param_groups: g["lr"] = lr_at(step)
    for micro in range(accum):
        x, y = train.next()
        if ddp: model.require_backward_grad_sync = micro == accum - 1   # all-reduce only once per step
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        (loss / accum).backward()
    gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    step += 1
    if master and step % 10 == 0:
        tps = cfg["batch_tokens"] / dt
        mfu = flops_per_token * tps / (cfg["peak_tflops"] * 1e12 * world)
        mlflow.log_metrics({"loss": loss.item(), "grad_norm": gn.item(), "lr": lr_at(step),
                            "tokens_per_s": tps, "mfu": mfu}, step)
        print(f"step {step}  loss {loss.item():.3f}  {tps / 1e3:.0f}k tok/s  MFU {mfu:.1%}")
    if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            vl = torch.stack([model(*val.next())[1] for _ in range(20)]).mean()
        if ddp: dist.all_reduce(vl, op=dist.ReduceOp.AVG)
        model.train()
        states = [None] * world
        if ddp: dist.all_gather_object(states, {"f": train.f, "pos": train.pos})
        else: states = [{"f": train.f, "pos": train.pos}]
        if master:
            mlflow.log_metric("val_loss", vl.item(), step)
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save({"model": raw.state_dict(), "opt": opt.state_dict(), "step": step,
                        "data": states, "cfg": cfg}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)                # atomic: a crash never leaves half a file
if ddp: dist.destroy_process_group()
```

`dl/gpt/configs/124m.yaml`:

```yaml
name: gpt124m-fineweb
model: { vocab_size: 32768, seq_len: 1024, n_layer: 12, n_head: 12, n_kv_head: 12, d_model: 768, ffn_hidden: 2048 }
train_glob: /data/fineweb_edu_32k/train_*.bin
val_glob: /data/fineweb_edu_32k/val_*.bin
micro_batch: 32
batch_tokens: 524288          # 2^19 tokens per optimizer step
steps: 4300                   # about 2.25B tokens, roughly 20 tokens per parameter
warmup: 200
lr: 6.0e-4
wd: 0.1
eval_every: 250
peak_tflops: 147              # per GPU, dense BF16; replace with your Phase 22 measurement
```

The vocabulary of 32,768 is already a multiple of 64. With this vocabulary the model is about 110M parameters, because the embedding table is smaller than GPT-2's.

### Build: an evaluation that means something

Validation loss tells you the model is learning; it doesn't tell you what it can do. HellaSwag is a classic first benchmark: each item has a context and four endings, and the model should prefer the correct one. Score each ending by its average per-token loss given the context and pick the lowest. `dl/gpt/eval_hellaswag.py` loads the validation split from Hugging Face (`Rowan/hellaswag`), applies that rule, and reports accuracy. Random guessing scores 25%; GPT-2 124M scored about 29–31%. A good FineWeb-Edu 124M model should land near that range.

### Verify

```bash
make dl CMD="python dl/gpt/train_tokenizer.py"
make dl CMD="python dl/gpt/prepare_fineweb.py"
make dl-dist CMD="dl/gpt/train.py --config dl/gpt/configs/124m.yaml"
```

- The startup log shows `accum` of 8 (32 sequences × 1024 tokens × 2 GPUs × 8 = 524,288).
- Loss drops fast in the first few hundred steps, then slowly. MFU settles somewhere between 30% and 50%. If it's under 20%, profile before training longer.
- **Kill the run and restart it.** It resumes at the last checkpoint's step, with the loss curve continuing smoothly in MLflow.
- After a few hours, validation loss is near 3.0 nats per token and HellaSwag is well above 25%. Load the checkpoint and generate text: fluent, factual-sounding, often wrong. That's a base model.

### Build: a mini scaling-law study

Train four sizes, each on about 20 tokens per parameter, keeping everything else fixed. Make a config per row:

| Name | Layers | d\_model | Heads | ffn\_hidden | Parameters | Tokens | Time at 120 TFLOPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| s20m | 6 | 384 | 6 | 1024 | About 23M | 460M | About 10 minutes |
| s45m | 8 | 512 | 8 | 1408 | About 43M | 850M | About 30 minutes |
| s110m | 12 | 768 | 12 | 2048 | About 110M | 2.2B | About 3.5 hours |
| s340m | 24 | 1024 | 16 | 2816 | About 340M | 6.8B | About 1.3 days (optional) |

Plot final validation loss against total training FLOPs on log-log axes. The points fall close to a straight line: loss falls as a power of compute. Fit that line with three points, predict the fourth, then check. This is how labs decide model size before spending real money.

### Build: a 30-day run you can pause and resume

The capstone of this phase: a 1.1B-parameter base model trained for about 30 days on about 40–45B tokens (more with three cards), roughly double the compute-optimal amount. It's the largest from-scratch model your box trains in a month, in the same range as nanochat's 1.9B cloud run. The engineering matters as much as the model: over 30 days, power blips, kernel updates, and your own need for a GPU will interrupt it. The run must survive all of them, losing at most a day.

| Requirement | How |
| --- | --- |
| Lose at most a day | A full checkpoint every 24 hours (configurable), plus one on any requested stop |
| Keep only the latest | Write to `last.pt.tmp`, flush to disk, then atomically rename over `last.pt`. The old one disappears only once the new one is complete |
| Pause on demand | A stop signal sets a flag; every rank agrees, saves, and exits cleanly within one step |
| Resume exactly | Model, optimizer, step and each rank's data position are restored; the learning-rate schedule is a function of the step |
| Restart after crashes | The service restarts automatically and resumes from `last.pt` |

**Data.** 45B tokens is more than the 10B sample from earlier in this phase. Re-run `prepare_fineweb.py` with `name="sample-100BT"`, writing to `/data/fineweb_edu_32k_100b/`. At two bytes per token, 45B tokens is about 90 GB of shards.

**The config,** `dl/gpt/configs/1b_30day.yaml`:

```yaml
name: gpt1b-30day
model: { vocab_size: 32768, seq_len: 2048, n_layer: 24, n_head: 16, n_kv_head: 4, d_model: 2048, ffn_hidden: 5632 }
train_glob: /data/fineweb_edu_32k_100b/train_*.bin
val_glob: /data/fineweb_edu_32k_100b/val_*.bin
micro_batch: 8
batch_tokens: 1048576          # 2^20 tokens per step: accumulation of 32 on two GPUs
steps: 41000                   # about 43B tokens; raise to about 62000 with three GPUs
warmup: 2000
lr: 3.0e-4
wd: 0.1
eval_every: 1000
ckpt_hours: 24
peak_tflops: 147
```

Parameter count, as in Phase 28: 24 layers × (2 × 2048² for queries and outputs + 2 × 2048 × 512 for keys and values + 3 × 2048 × 5632 for SwiGLU) plus 32768 × 2048 for the tied embedding, about 1.15B. Check the expected rate against reality in the first hour: at 120 TFLOPS, about 16,000 tokens per second, or about 1.4B tokens per day.

**Changes to `train.py`.** Replace the periodic checkpoint with a time-based one, and add a stop flag that every rank agrees on, so no rank waits forever on a collective call another rank skipped:

```python
import signal

stop_requested = False
def _request_stop(signum, frame):
    global stop_requested
    stop_requested = True                       # finish the current step, then save and exit
signal.signal(signal.SIGTERM, _request_stop)    # docker compose stop sends SIGTERM
signal.signal(signal.SIGUSR1, _request_stop)
last_ckpt = time.time()

def save_checkpoint():
    states = [None] * world
    if ddp: dist.all_gather_object(states, {"f": train.f, "pos": train.pos})
    else: states = [{"f": train.f, "pos": train.pos}]
    if master:
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        tmp = ckpt + ".tmp"
        with open(tmp, "wb") as fh:
            torch.save({"model": raw.state_dict(), "opt": opt.state_dict(), "step": step,
                        "data": states, "cfg": cfg}, fh)
            fh.flush(); os.fsync(fh.fileno())   # really on disk before we replace the old one
        os.replace(tmp, ckpt)                   # atomic: only ever one complete checkpoint
        json.dump({"step": step, "tokens": step * cfg["batch_tokens"], "saved_at": time.ctime()},
                  open(ckpt.replace(".pt", ".json"), "w"))
    if ddp: dist.barrier()

# at the end of every optimizer step, after logging:
    flags = torch.tensor([float(stop_requested),
                          float(time.time() - last_ckpt > cfg["ckpt_hours"] * 3600)], device="cuda")
    if ddp: dist.all_reduce(flags, op=dist.ReduceOp.MAX)   # if any rank wants it, all ranks do it
    if flags[0] or flags[1] or step == cfg["steps"]:
        save_checkpoint(); last_ckpt = time.time()
    if flags[0]:
        break                                    # clean exit; the next start resumes from last.pt
```

Add `import json` at the top. The resume block from earlier in this phase already restores everything. A 1.15B checkpoint with AdamW state is about 18 GB, which the root NVMe writes in well under a minute, so a daily save costs almost nothing. Shorten `ckpt_hours` to 6 if you'd rather lose less.

**Run it as a service.** Add to `compose.dl.yaml`:

```yaml
  pretrain-long:
    <<: *dl
    command: torchrun --standalone --nproc_per_node=2 dl/gpt/train.py --config dl/gpt/configs/1b_30day.yaml
    restart: unless-stopped        # crashes and reboots resume automatically
    stop_grace_period: 15m         # time to finish the step and write the checkpoint
    environment:
      CUDA_VISIBLE_DEVICES: "0,1"  # with a third card installed, keep GPU 2 free for daily work
      HF_HOME: /data/hf
      MLFLOW_TRACKING_URI: http://mlflow:5000
```

```makefile
pretrain-start:
	$(DC) --profile tools up -d pretrain-long
pretrain-pause:                     # saves a checkpoint, then stops
	$(DC) stop -t 900 pretrain-long
pretrain-resume:
	$(DC) --profile tools up -d pretrain-long
pretrain-status:
	cat $(DL_CKPT)/gpt1b-30day/last.json; $(DC) logs --tail 5 pretrain-long
```

Give the service `profiles: [tools]` like `dl`, so a plain `make up` doesn't start it by accident. For the MLflow run, make the trainer reuse the same run when it resumes (store the run ID in `last.json` and pass it to `mlflow.start_run(run_id=...)`), so a month of restarts shows as one continuous curve.

**Watch it like a production job.** Your Grafana stack already tracks this box. Add panels for tokens per second and validation loss (from MLflow), GPU temperature and power, and the DIMM ECC and temperature panels you built for this server. A month of full load is exactly when marginal memory or cooling problems show up. Alert if tokens per second drops by more than 20%, or if `last.json` is more than 26 hours old.

**Verify it before committing 30 days.**

1. Start the run, wait for one checkpoint (set `ckpt_hours: 0.1` for this test), then `make pretrain-pause`. The log shows a clean save and exit.
2. `make pretrain-resume`: the step count continues and the loss curve has no jump.
3. `docker kill` the container mid-step (an unclean crash). It restarts on its own and resumes from the last checkpoint, losing only the steps since.
4. Reboot the machine. The run comes back after boot.

Then set `ckpt_hours` back to 24 and let it run. Every few days, load `last.pt` into a notebook and sample text, and run HellaSwag on it. Watching a model go from babble to fluent prose over a month teaches more than any curve.

### Commit

`feat(dl): BPE tokenizer, FineWeb-Edu shards, resumable DDP pretraining, HellaSwag eval, scaling study`, 30-day run with daily checkpoints and safe pause

### Explore (optional)

- **Data parallel scaling.** Run the 124M config on one GPU (adjust `batch_tokens` or let accumulation double) and on two. Is it 2× faster? The gap is gradient all-reduce over PCIe; measure it with the profiler in Phase 32.
- **Learning-rate sensitivity.** Run s45m at 3e-4, 6e-4, 1.2e-3 and 2.4e-3. Small models tolerate higher rates; find the edge.
- **A warmup-stable-decay schedule.** Hold the learning rate flat and decay only in the last 10–20% of steps. You can then branch off decayed checkpoints at several lengths from one run.
- **The nanochat capstone.** Clone [nanochat](https://github.com/ido/nanochat) and run its pipeline with `--nproc_per_node=2` and a smaller device batch. It covers tokenizer, pretraining, mid-training, SFT and RL end to end. Expect several days at speedrun scale on your box; compare its code choices with yours.
- **Deduplicate.** Hash every 13-token window across a few shards and count repeats. Web data is surprisingly repetitive, and duplicates waste compute.

## Phase 30: Mixture of experts

### Brief

A mixture-of-experts (MoE) layer replaces the single feed-forward MLP in each block with many smaller "expert" MLPs and a **router** that sends each token to only a few of them. Total parameters grow a lot; compute per token stays the same. That's how models like Mixtral, DeepSeek-V3 and Qwen3's MoE variants get more capacity without proportionally more compute. You'll build one, train it against a dense model of equal per-token compute, and look at what the experts learn.

| Part | What it does | What goes wrong without care |
| --- | --- | --- |
| Router | A linear layer scoring each expert for each token | Learns to favor a few experts, leaving the rest idle (collapse) |
| Top-k selection | Each token uses only its k highest-scoring experts, usually 1 or 2 | k too small hurts quality; too large loses the savings |
| Gate weights | Router probabilities of the chosen experts, renormalized | Without them, the router gets no gradient signal |
| Load-balancing loss | Rewards spreading tokens evenly over experts | Collapse, then wasted parameters |
| Capacity factor | Maximum tokens per expert per batch, in fixed-shape implementations | Overflowing tokens are dropped |
| Shared expert | One expert every token always uses (DeepSeek's design) | Common knowledge gets duplicated across experts |
| Expert parallelism | Experts placed on different GPUs; tokens move with all-to-all communication | Communication becomes the bottleneck |

Terminology to keep straight: **total parameters** count every expert; **active parameters** count only what one token touches. A "30B-A3B" model has 30 billion total and 3 billion active.

### Build: the MoE layer

Add to `GPTConfig` in `dl/gpt/model.py`:

```python
    n_experts: int = 0            # 0 = dense model
    top_k: int = 2
    expert_hidden: int = 0        # hidden size of each expert
    aux_coef: float = 0.01        # weight of the load-balancing loss
```

Let `SwiGLU` take an optional hidden size (`def __init__(self, c, hidden=None)` and `h = hidden or c.ffn_hidden`). Then add:

```python
class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.E, self.k = c.n_experts, c.top_k
        self.router = nn.Linear(c.d_model, self.E, bias=False)
        self.experts = nn.ModuleList(SwiGLU(c, hidden=c.expert_hidden) for _ in range(self.E))
        self.aux_loss = torch.zeros(())
        self.last_load = None

    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)                                     # (N, D) with N = B*T tokens
        probs = self.router(xf).float().softmax(dim=-1)           # (N, E)
        w, idx = probs.topk(self.k, dim=-1)                       # chosen experts per token
        w = (w / w.sum(-1, keepdim=True)).to(x.dtype)             # renormalize over the chosen k
        out = torch.zeros_like(xf)
        for e, expert in enumerate(self.experts):                 # simple and readable; Phase 32 speeds it up
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, expert(xf[tok]) * w[tok, slot, None])
        # Switch-style balancing: E * sum(fraction of routed tokens * mean router probability)
        frac = torch.bincount(idx.flatten(), minlength=self.E).float() / idx.numel()
        self.aux_loss = self.E * (frac * probs.mean(0)).sum()
        self.last_load = frac.detach()
        return out.view(B, T, D)
```

In `Block.__init__`, use `self.mlp = MoE(c) if c.n_experts else SwiGLU(c)`. In `GPT.forward`, add the auxiliary loss when computing the training loss:

```python
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        if self.c.n_experts:
            loss = loss + self.c.aux_coef * sum(b.mlp.aux_loss for b in self.blocks) / len(self.blocks)
        return logits, loss
```

The gradient path is worth tracing once. `topk` itself isn't differentiable, but the output is multiplied by `w`, which comes from the router's softmax. So the router learns from how much each chosen expert's output helped. The balancing loss then pushes probability toward under-used experts.

The per-expert loop and the data-dependent `nonzero` cause `torch.compile` graph breaks. That's fine for learning: train this model without compile, or compile only the attention blocks.

### Build: a fair comparison

Match compute per token, not total parameters. The dense s45m model from Phase 29 has `ffn_hidden: 1408`. An MoE with 8 experts of hidden size 704 and top-2 routing does the same MLP math per token, with four times the MLP parameters:

```yaml
# dl/gpt/configs/s45m_moe8.yaml (copy s45m.yaml and change the model line)
name: s45m-moe8-top2
model: { vocab_size: 32768, seq_len: 1024, n_layer: 8, n_head: 8, n_kv_head: 8, d_model: 512,
         ffn_hidden: 1408, n_experts: 8, top_k: 2, expert_hidden: 704, aux_coef: 0.01 }
compile: false
```

In `train.py`, compute `flops_per_token` from **active** parameters when `n_experts` is set, so MFU stays honest. Then log expert load per layer every 100 steps from rank 0:

```python
if master and step % 100 == 0 and mc.n_experts:
    for i, b in enumerate(raw.blocks):
        mlflow.log_metric(f"moe_max_load_l{i}", b.mlp.last_load.max().item(), step)   # 1/E is perfect balance
```

### Verify

- Train dense s45m and s45m-moe8 on the same token count. At equal tokens, the MoE's validation loss should be lower. At equal **wall-clock time** the gap shrinks or reverses, because the routing loop is slow. Both facts matter.
- `moe_max_load` per layer stays near 0.125 to 0.25 (one-eighth is perfect). Set `aux_coef: 0` and rerun briefly: the maximum load climbs toward 0.5 or more as a few experts take over.
- Count parameters: total about 4× the dense model's MLP parameters plus the shared attention, active about the same as dense.

### Commit

`feat(dl): mixture-of-experts layer with top-k routing and load balancing; dense vs MoE comparison`

### Explore (optional)

- **What do experts specialize in?** Run the trained MoE over a validation shard and record which expert each token goes to in layer 4. Decode tokens per expert. You'll often find experts for punctuation, numbers, code-like text or particular word pieces, rather than clean "topics".
- **Shared expert.** Add one always-on expert alongside the routed ones, shrinking the others to keep compute equal. Compare loss and load balance.
- **Top-1 versus top-2.** Switch Transformer used top-1. Compare quality and router stability.
- **Fixed capacity.** Rewrite routing with a capacity factor of 1.25: each expert takes at most 1.25 × N × k / E tokens, the rest pass through the residual only. Count dropped tokens. Fixed shapes are what make fast kernels possible.
- **Expert parallelism on two GPUs.** Put experts 0–3 on GPU 0 and 4–7 on GPU 1, and exchange tokens with `torch.distributed.all_to_all_single`. Measure how much time goes to communication versus expert math. This is the central engineering problem of large MoE training.

## Phase 31: GPU programming I — CUDA C++

### Brief

Until now PyTorch has hidden the GPU from you. This phase opens it up: how thousands of threads are organized, where data lives, why most operations are limited by memory rather than math, and how to write, time and profile your own kernels. Then you load one into PyTorch. You won't hand-write production matrix multiplies (NVIDIA's libraries do that better), but understanding them is what lets you write the custom kernels that matter: fused operations and new architectures.

| Concept | What it means | Number on your GPU |
| --- | --- | --- |
| Streaming multiprocessor (SM) | An independent processor running many threads | Read it from `torch.cuda.get_device_properties(0).multi_processor_count` |
| Warp | 32 threads executing in lockstep; the real unit of execution | Always 32 |
| Thread block | A group of threads (up to 1,024) that share fast memory and can synchronize | You choose, often 128–512 |
| Grid | All the blocks of one kernel launch | You choose, sized to the data |
| Registers | Private per-thread storage, the fastest | Limited per SM; heavy use lowers occupancy |
| Shared memory | On-chip memory shared within a block, managed by you | Tens of KB per block; about 100× faster than global memory |
| L2 cache | Shared across all SMs | 96 MB on this card |
| Global memory (GDDR7) | The 72 GB everyone sees | About 1.34 TB/s |
| Tensor cores | Units that multiply small matrix tiles per instruction | Where the 147 BF16 TFLOPS come from |

The single most useful mental model is the **roofline**. Divide peak math rate by memory bandwidth: 147 × 10^12 / 1.34 × 10^12 is about 110 FLOPs per byte. An operation doing fewer FLOPs per byte than that is **memory-bound**: the cores wait for data. Adding two tensors does 1 FLOP per 6 bytes moved, so it's hopelessly memory-bound. A large matmul does hundreds of FLOPs per byte, so it's compute-bound. Most of a transformer's non-matmul work (norms, activations, residual adds, softmax) is memory-bound, which is why **fusing** several of those operations into one kernel, reading memory once instead of several times, is the main lever in Phase 32.

### Build: a first kernel

`dl/kernels/cuda/vadd.cu`:

```cpp
#include <cstdio>
#include <cuda_runtime.h>
#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 1; } } while (0)

__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's global index
    if (i < n) c[i] = a[i] + b[i];                    // guard: the last block may overhang
}

int main() {
    const int n = 1 << 28;                            // 268M floats, 1 GB per array
    size_t bytes = n * sizeof(float);
    float *a, *b, *c;
    CHECK(cudaMalloc(&a, bytes)); CHECK(cudaMalloc(&b, bytes)); CHECK(cudaMalloc(&c, bytes));
    cudaMemset(a, 0, bytes); cudaMemset(b, 0, bytes);
    int threads = 256, blocks = (n + threads - 1) / threads;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    vadd<<<blocks, threads>>>(a, b, c, n);           // warm-up
    cudaEventRecord(t0);
    for (int r = 0; r < 20; r++) vadd<<<blocks, threads>>>(a, b, c, n);
    cudaEventRecord(t1); CHECK(cudaEventSynchronize(t1)); CHECK(cudaGetLastError());
    float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= 20;
    printf("%.3f ms, %.0f GB/s effective bandwidth\n", ms, 3.0 * bytes / ms / 1e6);   // 2 reads + 1 write
    return 0;
}
```

```bash
make dl CMD="bash -c 'cd dl/kernels/cuda && nvcc -O3 -arch=sm_120 vadd.cu -o vadd && ./vadd'" GPUS=1
```

`-arch=sm_120` targets compute capability 12.0, your Blackwell workstation cards. The result should be a large fraction of 1,340 GB/s. That's your practical bandwidth ceiling; write it down.

### Build: coalescing

Threads in a warp should read neighboring addresses, so the hardware can combine 32 reads into a few wide transactions. Add a kernel that reads with a stride, `c[i] = a[(i * stride) % n]`, and time strides of 1, 2, 8, 32 and 128. Bandwidth collapses as the stride grows, even though the amount of data is the same. This is why tensor layout (row-major versus column-major, contiguous or not) matters so much.

### Build: a reduction with shared memory and warp shuffles

Summing a large array teaches the two cooperation tools inside a block. `dl/kernels/cuda/reduce.cu`, the kernel:

```cpp
__inline__ __device__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);   // exchange registers inside the warp, no memory
    return v;
}

__global__ void reduce_sum(const float* x, float* out, int n) {
    __shared__ float partial[32];                      // one slot per warp in the block
    float v = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        v += x[i];                                     // grid-stride loop: any n, any grid size
    v = warp_sum(v);
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) partial[warp] = v;
    __syncthreads();                                   // wait for every warp's result
    if (warp == 0) {
        v = lane < blockDim.x / 32 ? partial[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) atomicAdd(out, v);              // combine blocks
    }
}
```

Check the result against a CPU sum (expect small floating-point differences, and explain why the order of additions changes them). Softmax, LayerNorm and RMSNorm all contain exactly this pattern: a reduction across a row.

### Build: matrix multiply, naive then tiled

The naive kernel has each thread compute one output element by reading a full row of A and column of B from global memory, so each input is fetched many times. The tiled version loads a tile of A and a tile of B into shared memory once, and every thread in the block reuses them. `dl/kernels/cuda/matmul.cu`:

```cpp
#define TILE 32
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y, col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < N; t += TILE) {
        As[threadIdx.y][threadIdx.x] = (row < N && t + threadIdx.x < N) ? A[row * N + t + threadIdx.x] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (col < N && t + threadIdx.y < N) ? B[(t + threadIdx.y) * N + col] : 0.f;
        __syncthreads();                               // tile fully loaded
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                               // done reading before the next load
    }
    if (row < N && col < N) C[row * N + col] = acc;
}
// launch: dim3 threads(TILE, TILE); dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
```

Time naive, tiled and `torch.matmul` in FP32 at N = 4096, and compute TFLOPS for each as 2N³ / time. The tiled kernel beats naive by several times; cuBLAS (behind `torch.matmul`) beats tiled by several more, using register tiling, vectorized loads, pipelined copies and, in BF16, tensor cores. Read CUTLASS's documentation on its tile hierarchy to see the gap you didn't close; that's the honest end of hand-written matmuls.

### Build: profile it

```bash
make dl CMD="bash -c 'cd dl/kernels/cuda && ncu --set full -o matmul_report ./matmul'" GPUS=1
make dl CMD="bash -c 'cd dl/kernels/cuda && nsys profile -o vadd_timeline ./vadd'" GPUS=1
```

Open the reports in the Nsight Compute and Nsight Systems desktop apps (free, on your Mac). For the tiled matmul, find: achieved occupancy, memory throughput as a share of peak, shared-memory bank conflicts, and where it sits on the built-in roofline chart.

### Build: call your kernel from PyTorch

`dl/kernels/cuda/torch_ext.py` compiles CUDA source at import time and exposes it as a Python function:

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
__global__ void silu_mul_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = a[i]; out[i] = x / (1.f + __expf(-x)) * b[i]; }   // silu(a) * b, one pass
}
torch::Tensor silu_mul(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dtype() == torch::kFloat32);
    auto out = torch::empty_like(a); int n = a.numel();
    silu_mul_kernel<<<(n + 255) / 256, 256>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""
ext = load_inline("silu_mul_ext", cpp_sources="torch::Tensor silu_mul(torch::Tensor a, torch::Tensor b);",
                  cuda_sources=cuda_src, functions=["silu_mul"], extra_cuda_cflags=["-O3"])

a, b = torch.randn(64, 4096, 2048, device="cuda"), torch.randn(64, 4096, 2048, device="cuda")
torch.testing.assert_close(ext.silu_mul(a, b), torch.nn.functional.silu(a) * b)
print("matches PyTorch")
```

That's exactly the elementwise core of the SwiGLU MLP in your GPT. Time it against the unfused PyTorch expression: PyTorch writes `silu(a)` to memory and reads it back; your kernel doesn't.

### Verify

- `vadd` reports bandwidth above about 80% of 1,340 GB/s.
- The coalescing experiment shows bandwidth falling as stride grows.
- The reduction matches a CPU sum within a small tolerance.
- Tiled matmul is clearly faster than naive and clearly slower than `torch.matmul`; you can say why in one sentence each.
- The PyTorch extension matches `silu(a) * b` and runs faster than the unfused version.

### Commit

`feat(kernels): CUDA vector add, coalescing, reduction, tiled matmul, PyTorch extension, profiles`

### Explore (optional)

- **Occupancy.** Rerun `vadd` with 32, 64, 128, 256, 512 and 1,024 threads per block. Then add a large per-thread local array to burn registers and watch occupancy fall in `ncu`.
- **Vectorized loads.** Rewrite `vadd` to read `float4` (four floats per instruction). See how close to peak bandwidth you can get.
- **Bank conflicts.** In the tiled matmul, change the shared array to `[TILE][TILE + 1]` and see the conflicts disappear in `ncu`. Explain why padding by one column works.
- **An online softmax.** Write a row-wise softmax kernel that computes the max and the sum in a single pass, rescaling the running sum when the max changes. This trick is the heart of flash attention.
- **Backward pass.** Write the gradient kernel for `silu_mul` and wrap both in a `torch.autograd.Function`. Check it with `torch.autograd.gradcheck` in FP64.

## Phase 32: GPU programming II — Triton, custom ops and fused kernels

### Brief

Triton lets you write GPU kernels in Python at the level of **blocks** of data instead of individual threads. The compiler handles thread layout, shared memory and much of the tuning. It's how `torch.compile` generates kernels, how most new model architectures get fast custom kernels, and the practical tool for you. This phase: read what the compiler already writes for your GPT, write a fused RMSNorm with its backward pass, plug it into the model, and understand flash attention, the most important fused kernel in modern AI.

| Level | Tool | You write | Use it when |
| --- | --- | --- | --- |
| Library calls | PyTorch ops, cuBLAS, cuDNN | Nothing | Always the starting point |
| Automatic fusion | `torch.compile` (generates Triton) | Nothing; you read its output | Most elementwise and reduction chains |
| Block-level kernels | Triton | Python functions over tiles | Fusions the compiler misses; new architectures |
| Thread-level kernels | CUDA C++, CUTLASS, CuTe | Threads, shared memory, tensor core instructions | Last few percent; things Triton can't express |

### Build: read the compiler's kernels first

```bash
make dl CMD="bash -c 'cd dl/gpt && TORCH_LOGS=output_code python train_small.py 2>&1 | head -400 > /data/compile_output.txt'" GPUS=1
```

Open the file and find a generated `triton_` kernel fusing RMSNorm's operations, or SwiGLU's multiply. You'll see the same `tl.load`, `tl.sum`, `tl.store` you're about to write. Then profile one training step to see which kernels take the time:

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    for _ in range(5):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
prof.export_chrome_trace("/data/gpt_step_trace.json")      # open in chrome://tracing or Perfetto
```

Typically matmuls dominate, then attention, then everything else. In the two-GPU run, look for NCCL all-reduce kernels and whether they overlap with backward computation.

### Build: a fused RMSNorm in Triton

RMSNorm written in PyTorch reads the input several times (square, mean, multiply, multiply by weight). The fused version reads each row once, keeps it in registers, and writes once. `dl/kernels/triton/rmsnorm.py`:

```python
import torch, triton, triton.language as tl

@triton.jit
def _rmsnorm_fwd(X, W, Y, RSTD, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)                                 # one program instance per row
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)   # the row reduction from Phase 31, in one line
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)
    tl.store(RSTD + row, rstd)                             # saved for the backward pass

class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        M, N = x2.shape
        y = torch.empty_like(x2)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)
        _rmsnorm_fwd[(M,)](x2, w, y, rstd, x2.stride(0), N, eps, BLOCK=triton.next_power_of_2(N))
        ctx.save_for_backward(x2, w, rstd); ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, g):
        # y = w * x * r, with r = (mean(x^2) + eps)^-1/2
        # dx = r * (g*w - x * r^2 * mean(g*w*x));   dw = sum over rows of g * x * r
        x, w, rstd = ctx.saved_tensors
        g = g.reshape(x.shape).float(); xf = x.float(); r = rstd[:, None]
        gw = g * w.float()
        dx = r * (gw - xf * r * r * (gw * xf).mean(-1, keepdim=True))
        dw = (g * xf * r).sum(0)
        return dx.to(x.dtype).view(ctx.shape), dw.to(w.dtype), None

def rmsnorm(x, w, eps=1e-5):
    return RMSNormFn.apply(x, w, eps)
```

The backward here is written in PyTorch for clarity; it's correct, just not fused. Fusing it in Triton is the first exercise. Derive the `dx` formula yourself on paper once: it's the chain rule through the normalization, and it's the same derivation for LayerNorm with a mean term added.

`dl/kernels/triton/test_rmsnorm.py` checks correctness, including gradients, before you trust it:

```python
import torch, triton
from rmsnorm import rmsnorm

def ref(x, w, eps=1e-5):
    xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w

for dtype in (torch.float32, torch.bfloat16):
    x = torch.randn(8, 1024, 768, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(768, device="cuda", dtype=dtype, requires_grad=True)
    y1 = rmsnorm(x, w); y2 = ref(x, w)
    torch.testing.assert_close(y1, y2, atol=1e-2 if dtype == torch.bfloat16 else 1e-5, rtol=1e-2)
    g = torch.randn_like(y1)
    gx1, gw1 = torch.autograd.grad(y1, (x, w), g); gx2, gw2 = torch.autograd.grad(y2, (x, w), g)
    torch.testing.assert_close(gx1, gx2, atol=5e-2, rtol=5e-2); torch.testing.assert_close(gw1, gw2, atol=5e-1, rtol=5e-2)

x = torch.randn(32 * 1024, 4096, device="cuda", dtype=torch.bfloat16); w = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
for name, fn in (("triton", lambda: rmsnorm(x, w)), ("eager", lambda: ref(x, w)), ("compiled", torch.compile(lambda: ref(x, w)))):
    ms = triton.testing.do_bench(fn)
    print(f"{name:9} {ms:.3f} ms  {2 * x.numel() * 2 / ms / 1e6:.0f} GB/s")   # read x, write y, BF16
```

Expect Triton and the compiled version to be close to each other and both well ahead of eager PyTorch, all near your measured bandwidth. That's the lesson: for memory-bound operations, the win is reading memory once.

In `model.py`, make `RMSNorm.forward` call `rmsnorm(x, self.weight, self.eps)` behind a config flag, and compare tokens per second on the 124M run with `torch.compile` off and on.

### Understand: flash attention

Standard attention writes the T × T score matrix to memory, applies softmax, and reads it back to multiply by V. At T = 8,192 with 32 heads, that matrix is about 4 GB per layer per sequence in BF16. Flash attention never writes it:

1. Split Q into row blocks. Each program instance takes one block of queries.
2. Loop over blocks of K and V, loading each into on-chip memory.
3. For each K block, compute scores for the query block, then update a **running softmax**: keep the running maximum and running sum per row, rescale the accumulated output whenever the maximum changes (the online softmax from Phase 31's exercises).
4. After the last K block, divide by the running sum and write the output once.

Memory traffic drops from order T² to order T, which makes attention compute-bound and long contexts practical. The backward pass recomputes scores block by block instead of storing them: spending FLOPs to save memory, the same trade as activation checkpointing in Phase 33.

Work through Triton's official fused-attention tutorial (`python/tutorials/06-fused-attention.py` in the Triton repository) line by line, mapping each part to the four steps above. Then benchmark it against `F.scaled_dot_product_attention` at sequence lengths 1K to 16K. PyTorch picks a fused backend automatically; `torch.nn.attention.sdpa_kernel` lets you force a specific one to compare.

### Verify

- `test_rmsnorm.py` passes for FP32 and BF16, gradients included.
- The benchmark shows the fused kernel near your Phase 22 bandwidth number.
- The 124M model trains identically (same loss curve within noise) with the Triton RMSNorm, and you know whether it's faster than `torch.compile`'s own fusion.
- You can explain, without notes, why flash attention is faster even though it does more arithmetic.

### Commit

`feat(kernels): fused Triton RMSNorm with autograd, tests and benchmarks; model integration`

### Explore (optional)

- **Fuse the backward.** Write the RMSNorm backward in Triton: one kernel for `dx` per row, and a second pass (or atomics) for `dw` across rows. Compare with [Liger Kernel](https://github.com/linkedin/Liger-Kernel)'s implementation afterward.
- **Fused SwiGLU.** Write `silu(gate) * up` and its backward in Triton, and replace the MLP's elementwise part.
- **Fused cross-entropy.** The logits for a 32K vocabulary at batch 32 × 1,024 are 4 GB in FP32. Compute loss and gradient in chunks without materializing all of it. This saves more memory than anything else at large vocabularies.
- **Autotuning.** Wrap a kernel in `@triton.autotune` with several block sizes and warp counts, and see what it picks on your GPU.
- **Faster MoE.** Sort tokens by expert, run each expert's matmul on a contiguous slice, and unsort. Check whether your PyTorch version offers a grouped matmul for this; if so, compare it with the per-expert loop from Phase 30.
- **Register as a real op.** Use `torch.library.custom_op` with a fake (shape-only) implementation for your CUDA kernel from Phase 31, so `torch.compile` can trace through it without a graph break.

## Phase 33: The 7B model

### Brief

Your GPT file already contains every part of a modern 7B model; only the numbers change. In this phase you configure a 7.25B-parameter Llama-style model, account for every gigabyte it needs, and make it trainable on two 72 GB GPUs (or three, with your spare) using sharding, activation checkpointing and your 1 TB of system RAM. You'll train it briefly from scratch to prove the stack, measure real throughput, and then do what's actually productive at this size: **continued pretraining** of an open 7B model on your own domain.

| Technique | Tool | What it saves | What it costs |
| --- | --- | --- | --- |
| Mixed precision | FSDP2 `MixedPrecisionPolicy` | Half the memory for weights and activations | Needs FP32 master weights and FP32 gradient reduction for stability |
| Sharding (ZeRO-3 style) | FSDP2 `fully_shard` | Each GPU stores 1/N of weights, gradients and optimizer state | Gathers each layer's weights before use, frees them after |
| Activation checkpointing | `checkpoint_wrapper` | Stores only each block's input, not every intermediate | Recomputes the forward pass during backward, about 30% more compute |
| CPU offload | FSDP2 `CPUOffloadPolicy` | Keeps sharded state in system RAM | PCIe transfers every step; optimizer step runs on CPU |
| Meta-device init | `torch.device("meta")` | Builds the model structure without allocating 29 GB first | Weights must be initialized or loaded after sharding |
| Distributed checkpoints | `torch.distributed.checkpoint` | Each rank saves its own shard, in parallel | Needs a conversion step to load as one file |

### The architecture

`dl/big/configs/lake7b.yaml`:

```yaml
model:
  vocab_size: 32768          # your Phase 29 tokenizer
  seq_len: 4096
  n_layer: 32
  n_head: 32
  n_kv_head: 8               # grouped-query attention, 4 query heads per KV head
  d_model: 4096
  ffn_hidden: 14336          # 3.5 x d_model, as in Llama 3 8B
  rope_theta: 500000.0       # larger base for longer contexts
  tie_embeddings: false
```

Count it, as in Phase 28. Per layer: queries and outputs 2 × 4096² = 33.6M; keys and values 2 × 4096 × 1024 = 8.4M; SwiGLU 3 × 4096 × 14336 = 176.2M. That's 218.2M per layer, 6.98B for 32 layers, plus 2 × 32768 × 4096 = 0.27B for untied embeddings: **about 7.25B parameters**. Almost 80% sits in the MLPs, which is why MoE targets them.

### The memory budget

Full training with AdamW and mixed precision needs about 16 bytes per parameter before activations:

| Item | Bytes per parameter | Total at 7.25B | Per GPU, sharded over 2 |
| --- | --- | --- | --- |
| BF16 weights | 2 | 14.5 GB | 7.3 GB |
| BF16 or FP32 gradients | 2–4 | 14.5–29 GB | 7.3–14.5 GB |
| FP32 master weights | 4 | 29 GB | 14.5 GB |
| Adam first and second moments (FP32) | 8 | 58 GB | 29 GB |
| **Total** | **16–18** | **116–130 GB** | **58–65 GB** |

On two 72 GB cards, 58–65 GB per GPU fits only on paper: activations, temporary buffers while each layer is gathered, and memory fragmentation push it over. That leaves two good plans:

1. **Two cards, with offload.** FSDP2's `CPUOffloadPolicy` keeps all sharded state in system RAM and streams each layer to the GPU when needed. It's the simplest setup, and slow. A faster variant offloads only the FP32 master weights and Adam state (about 87 GB) and keeps BF16 weights and gradients on the GPUs; DeepSpeed ZeRO-3 with `offload_optimizer` does exactly that.
2. **Three cards, no offload.** Install your spare: sharded three ways, state is about 39–43 GB per GPU, leaving roughly 30 GB each for activations. Full fine-tuning then runs entirely on GPU and is several times faster. Your ROMED8-2T has the PCIe slots for it.

With checkpointing, activations at 4,096 tokens per GPU are about 2 × d\_model × tokens × layers bytes, roughly 1 GB, plus one layer's full intermediates while recomputing. The trainer below supports both plans through its `cpu_offload` setting.

For contrast: **inference** needs only the 14.5 GB of weights plus the KV cache, and **LoRA** adds only adapter weights and their optimizer state. That's why Phases 23 and 25 use LoRA.

### Build: an FSDP2 trainer

Two small additions to `GPT` in `model.py` make meta-device initialization work: an `init_weights()` method that runs `self.apply(self._init)` plus the residual scaling loop, and a `reset_buffers()` method that recomputes the RoPE `cos` and `sin` tables on the current device. Move those lines out of `__init__` and call them from there for normal use.

`dl/big/train_fsdp.py`, the setup (the training loop itself matches Phase 29):

```python
import os, sys, yaml, torch, torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
import torch.distributed.checkpoint as dcp
sys.path.insert(0, "dl/gpt")
from model import GPT, GPTConfig

cfg = yaml.safe_load(open(sys.argv[1])); offload = cfg.get("cpu_offload", True)
dist.init_process_group("nccl"); rank = dist.get_rank(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

with torch.device("meta"):                                   # structure only, no memory yet
    model = GPT(GPTConfig(**cfg["model"]))
for i, block in enumerate(model.blocks):
    model.blocks[i] = checkpoint_wrapper(block)              # recompute activations in backward

mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
kw = {"mp_policy": mp, **({"offload_policy": CPUOffloadPolicy()} if offload else {})}
for block in model.blocks:
    fully_shard(block, **kw)                                 # each block is gathered, used, freed in turn
fully_shard(model, **kw)                                     # embeddings, final norm, head

model.to_empty(device="cpu" if offload else "cuda")          # allocate this rank's shards
with torch.no_grad():
    model.init_weights(); model.reset_buffers()
if rank == 0:
    print(f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters (counted on sharded tensors)")

opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.1,
                        foreach=True)                        # with offload, this step runs on the CPU
# ... the Phase 29 loop: micro-batches, loss.backward(), clip, opt.step() ...
# checkpoints: each rank writes its own shard
dcp.save({"model": model.state_dict(), "opt": opt.state_dict()}, checkpoint_id=f"/ckpt/lake7b/step_{step}")
```

The RoPE buffers live on the GPU while weights are offloaded; make `reset_buffers()` place them on `cuda` explicitly. If `CPUOffloadPolicy` is too slow on your PyTorch version, the main alternative is DeepSpeed ZeRO-3 with `offload_optimizer` set to CPU, which ships a vectorized CPU Adam. It's a good comparison exercise.

### Verify: prove it and measure it

```bash
make dl-dist CMD="dl/big/train_fsdp.py dl/big/configs/lake7b_smoke.yaml"
```

Use a smoke config with `seq_len: 2048`, micro-batch 1, 50 steps and your FineWeb-Edu shards.

- Startup prints about 7.25B parameters.
- Loss falls from about 10.4 (the log of 32,768, i.e. uniform guessing) over the first steps. That proves forward, backward, sharding and the optimizer all work.
- Record peak GPU memory (`torch.cuda.max_memory_allocated()`), host RAM use, seconds per step, and tokens per second.
- Kill the run after a checkpoint, restart from it, and confirm the loss continues.

Now do the projection. At 6 × 7.25B FLOPs per token, ideal throughput at 120 TFLOPS is about 2,800 tokens per second; with offload expect well under that. Compute from your measured rate how long 100M tokens and 1B tokens would take. That number is why the next part exists.

### Build: continued pretraining of an open 7B model

Pretraining from scratch at this size isn't possible on one box. Continued pretraining is: start from an open base model and keep training it on text from your domain. The model keeps its general knowledge and gains depth where you need it.

Good starting points are fully open model families that publish their data and code, such as AI2's OLMo 7B models, which make the effects of your training easiest to reason about. Base (not instruct) versions are the right starting point for continued pretraining.

1. **Build a domain corpus.** For example: Spark, Iceberg, Trino, Kafka and Airflow documentation; your own repo; and the DaVinci Resolve manual (which Phase 25 can use). Clean to plain text, deduplicate, and tokenize with **the base model's tokenizer**, not yours. Aim for 50–500M tokens.
2. **Mix in general data.** Add 10–20% FineWeb-Edu so the model doesn't forget general language (catastrophic forgetting is real at small learning rates too).
3. **Load and shard.** Load the Hugging Face model with `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)`, then apply `checkpoint_wrapper` and `fully_shard` to each decoder layer in `model.model.layers` exactly as above.
4. **Train gently.** Peak learning rate around 1e-5 to 3e-5, short warmup, cosine decay, sequence length 4,096, about 0.5–1M tokens per step through gradient accumulation.
5. **Evaluate both ways.** Measure perplexity on held-out domain text (should drop clearly) and on general text plus HellaSwag (should barely move). Log both to MLflow at every checkpoint.

A cheaper variant for a first pass: continued pretraining with QLoRA (Phase 23's tools) on one GPU. It adapts less deeply but finishes in hours.

### Commit

`feat(big): 7.25B Llama-style config, FSDP2 trainer with checkpointing and CPU offload, DCP checkpoints`

### Explore (optional)

- **Where does the time go?** Profile one step with offload on and off (use a 3B config so offload-off fits). Measure PCIe transfer time against compute.
- **Gradient precision.** Compare `reduce_dtype` of BF16 versus FP32 on a 1B config's loss curve over 1,000 steps.
- **8-bit optimizer states.** Try a memory-efficient optimizer (8-bit Adam, Adafactor, or a newer one such as Muon) on the 1B config and compare loss and memory.
- **Long context.** Train the 124M model at 1K, then extend to 8K with a larger RoPE base and a short fine-tune at the longer length. Measure loss by position to see whether it really uses the extra context.
- **Convert and serve.** Convert your DCP checkpoint to a single `safetensors` file with `torch.distributed.checkpoint.format_utils`, write a Hugging Face-compatible config (your architecture matches Llama's), and serve it with vLLM.

## Phase 34: Beyond next-token prediction — a JEPA from scratch

### Brief

Every model so far predicts the raw data itself: the next token, or a class label. A **joint-embedding predictive architecture (JEPA)** predicts in representation space instead. It shows the model part of an input, and asks it to predict the *embedding* of the hidden part, not its pixels. The argument from LeCun's group is that predicting every pixel wastes capacity on unpredictable detail (exact leaf textures), while predicting embeddings forces the model to learn what's predictable: objects, shapes, layout. I-JEPA applies this to images; V-JEPA and V-JEPA 2 apply it to video and use the result as a world model for planning. You'll build I-JEPA at small scale and test its features with a linear probe.

| Self-supervised family | Training signal | Weak spot |
| --- | --- | --- |
| Masked autoencoder (MAE) | Reconstruct masked pixels | Spends effort on low-level detail |
| Contrastive (SimCLR, CLIP-style) | Pull two views of an image together, push other images apart | Needs hand-designed augmentations and many negatives |
| JEPA | Predict a hidden region's embedding from a visible region's embedding | Can collapse to a constant output; the design must prevent it |

| Component | Role |
| --- | --- |
| Context encoder (a ViT) | Embeds only the visible patches; this is what you keep after training |
| Target encoder | Same architecture; embeds all patches. Its weights are a slow moving average (EMA) of the context encoder's, with no gradients |
| Predictor (a narrow ViT) | Takes context embeddings plus mask tokens at the target positions and predicts the target embeddings |
| Masking strategy | A large context block, and several smaller target blocks removed from it |
| Loss | Distance between predicted and actual target embeddings |

**Why doesn't it collapse?** If both encoders output the same constant, the loss would be zero. Two asymmetries prevent that: the target encoder gets no gradients and changes only slowly through the EMA, and only the context side has a predictor. Monitoring the spread of embeddings across a batch is how you'd catch collapse anyway.

### Build: data

STL-10 suits this well: 100,000 unlabeled 96 × 96 images for self-supervised training, plus a small labeled set (5,000 train, 8,000 test, 10 classes) for evaluation. `torchvision.datasets.STL10` downloads it (`split="unlabeled"` and `split="train"` / `"test"`). Use light augmentation for JEPA: random resized crop and horizontal flip only.

### Build: the model

`dl/jepa/ijepa.py`:

```python
import copy, math, random, torch, torch.nn as nn, torch.nn.functional as F

IMG, P = 96, 8; G = IMG // P                      # 12 x 12 grid of 8-pixel patches

def sincos_2d(d, g):
    y, x = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    om = 1.0 / 10000 ** (torch.arange(d // 4) / (d // 4))
    parts = [f(c.flatten()[:, None] * om[None]) for c in (x, y) for f in (torch.sin, torch.cos)]
    return torch.cat(parts, 1)                      # (g*g, d), fixed positional embedding

def blocks(d, depth, heads):
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)

class Encoder(nn.Module):
    def __init__(self, d=384, depth=12, heads=6):
        super().__init__()
        self.patch = nn.Conv2d(3, d, P, P)
        self.register_buffer("pos", sincos_2d(d, G))
        self.body, self.norm = blocks(d, depth, heads), nn.LayerNorm(d)
    def forward(self, x, keep=None):                # keep: indices of visible patches
        t = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        if keep is not None:
            t = t[:, keep]
        return self.norm(self.body(t))

class Predictor(nn.Module):
    def __init__(self, d=384, dp=192, depth=6, heads=6):
        super().__init__()
        self.inp, self.out = nn.Linear(d, dp), nn.Linear(dp, d)
        self.mask = nn.Parameter(torch.zeros(1, 1, dp)); nn.init.trunc_normal_(self.mask, std=0.02)
        self.register_buffer("pos", sincos_2d(dp, G))
        self.body, self.norm = blocks(dp, depth, heads), nn.LayerNorm(dp)
    def forward(self, ctx, ctx_idx, tgt_idx):
        z = self.inp(ctx) + self.pos[ctx_idx]
        m = self.mask.expand(ctx.size(0), len(tgt_idx), -1) + self.pos[tgt_idx]   # "predict what's here"
        h = self.norm(self.body(torch.cat([z, m], 1)))
        return self.out(h[:, -len(tgt_idx):])

def rect(scale, aspect):
    area = random.uniform(*scale) * G * G; ar = math.exp(random.uniform(*map(math.log, aspect)))
    h = max(1, min(G, round(math.sqrt(area * ar)))); w = max(1, min(G, round(math.sqrt(area / ar))))
    top, left = random.randint(0, G - h), random.randint(0, G - w)
    return {(top + i) * G + left + j for i in range(h) for j in range(w)}

def sample_masks(n_targets=4):
    """One mask set per batch, for simplicity (the paper samples per image)."""
    targets = [rect((0.15, 0.2), (0.75, 1.5)) for _ in range(n_targets)]
    ctx = rect((0.85, 1.0), (1.0, 1.0)) - set().union(*targets)   # context never overlaps targets
    to_t = lambda s: torch.tensor(sorted(s), device="cuda")
    return to_t(ctx), [to_t(t) for t in targets]
```

### Build: training

`dl/jepa/train_ijepa.py`, the core loop:

```python
enc = Encoder().cuda(); pred = Predictor().cuda()
tgt = copy.deepcopy(enc).requires_grad_(False)                  # the target encoder
params = list(enc.parameters()) + list(pred.parameters())
opt = torch.optim.AdamW(params, lr=1.5e-4, weight_decay=0.05)
total = epochs * len(loader)

for step, (x, _) in enumerate(train_iter(loader, epochs)):
    x = x.cuda(non_blocking=True)
    ctx_idx, tgt_list = sample_masks()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            h = tgt(x)                                            # embeddings of ALL patches
            h = F.layer_norm(h, h.shape[-1:])                     # normalized targets
        z = enc(x, keep=ctx_idx)                                  # visible patches only
        loss = sum(F.smooth_l1_loss(pred(z, ctx_idx, t), h[:, t]) for t in tgt_list) / len(tgt_list)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m = 0.996 + (1.0 - 0.996) * step / total                     # EMA momentum ramps toward 1
    with torch.no_grad():
        for pt, pc in zip(tgt.parameters(), enc.parameters()):
            pt.mul_(m).add_(pc.detach(), alpha=1 - m)
    if step % 100 == 0:
        spread = z.float().std(dim=0).mean().item()               # near zero = collapse
        mlflow.log_metrics({"loss": loss.item(), "embedding_std": spread, "ema": m}, step)
```

Add a cosine learning-rate schedule with warmup as in earlier phases, batch size 256, and 100–200 epochs over the unlabeled split. A ViT-Small (about 22M parameters) at this resolution trains on one GPU in several hours.

### Build: evaluate with a linear probe

Freeze the context encoder, average its patch embeddings into one vector per image, and train only a linear classifier on the 5,000 labeled STL-10 images. Test on the 8,000 test images. Run the same probe on:

1. a randomly initialized encoder (the floor),
2. your I-JEPA encoder,
3. optionally, the same ViT trained with labels from scratch on the 5,000 images (shows how much the unlabeled data helped).

### Verify

- Loss decreases steadily and `embedding_std` stays well above zero. If it heads to zero, you have collapse: check that the target encoder has no gradients and that the EMA runs.
- Linear-probe accuracy for I-JEPA is far above the random encoder's. Record all three probe numbers.
- Visualize nearest neighbors: for a few test images, find the closest training images by embedding. They should share objects and shape, not just colors.

### Commit

`feat(jepa): I-JEPA with EMA target encoder, block masking, predictor, linear-probe evaluation`

### Explore (optional)

- **Against MAE.** Train a masked autoencoder of the same size on the same data (decode masked patches back to pixels) and compare linear-probe accuracy and training time.
- **Mask design.** Try small random patches instead of blocks, or targets that overlap the context. I-JEPA's paper found block masking essential; see if you agree.
- **Remove an asymmetry.** Set EMA momentum to 0 (target equals context encoder) or drop the predictor, and watch for collapse.
- **Video.** Extend to short video clips with 3D patches (time × height × width), predicting masked space-time regions, in the direction of V-JEPA. Screen recordings from Phase 25 make an interesting dataset.
- **Planning in latent space.** With a video JEPA, learn to predict the embedding of the next frame given the current one plus an action. That's the basic shape of a learned world model.

## Phase 35: A playbook for specialized models

### Brief

You now have every tool from a single neuron to RL on a 7B model. The skill that matters most is choosing the cheapest one that solves the problem. Work down this ladder and stop at the first rung that meets your bar, measured on an evaluation you wrote **before** training.

| Rung | Method | When it's enough | Cost on your box |
| --- | --- | --- | --- |
| 1 | Prompting and few-shot examples | The model knows the domain; it needs format or style | Minutes |
| 2 | Retrieval (RAG) over your documents | The gap is facts that change, or private knowledge | Hours |
| 3 | LoRA SFT | It needs a new skill or output format, with hundreds to thousands of examples | Hours on one GPU |
| 4 | DPO or GRPO on top of SFT | You can rank answers, or better, verify them with a program | Hours to a day |
| 5 | Continued pretraining, then 3–4 | The domain's language itself is unfamiliar (a new field, codebase or notation) | Days |
| 6 | Train a small model from scratch | A narrow task where a 100M–1B model is enough and speed matters | Days |
| 7 | A new architecture | The data has structure existing models handle badly (Phase 34's motivation) | Open-ended research |

Three rules held in every phase of this tutorial and will hold for whatever you build next:

1. **The evaluation comes first.** Phase 23's execution accuracy and Phase 25's state checks made progress measurable. Without them, every change is guesswork.
2. **Data beats architecture.** Verified examples, held-out templates and on-policy preference pairs did more than any hyperparameter.
3. **Measure your own hardware.** Your Phase 22 benchmarks, not spec sheets, decided what was feasible.

### Where to go next

| Direction | What you'd learn | Starting point |
| --- | --- | --- |
| FP8 training | Blackwell's tensor cores run FP8 at twice BF16's rate; scaling and stability tricks | PyTorch's `torchao` float8 training on your 124M and 1B configs |
| Inference optimization | Quantization (FP8, INT4, FP4), speculative decoding, paged attention | Serve your 7B model in vLLM at different precisions and measure tokens per second and quality |
| Distillation | Training a small model to imitate a large one | Use a 7B model's outputs to train your 124M–350M model for one task |
| State-space and hybrid models | Linear-time sequence mixing (Mamba-style layers) | Replace some attention layers in your GPT and compare at long context |
| Diffusion language models | Generating text by iterative denoising instead of left to right | Adapt your GPT into a masked-diffusion model on TinyStories |
| Multi-node training | Real network communication, tensor and pipeline parallelism | A second GPU machine on your homelab network, with FSDP plus tensor parallelism |
| Agents with memory | Long-horizon tasks that span sessions | Extend Phase 25 with a task notebook the agent reads and writes |

### Planned next tutorials

Once Parts A and B are done, two follow-ons would close the loop, if they still make sense by then:

1. **An inference engine.** Serve your own trained models and popular mid-sized open models efficiently: continuous batching, a paged KV cache, quantization (FP8, INT4, FP4 on Blackwell), speculative decoding and multi-LoRA serving, built up yourself and compared against vLLM on your GPUs.
2. **An agent harness.** The layer that turns models into dependable tools: tool calling, memory, the decision gate from Phase 25 generalized, evaluation suites and tracing, running on your own served models.

Both would continue in the same repo under the same conventions, as new stacks in `platform/`.

## Sources

- [RTX PRO 5000 Blackwell specifications (third-party)](https://hmc-tech.com/gpus/nvidia-rtx-pro-5000-blackwell-vs-nvidia-rtx-pro-6000-blackwell)
- [NVIDIA RTX PRO 5000 Blackwell datasheet](https://www.pny.com/File%20Library/Company/Support/Product%20Brochures/NVIDIA%20RTX/English/nvidia-rtx-pro-5000-blackwell.pdf)
- [nanochat (PyPI page, FLOPs of the speedrun)](https://pypi.org/project/nanochat)
- [nanochat repository and model miniseries](https://github.com/ido/nanochat)
- [TRL GRPO trainer and custom reward functions](https://huggingface.co/docs/trl/en/grpo_trainer)
- [vLLM with TRL: server and colocate modes](https://docs.vllm.ai/training/trl.html)
- [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
- [GroundGUI: GUI grounding with SFT and RL](https://arxiv.org/pdf/2610.19100)
- [KV-Ground: Qwen3-VL-based GUI grounding model](https://huggingface.co/vocaela/KV-Ground-8B-BaseGuiOwl1.5-0315)

* [What is Jev? (TypeSafe's typed-decision model)](https://medium.com/@sathishkraju/what-is-jev-a-practical-look-at-typesafes-system-one-model-3b7c0fe34f6b)
* [Laya model guide: architecture, limits and API](https://dev.to/aimodels-fyi/a-beginners-guide-to-the-laya-model-by-convaiinnovations-on-huggingface-bg)
* [Releasing the Inner Jev: open decision models from LLM letter readouts](https://arxiv.org/pdf/2610.17205)
* [Do a typed-decision model's probabilities obey the probability axioms?](https://arxiv.org/pdf/2609.33209)
* [Labels override definitions in Jev-style models](https://arxiv.org/pdf/2610.02586)
* [JevAdvBench: adversarial robustness of calibrated decision models](https://arxiv.org/pdf/2609.31142)
* [NVIDIA RTX PRO 5000 Blackwell datasheet (48 and 72 GB versions)](https://www.pny.com/File%20Library/Company/Support/Product%20Brochures/NVIDIA%20RTX/English/nvidia-rtx-pro-5000-blackwell.pdf)

