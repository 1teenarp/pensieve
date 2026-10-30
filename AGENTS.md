# Working agreement for coding agents

You are building this repo one tutorial phase at a time. The tutorials in `docs/tutorials/` are the
plan; `README.md` explains the project. Read both before your first phase.

## Starting a phase
- Work only on the phase you were assigned. Do not start the next one.
- Read the phase in full before changing anything.
- Read the previous phase's journal entry and carry forward any deviations it records.
- Branch: `phase/<tag>`, for example `phase/t1-p03`. Tutorial 0 uses `phase/t0`.

## Journal
- At the start of the phase, copy `docs/journal/TEMPLATE.md` to `docs/journal/<tag>.md`.
- Fill it as you go, not at the end:
  - Verify results: each Verify command and its real output or numbers.
  - Deviations: record each one when you make it, with the error that forced it.
  - Findings: anything surprising (timings, sizes, versions, behaviour).
  - Needs owner action: the exact commands or decisions you are waiting on.
- Write "Built" last, as a short summary.
- Commit the entry in the same pull request as the code.

## Doing the work
- Follow the Build steps in order. Run every Verify step and record the actual result.
- Keep the diff inside the phase's scope. No drive-by refactors.
- If tutorial code fails with current versions, make the smallest fix that works and record it under
  Deviations, with the error you saw.
- If a Verify step still fails after one reasonable fix, stop. Write what failed in the journal and
  open a draft pull request titled "BLOCKED: <tag>". Do not work around it silently.
- The final commit message is the phase's Commit line.

## Owner-only steps: prepare, never run
- Anything on the host outside Docker: packages, BIOS, drivers, NVIDIA toolkit, modprobe, mkfs, ZFS,
  fstab, reboots.
- Anything destructive: `rm -rf` on data folders, `docker volume rm`, dropping tables or buckets,
  deleting checkpoints, `git push --force`.
- Write the exact commands in the journal under "Needs owner action", and wait.

## Never
- Commit `.env`, secrets, data, checkpoints, model weights or screen recordings.
- Use a `latest` tag, or change `versions.env` without explaining why in the journal.
- Merge your own pull request or create tags. The owner does both.

## Done means
- Every Verify step passed, with results in `docs/journal/<tag>.md`.
- `pre-commit` passes.
- The pull request uses the template and links the journal entry.
