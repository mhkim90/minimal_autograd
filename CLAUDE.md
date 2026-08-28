# CLAUDE.md

## Purpose

This repository is a minimal reverse-mode automatic differentiation library in
C++17, built on top of Eigen3. It implements a tape-based autograd engine, a
module system (`Linear`, `Sequential`, `Conv2d`, `MaxPool2d`, `AvgPool2d`,
`DepthwiseConv2d`, `NearestUpsample2d`, `GroupNorm`), losses (`mse_loss`,
`cross_entropy`), optimizers (`SGD`, `Adam`), and diffusion-model primitives
(`randn`, `sinusoidal_time_embedding`, `q_sample`). CPU (Eigen, optionally
OpenMP) is the default backend; an optional CUDA backend
(`-DAUTOGRAD_USE_CUDA=ON`) mirrors core ops, `Linear`, losses, `Conv2d`, and
`MaxPool2d` via `.cuda()`/`.cpu()` on `Var`. Intended for teaching and small
experiments. Use these instructions for all code work in this repo.

## General Rules

- Prefer small, surgical changes.
- Keep code simple and avoid speculative abstractions.
- Use the existing repository structure and conventions.
- When multiple independent reads are needed, use parallel tool calls.
- State assumptions explicitly and surface ambiguity before writing code.
- Define success criteria before editing and verify with the smallest useful
  check.
- Before editing, inspect the relevant files first.
- After edits, run the smallest useful validation for the touched area.

## Build, Test, and Validation

- Build with CMake: `mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)`
- Core test binaries: `./test_core`, `./test_nn`, `./test_conv` -- all should print `ALL TESTS PASSED`.
- Optional CUDA backend: configure with `-DAUTOGRAD_USE_CUDA=ON` (needs CMake 3.18+ and a CUDA toolkit; use the `cuda` symlink, e.g. `PATH=/usr/local/cuda/bin:$PATH`, if a versioned `cuda-X.Y` path isn't directly accessible). Adds `./test_cuda_core`, which should print `ALL CUDA CORE TESTS PASSED`. CUDA `Conv2d`/`MaxPool2d` are cross-checked against the CPU path in that test; keep both numerically consistent when touching either.
- Optional extended ops/tests: configure with `-DAUTOGRAD_BUILD_ADVANCED_OPS=ON` to build `./test_extensions`, `./test_diffusion`, `./test_smoke`.
- If a change affects a single op or module, run only the relevant test binary.
- Do not introduce new tooling unless it is clearly needed.

## Library and Dependency Rules

- Prefer the libraries already used in the repository (Eigen3, C++17 stdlib).
- Avoid adding a new dependency unless it materially improves correctness or maintainability.
- If a new library is necessary, keep the scope narrow and document why it is needed.
- Keep includes minimal and remove unused dependencies when possible.

## Plan Authority

- The applicable checked-in plan is authoritative for its scope, phase
  boundaries, and gates. Plan approval does not authorize unrelated changes.
- If implementation evidence conflicts with the applicable plan, stop and
  report the conflict before editing.

## Claude Workflow

Use these skills only when their named trigger applies:

- **karpathy**: When building, modifying, reviewing, or debugging code; keep
  changes simple, surgical, and goal-driven.
- **grilled-me**: When drafting or reviewing plans; stress-test assumptions,
  scope, and failure modes before presenting.
- **plan-audit**: When an externally authored plan, issue, PR description, or
  design document must be audited before phased implementation.
- **handoff**: When ending a session, switching context, or preserving progress
  for another agent.
- **memory-continuity**: When resuming or crossing repositories; current Git,
  plan, handoff, and test evidence remain authoritative.
- **phase-gated-implementation**: For approved multi-phase work; it owns
  controller, routing, evidence, approval, and lazy policy loading.
- **opencode-delegate**: For approved tedious, mechanical, or long-running
  work; follow its async lifecycle and stop/degrade rules.

For phased work, use `plan-audit` for an external artifact and `grilled-me` for
a plan drafted in-session. Resolve blocking findings and obtain owner approval
for required scope and gates before relying on continuation.

Use Caveman `full` / `korean-full` by default; load `caveman` only to change
mode or apply its ambiguity safeguards.

## GitHub Operations

- Use normal authenticated `gh` commands for GitHub operations when permitted
  by Claude's current permissions.
- Use local `git` for worktree inspection and local version-control operations.
- Never read, print, or expose GitHub credentials or tokens.

See [skills/karpathy-best-practices/SKILL.md](.claude/skills/karpathy-best-practices/SKILL.md), [skills/grilled-me/SKILL.md](.claude/skills/grilled-me/SKILL.md), [skills/plan-audit/SKILL.md](.claude/skills/plan-audit/SKILL.md), [skills/handoff/SKILL.md](.claude/skills/handoff/SKILL.md), [skills/memory-continuity/SKILL.md](.claude/skills/memory-continuity/SKILL.md), [skills/phase-gated-implementation/SKILL.md](.claude/skills/phase-gated-implementation/SKILL.md), and [skills/opencode-delegate/SKILL.md](.claude/skills/opencode-delegate/SKILL.md).
