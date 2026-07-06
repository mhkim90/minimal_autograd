# AGENTS.md

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

## Coding Standards

Load these skills at the start of relevant Codex tasks:

- **karpathy**: When building, modifying, reviewing, or debugging code; keep changes simple, surgical, and goal-driven.
- **grilled-me**: When drafting or reviewing plans; stress-test assumptions, risks, scope creep, and failure modes before presenting.
- **handoff**: When ending a session, switching context, or preserving progress for another agent.
- **caveman**: When token efficiency is requested; use `full` for English and `korean-full` for Korean unless the user requests another level.
- **phase-gated-implementation**: When implementing multi-phase work from a plan, PR description, issue, design doc, or approved checklist.
- **opencode-delegate**: When tedious, mechanical, or long-running work can be delegated; keep final verification local.
- **claude-delegate**: When an independent read-only second opinion is useful for a plan, risk review, or reasoning check.

See [.codex/skills/karpathy/SKILL.md](.codex/skills/karpathy/SKILL.md), [.codex/skills/grilled-me/SKILL.md](.codex/skills/grilled-me/SKILL.md), [.codex/skills/handoff/SKILL.md](.codex/skills/handoff/SKILL.md), [.codex/skills/caveman/SKILL.md](.codex/skills/caveman/SKILL.md), [.codex/skills/phase-gated-implementation/SKILL.md](.codex/skills/phase-gated-implementation/SKILL.md), [.codex/skills/opencode-delegate/SKILL.md](.codex/skills/opencode-delegate/SKILL.md), and [.codex/skills/claude-delegate/SKILL.md](.codex/skills/claude-delegate/SKILL.md).

### Quick Reminders

- State assumptions explicitly; surface ambiguity before writing code.
- Write the minimum code that solves the problem; no speculative abstractions.
- Touch only the files and lines required; match existing style.
- Define success criteria before editing; verify with the smallest useful check.
- Before acting on a plan, ask: am I solving more than asked? What is the most likely failure?
