# CLAUDE.md

## Purpose

This repository is a minimal reverse-mode automatic differentiation library in C++17, built on top of Eigen3. It implements a tape-based autograd engine, a module system (`Linear`, `Sequential`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `DepthwiseConv2d`, `NearestUpsample2d`, `GroupNorm`), losses (`mse_loss`, `cross_entropy`), optimizers (`SGD`, `Adam`), and diffusion-model primitives (`randn`, `sinusoidal_time_embedding`, `q_sample`). CPU (Eigen, optionally OpenMP) is the default backend; an optional CUDA backend (`-DAUTOGRAD_USE_CUDA=ON`) mirrors core ops, `Linear`, losses, `Conv2d`, and `MaxPool2d` via `.cuda()`/`.cpu()` on `Var`. Intended for teaching and small experiments. Use these instructions for all code work in this repo.

## General Rules

- Prefer small, surgical changes.
- Keep code simple and avoid speculative abstractions.
- Use the existing repository structure and conventions.
- When multiple independent reads are needed, use parallel tool calls.
- Before editing, inspect the relevant files first.
- After edits, run the smallest useful validation for the touched area.

## Build, Test, and Validation

- Build with CMake: `mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)`
- Core test binaries: `./test_core`, `./test_nn`, `./test_conv` — all should print `ALL TESTS PASSED`.
- Optional CUDA backend: configure with `-DAUTOGRAD_USE_CUDA=ON` (needs CMake 3.18+ and a CUDA toolkit; use the `cuda` symlink, e.g. `PATH=/usr/local/cuda/bin:$PATH`, if a versioned `cuda-X.Y` path isn't directly accessible). Adds `./test_cuda_core`, which should print `ALL CUDA CORE TESTS PASSED`. CUDA `Conv2d`/`MaxPool2d` are cross-checked against the CPU path in that test — keep both numerically consistent when touching either.
- Optional extended ops/tests: configure with `-DAUTOGRAD_BUILD_ADVANCED_OPS=ON` to build `./test_extensions`, `./test_diffusion`, `./test_smoke`.
- If a change affects a single op or module, run only the relevant test binary.
- Do not introduce new tooling unless it is clearly needed.

## Library and Dependency Rules

- Prefer the libraries already used in the repository (Eigen3, C++17 stdlib).
- Avoid adding a new dependency unless it materially improves correctness or maintainability.
- If a new library is necessary, keep the scope narrow and document why it is needed.
- Keep includes minimal and remove unused dependencies when possible.

## Coding Standards

Always load these skills at the start of relevant tasks:

- **karpathy**: When building/writing code — Karpathy's LLM coding best practices (simplicity, surgical changes, goal-driven execution)
- **grilled-me**: When planning or reviewing plans — adversarial self-review to stress-test a plan before presenting
- **handoff**: When ending a session or switching context — write a HANDOFF.md so the next agent can continue seamlessly
- **caveman**: When token efficiency is needed — ultra-compressed output (~75% fewer tokens) while keeping full technical accuracy; trigger with `/caveman` or "talk like caveman"
- Language-aware caveman default: use `full` for English responses and `korean-full` for Korean responses unless the user explicitly requests another level.
- **opencode-delegate**: Delegate **tedious / mechanical / long-running** work — applying precise edits, sweeps, run→inspect→tweak loops, test runs. Via `mcp__opencode__opencode_run_async` / `mcp__opencode__opencode_session_fork_async` plus job/session tools.
- **codex-delegate**: Delegate **hard reasoning, adversarial design review, and difficult implementation** (numerical correctness, gradient derivation, autograd engine design, optimizer behavior) to Codex (GPT-5.5). DISCUSS (read-only) for reasoning & second opinions; EXECUTE (workspace-write) for hard multi-file changes. Via `mcp__codex__codex` / `mcp__codex__codex-reply`.

### Delegation model

Claude's role can be to **manage / orchestrate**, not to do all the heavy work
inline: decompose the goal, write precise cold-context specs, route each piece to the
right engine, verify results (the relevant test binaries above), decide keep/revert,
and report.

- **Codex** ← hard reasoning, adversarial review, and implementing *difficult* things.
- **OpenCode** ← tedious / mechanical / long-running jobs.
- **Claude** ← routing, verification, and decisions only.

Before doing hard reasoning or a large edit inline, ask "Codex (hard) or OpenCode
(tedious)?" — default to delegating for tasks that fit either bucket cleanly.

> **MCP availability restriction**: `mcp__opencode__*` and `mcp__codex__*` tools are
> **only available when Claude is running as a remote-controlled agent (Claude Code Remote / CCR)**.
> They are NOT accessible in standard local Claude Code sessions. Before invoking
> either `opencode-delegate` or `codex-delegate`, confirm the session has these MCP
> tools loaded (they appear in the deferred-tools list). If they are absent, fall
> back to inline execution.
> Also note: the Codex MCP environment does **not** provide GPU access. Do not
> delegate GPU-dependent builds, CUDA tests, benchmarks, profilers, or device
> probes to `codex-delegate`; use Codex only for GPU/CUDA reasoning or code edits,
> then run GPU validation from Claude/OpenCode/local shell where a GPU is available.

See [skills/karpathy-best-practices/SKILL.md](skills/karpathy-best-practices/SKILL.md), [skills/grilled-me/SKILL.md](skills/grilled-me/SKILL.md), [skills/handoff/SKILL.md](skills/handoff/SKILL.md), [skills/caveman/SKILL.md](skills/caveman/SKILL.md), [skills/opencode-delegate/SKILL.md](skills/opencode-delegate/SKILL.md), and [skills/codex-delegate/SKILL.md](skills/codex-delegate/SKILL.md).

### Quick Reminders

- State assumptions explicitly; surface ambiguity before writing code.
- Write the minimum code that solves the problem — no speculative abstractions.
- Touch only the files and lines required; match existing style.
- Define success criteria before editing; verify with the smallest useful check.
- Before acting on a plan, ask: am I solving more than asked? What is the most likely failure?
