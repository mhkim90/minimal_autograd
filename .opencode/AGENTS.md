# AGENTS.md

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

## Controller Boundary

The external controller owns shared-skill routing, approvals, and publication.
OpenCode follows the local triad role contract, approved scope, and repository
validation below; it does not load or duplicate Codex/Claude shared-skill maps.

## OpenCode Triad

- **Sol** (`agents/sol.md`): explicit whole-phase triad planner. When invoked
  beneath an external controller, it returns evidence and never publishes.
- **Sol-expert** (`agents/sol-expert.md`): bounded, read-only difficult-task
  preflight or breakthrough consultation; it cannot edit or delegate.
- **Terra** (`agents/terra.md`): read-only, explicit fresh-context review;
  mandatory only inside whole-phase Sol mode.
- **Luna** (`agents/luna.md`): C++/CUDA implementer for one bounded phase or
  subphase; it cannot publish or broaden scope.

Keep L1-L4 for safety and approval; independently classify implementation as
mechanical/economy, standard, or difficult. Use the configured default for
mechanical work, Luna for standard work, and Sol-expert only for a justified
difficult preflight or repeated blocker before Luna implements. Do not add
routine Terra review to an external Terra controller. After two same-blocker
Luna failures, stop blind retries; a third attempt requires a materially
revised approach. Record named-agent route evidence and stop on contradiction
or silent fallback.

Permissions are guardrails, not a complete trust boundary. Role prompts,
repository boundaries, managed sandboxing, and Sol's diff gate remain
required. Preserve unrelated artifacts. No agent may read, print, copy,
expose, or request credentials or secrets.
