# OpenCode five-role roster

This repository is `minimal_autograd`, a C++17 reverse-mode automatic
differentiation library with an optional CUDA backend. Follow the repository
root `AGENTS.md`, existing CMake conventions, and the approved phase scope.
Use focused CMake/CTest checks and CUDA tests when relevant.

## Roles and routing

- **Astra orchestrator** (`agents/astra-orchestrator.md`) is an explicit
  controller-bound whole-phase route (`agent="astra-orchestrator"`). It plans,
  gates, and recommends stop/go beneath an external Codex or Claude controller;
  it is not a standalone primary route. The controller owns final gates,
  independent review, and publication. This boundary is procedural, not caller
  authentication.
- **Astra expert** (`agents/astra-expert.md`) is fresh, bounded read-only
  preflight or blocker analysis: one consultation and at most one follow-up.
  It never edits, delegates, publishes, or reviews its own recommendation.
- **Luna** (`agents/luna.md`) is the normal implementation route
  (`agent="luna"`) for mechanical/economy and standard work. It cannot publish
  or broaden scope. Follow **red -> implement -> green -> GPU** with at most 3
  attempts; after two failures on one blocker, consult Astra expert and attempt
  a third time only after a materially revised approach.
- **Sol implementer** (`agents/sol-implementer.md`) is the justified strong
  implementation route; **Astra implementer** (`agents/astra-implementer.md`)
  is the justified hard-task implementation route. Both use Luna's bounded
  edit/test contract, never plan, review, delegate, stage, commit, push, create
  PRs, or publish. Omit model and variant for every named role; a conflicting
  override stops rather than silently changing model or effort.
- The configured default route is used only when the user explicitly requests
  it: omit `agent`, `model`, and `variant`. Never infer it as a named role.

## PR lifecycle and approval

Maintain a non-authoritative lifecycle ledger entry for every workflow-touched
PR, including created, adopted, updated, readied, reopened, closed, and merged
events. Record the host/repository/PR, source-or-target role, branch, head,
base, complete file set and modes, classification, lifecycle state, owner
disposition, authority evidence, and last validation. The ledger grants no authority. Every
entry must be merged, explicitly retained by PR-specific boundary-scoped owner
direction, or explicitly closed before successful phase completion, entering
implementation, downstream sync, or success reporting; unresolved entries may
only appear in a terminal Stop report.

Combined plan-only approval and verified non-operational documentation-only
approval are narrow exceptions. Bind and revalidate repository, PR, head, base,
eligibility, complete file set, file modes, and classification immediately
before and after approval. Plan-plus-implementation, operational or uncertain
documentation, profiles, skills, agents, AGENTS.md, CLAUDE.md, `.opencode`, policy,
workflow, configuration, code, generated, renamed, symlinked, or executable
content retains ordinary final merge control.

## Gates and safety

Role prompts are authoritative. L3 requires fresh expert preflight and a
distinct independent post-implementation reviewer through the external
controller; if that reviewer is unavailable, stop. L4 requires explicit owner
direction before implementation. Delegated paths stay within
the caller's working directory; external paths are denied. Permissions are
guardrails, not a complete trust boundary, so preserve repository boundaries,
sandboxing, and diff gates. Keep unrelated artifacts untouched, and never read,
print, copy, expose, or request credentials or secrets.
