# OpenCode Triad

This repository is `opencode_mcp`, a Python 3.12 MCP server. The main
implementation is `server.py`; tests are under `tests/` and use pytest with
mocked HTTP interactions. There is no compiled build. GPU access may be used
for an explicit smoke check when relevant, but it is not required for normal
tests.

## Roles and routing

- **Sol** (`agents/sol.md`) is the explicit whole-phase triad route
  (`agent="sol"`): planner and decider for risk, gates, delegation, and
  stop/go. Beneath an external controller it returns diff/evidence only and
  does not commit, push, or create a PR; the controller retains publication.
- **Sol-expert** (`agents/sol-expert.md`) is bounded, read-only difficult
  preflight or breakthrough analysis: one initial consultation and at most one
  follow-up. It never edits, delegates, or publishes.
- **Luna** (`agents/luna.md`) is the normal implementation route
  (`agent="luna"`) for mechanical/economy and standard work. It cannot publish
  or broaden scope. Follow **red -> implement -> green -> GPU** with at most 3
  attempts; after two failures on one blocker, consult Sol-expert and attempt
  a third time only after a materially revised approach.
- **Terra implementer** (`agents/terra-implementer.md`) is the explicit exact-
  model implementation route (`agent="terra-implementer"`) only after an
  explicit owner exact-model request or an already-approved recorded route.
  Never pass a model or variant with this named agent; doing so is a route
  contradiction and stops the phase. It is a semantic Luna clone for bounded
  edits, focused inspection, and tests, with no planner, reviewer, delegation,
  staging, commit, push, PR, or publication authority.
- **Sol implementer** (`agents/sol-implementer.md`) has the same bounded Luna
  implementation contract for an explicit exact-model Sol request
  (`agent="sol-implementer"`). A model or variant with the named agent is a
  contradiction and stops the phase; it cannot substitute for Sol planning,
  Terra review, delegation, staging, commit, push, PR, or publication.
- **Terra** (`agents/terra.md`) remains an explicit, fresh-context, read-only
  review (`agent="terra"`) with a recorded reason. Only explicit whole-phase
  Sol mode makes L3 preflight and non-trivial post-review mandatory; Terra is
  never a routine external-controller review.
- **Astra expert** (`agents/astra-expert.md`) is the named, fresh, read-only
  Astra consultation route (`agent="astra-expert"`) for the existing bounded
  escalation cases. Omit model and variant; selector/model/role mismatch stops
  rather than silently falling back. It never implements, publishes, delegates,
  waives L4 direction, resets retries, replaces required review, or self-reviews.
- The configured default route is used only when the user explicitly requests
  it: omit `agent`, `model`, and `variant`. Never infer it as a default Sol or
  Terra route.

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

Role prompts are authoritative. In explicit whole-phase Sol mode, L3 requires
Terra preflight; L4 stops before implementation. Delegated paths stay within
the caller's working directory; external paths are denied. Permissions are
guardrails, not a complete trust boundary, so preserve repository boundaries,
sandboxing, and diff gates. Keep unrelated artifacts untouched, and never read,
print, copy, expose, or request credentials or secrets.
