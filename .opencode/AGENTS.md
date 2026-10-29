# OpenCode roles

This repository is `minimal_autograd`, a C++17 reverse-mode automatic
differentiation library with an optional CUDA backend. Follow the repository
root `AGENTS.md`, existing CMake conventions, and the approved phase scope.
Use focused CMake/CTest checks and CUDA tests when relevant.

## Roles and routing

- [Luna](agents/luna.md) is the normal standard/mechanical implementation route, `agent="luna"`.
- [Sol implementer](agents/sol-implementer.md) handles justified strong implementation, `agent="sol-implementer"`.
- [Terra implementer](agents/terra-implementer.md) is owner-requested opt-in implementation, `agent="terra-implementer"`, never a new default.
- [Astra implementer](agents/astra-implementer.md) handles justified hard implementation, `agent="astra-implementer"`.
- [Astra expert](agents/astra-expert.md) supplies fresh, focused read-only advice on material questions or blockers. It never edits, publishes, delegates, or reviews its own recommendation.
- [Astra orchestrator](agents/astra-orchestrator.md) supplies explicitly approved whole-phase planning/gate advice beneath an external Codex or Claude controller. It does not itself invoke other roles or publish; this is not a standalone primary route or caller-authentication mechanism.

Named profiles bind their exact model and high effort. Omit model and variant on calls, continuations, and forks; conflicting overrides, rejected selectors, role/model contradictions, or silent fallback stop the phase. Use the configured default only when the owner requests it, omitting agent/model/variant. Missing telemetry alone is not proof of another model.

Implementers edit/test one bounded phase, following **red -> implement -> green -> GPU** when relevant. They never plan, review, delegate, invoke subagents, stage, commit, or publish. Fix ordinary in-scope mistakes automatically. Preserve explicit task allocations, shared limits, consumed history, and deadlines across reloads and review fixes; reload never reopens an exhausted allocation. No default numeric candidate/consultation cap applies. Repeated same-blocker failure returns evidence to the controller for a changed approach or useful help, not blind retries.

## Gates and approval

The external controller owns scope, final gates, independent review, and publication. Planning review is recommended for substantive plans: `grilled-me` for self-authored plans, `plan-audit` for external authorship. Expert preflight is risk-driven. L3 independent finish review remains required: Claude for a Codex controller, Codex for a Claude controller. Neither an OpenCode expert nor an advice author's self-review replaces it; unavailable required review stops the phase. L4 requires explicit owner design direction before implementation.

One approved plan covers its local edits, tests, reviews, and controller commits without per-step prompts, normally in one plan-plus-implementation branch/PR. Continue approved local successors until a manual boundary, changed scope/authority, or real blocker; this supplies no new checkout or publication authority.

Publication requires explicit authority for the named actions. Grouped requests may cover enumerated actions together. An eligible draft may use one request to mark it ready and merge it; merge-only approval never implies readiness. Exact plan-only or genuinely nonoperational documentation-only PRs may combine content and merge approval, explicitly including readiness if needed. Skills, profiles, guidance, policy, mixed, uncertain, or operational content do not qualify for that exception.

Qualified mechanical shared-skill bundles may explicitly include administrative readiness in delivery approval and use one final bundle merge request under `skill-sync`. Codex skill-sync remains `opencode_mcp`-source-local and is never distributed. Source approval never transfers to targets. Preserve target-only drift/deletion, approved variants, and dirty active-worktree protections.

Owner-facing task/plan/PR names suffice; the controller resolves concrete scope, head/base, eligibility, and permitted actions and rechecks required checks/reviews before acting. No SHA-only approval, magic reply, or duplicate disclosure prompt for necessary in-scope material already bound to a configured delegate/reviewer. Host protections remain unchanged. Keep useful actual-state and authority evidence, not a compulsory PR-lifecycle ledger. GitHub routes follow the active runtime skill; workflows requiring GitHub MCP never silently fall back to `gh`.

## Capability and safety

Confirm the expected checkout and relevant capabilities using actual delegate tools. Missing native apply_patch does not prove editing is unavailable; discover permitted alternatives and verify authorized edit results. Do not impose universal marker/hash probes, invent extra writes, install tools, or assume fixed tool names.

Delegated paths stay within the caller's working directory; contained source does not guarantee access to linked/common Git metadata. Access denial stops affected work. Never copy/redirect metadata, widen permissions, or relaunch to evade it. A separately owner-approved independent clone is an option after metadata-access failure, not automatic recovery; preserve original dirty work and plan/P0 provenance and reverify capabilities there.

Wrapper completion, printed JSON, and prose are not evidence of native tool execution. Distinguish observed invocation failure, reported absence, denial, and unknown cause; unknown stays unknown. Wait required jobs and commands to terminal and retrieve output before success, commits, publication, or advancement. Never duplicate live/unknown work or treat cancellation requests as terminal proof. At owner stop or declared deadline, report each unresolved resource and retrieval state honestly as pending.

Role prompts and task scope remain authoritative. Permissions are guardrails, not authorization or a complete sandbox. Preserve unrelated artifacts and checkout boundaries. Never read, print, copy, expose, or request credentials or secrets.
