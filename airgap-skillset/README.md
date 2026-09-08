# Airgap Skillset (single-model / closed-network)

Replacement skill package for a closed network with exactly one backend
model (e.g. DeepSeek V4 Flash) reachable through whichever harness is in use
there (Claude Code, OpenCode, or Copilot — all support sub-agents). Drafted
in the open-network `claude_mcp` repo for review, to be carried across by
hand and dropped in as `.claude/skills/` (or the equivalent path) on the
closed side.

## What changed vs. the open-network skillset

- **Removed**: `opencode-delegate`, `codex-delegate`. Both exist to route
  work across *different* models (Luna / Sol-expert / Codex Terra). With one
  model available, that routing has nothing to select between, and the
  route-evidence bookkeeping those skills required (verifying a named agent
  actually bound to the model it claimed) is meaningless when every role
  resolves to the same weights.
- **Rewritten**: `phase-gated-implementation` — dropped multi-agent routing
  and route-evidence sections; kept L1-L4 risk gating, red/green gates,
  plan-first PR workflow, attempt caps, and the phase report. Ceremony now
  scales by difficulty tier instead of by backend selection. Restructured
  again on 2026-08-18 to track an open-network refactor — see "Synced
  2026-08-18" below.
- **Added**: `isolated-review` — the practical replacement for the old
  "fresh-context Codex Terra review" trigger. Since the reviewer is the same
  model as the implementer, independence comes from information isolation
  (fresh context, diff + gate evidence only, no shared reasoning/history)
  and a structured adversarial checklist, not from different weights. It is
  weaker than a genuinely different model and says so in its own Limits
  section.
- **Unchanged**: `plan-audit`, `grilled-me`, `karpathy-best-practices`,
  `handoff`. All four are model-agnostic and carry over as-is.
- **Added**: `autonomous-goal-loop` — unattended long-running execution
  toward a machine-checked goal, ported from `../lab`. Unlike every other
  skill in this package, it is not documentation-only: it ships a real,
  tested Python reference implementation
  (`autonomous-goal-loop/reference-impl/`) alongside its `SKILL.md` and
  references. See "Synced 2026-09-04" below and that skill's own
  `PORTING.md` for the exact source-to-target mapping.
- **Included as-is**: `airgap-export` / `airgap-triage`, copied verbatim
  from the closed-network originals (no content changes made). They are the
  genuine independent-model channel (a real external LLM, not the local
  model reviewing itself) and are what `phase-gated-implementation`'s
  escalation path calls out by name. See "Open question" below before
  treating them as a drop-in fit for that escalation path.

## Escalation model

Routine review: `isolated-review` (same model, isolated context, cheap,
automatic per phase).

Hard cases: `airgap-export` (different model, human-carried round trip,
expensive, reserved for a repeated blocker, a high-risk irreversible step,
or an isolated-review blocker the controller can't resolve locally). This
mirrors where Sol-expert/Codex Terra sat in the original routing, but the
independence is now real (external model) instead of assumed (same model,
different role prompt).

## Resolved — airgap-export widened for design-review

The CLASS/mode gap above was resolved by widening `airgap-export` rather
than narrowing the escalation trigger:

- Added `CLASS: design-review` to the OUT format in both `airgap-export`
  and `airgap-triage`, with field-meaning notes (OBSERVED = approach as
  designed, EXPECTED = property/risk it must satisfy, TRIED = alternatives
  already considered) and a worked example, alongside the existing
  bug-diagnosis fields. `phase-gated-implementation`'s second escalation
  trigger now names `CLASS: design-review` explicitly.
- Hard rule 3 changed from an unconditional "algorithm bodies never cross"
  to domain-conditional: most algorithms are not the secret, so masking
  names (per the existing Naming rule) is the default even for real
  control flow and math. The heavier dummy-payload replacement is reserved
  for a short, site-extendable sensitive-domain list (lithography, OPC,
  EUV process algorithms) documented in a new "Domain sensitivity"
  section, with an explicit "ask a human when it's ambiguous" rule rather
  than a silent default either way.
- The leak-scan section and its hard rule were left untouched, per
  instruction. The sensitive-domain list is meant to also feed `terms.txt`
  so leak-scan remains the backstop if drafting misses a case.

Not addressed (flagged, not blocking): the CUDA/HPC-specific repro rules
still lack a non-GPU worked example; the generic "follow the source, strip
that language's comment syntax" fallback already covers it, so this is a
documentation gap, not a functional one.

## Synced 2026-08-18 — open-network phase-gate refactor

The open-network `.claude/skills` went through a phase-gate refactor
(`22fb4f9`, `3dadd8a`, `b79b510`, `e26223b`, `41ff466`, and the compact-skill
sync in `e8c10cc`) after this package was first drafted. Ported the parts
that are model-agnostic; left out the parts tied to open-network-only MCP
infrastructure:

- **karpathy-best-practices**: resynced to the compact upstream version
  verbatim. Not a pure format change: the new "Verify" section adds a
  requirement the old "Goal-Driven Execution" section didn't have —
  inspect the final diff for scope/simplicity/unintended API or behavior
  changes and stale artifacts, and don't claim completion without green
  evidence or an explicit reported limitation. Flagged here because an
  earlier version of this note called the sync "no semantic change," which
  was inaccurate.
- **phase-gated-implementation**: split from one file into a short core
  plus `references/publication.md` and `references/escalation.md`, loaded
  lazily right before their transition instead of upfront — same pattern
  the open-network version adopted, sized down to two references since
  there's no Luna/Sol-expert/Terra routing to document separately.
  - Adopted **delivery topology** as-is (one PR per releasable/revertible
    deliverable by default, split at named risk boundaries, bundle
    compatible phases under one initiative plan) — this is publication
    hygiene, unrelated to how many models are involved.
  - Adopted the tightened **L1 fast path** exclusion list as-is (no
    policy/architecture/API/security/release-config/dependency/migration/
    generated-output/workflow/approval-behavior/operational-doc work, no
    new cross-repo rollout without an already-merged source + exact
    manifest) — same reasoning, model-agnostic.
  - **Not adopted, by instruction**: the open-network "Usage correlation"
    section (`usage_mcp` `workflow_id`/`window_id` binding tied to a Codex
    controller's `context_window.window_id`). The closed network runs a
    local model with no meaningful token/budget constraint, so no
    usage-tracking section, gate, or phase-report field was added — this
    package already had none before the sync and stays that way
    deliberately, not by omission.
  - **Generalized, by instruction**: the open-network "Memory is
    discovery-only" invariant (tied to a `memory-continuity` skill backed
    by an MCP memory service) was ported with the MCP dependency stripped
    out. The rule keeps its shape — a local continuity note is a discovery
    lead, never approval evidence — but assumes no memory MCP is available;
    it only asks the controller to check current Git/plan/HANDOFF state and
    treat any local note (e.g. `HANDOFF.md`) the same way `handoff` already
    produces one.
  - `codex-delegate` on the open-network side lost its EXECUTE mode
    (DISCUSS/read-only only now). Not directly applicable — `codex-delegate`
    isn't in this package — but it confirms the same direction this
    package already took with `airgap-export`'s `design-review` addition:
    review/opinion channels stay read-only, never implementers.
- `opencode-delegate` changed on the open-network side too, but it isn't
  part of this package, so no action.

## Review follow-up (post 2026-08-18 sync)

Three findings against the router/references restructure:

1. The karpathy "no semantic change" claim was wrong — fixed above.
2. The pre-restructure draft had an explicit numbered `Standard loop`
   (8 steps); the router+Invariants restructure covers the same ground as
   unordered invariants, which reads worse as a sequential procedure. Added
   a `## Sequence` section back to the core, pointing each step at the
   reference with its detail instead of inlining it. Whether this actually
   keeps a single-model controller from dropping a step still needs
   real-usage validation — the section is a mitigation, not a proof it
   works.
   - Follow-up: the first version of that `Sequence` dropped the
     pre-implementation owner-approval gate — it first pointed at
     `publication.md` at step 5 (gate/publish), so a controller following
     it literally would start implementing before ever opening the file
     that defines plan approval. The v1 `Standard loop` carried that
     requirement inline in its step 1 and it was lost in the move. Step 1
     now reads "read `references/publication.md`; confirm plan/audit
     currency, plan-PR approval SHA, ..." — worth recording that the
     step-drop mitigation itself shipped with exactly the step-drop it
     exists to prevent.
3. Lazy reference loading is self-enforced only: the core says "read this
   before X" with no mechanical check, so committing without having read
   `publication.md` is a live failure mode (inherited from the open-network
   design unchanged, not introduced here). Added an explicit read-check
   line to the Gate section as a cheap mitigation — it still isn't
   enforcement, since nothing here can verify a reference was actually
   read versus assumed.

## Synced 2026-08-20 — compactification refactor + commit-first publication

Upstream (`41b60bc`) split several skills into core+references via a new
`instruction-compactification` skill, reworked owner-approval semantics,
and changed OpenCode routing. Ported what applies:

- **plan-audit**: adopted upstream's core/references split verbatim (core
  115 lines + `references/audit-procedure.md` + `references/output-format.md`).
  Verified as pure relocation before copying — the pre-split upstream file
  was byte-identical to this package's copy, and the extracted sections
  match the originals with only `###`→`##` heading promotion.
- **Commit-first publication** (closed-network-specific, not an upstream
  port): the closed network has its own GitHub, but work lands as commits
  and is batched into a PR only on request. `references/publication.md`
  now defaults to commit-based publication and states that no PR is
  opened, updated, or merged unless the user explicitly asks. Plan
  approval moved from "plan-only draft PR" to a committed plan artifact
  with an SHA-tied approval. Delivery topology is retained — it still
  decides which phases may be batched into one PR later. A new "When a PR
  is explicitly requested" section covers that path. Core invariants,
  `Sequence` step 1, and the phase report were updated to match.
- **Per-action owner approval**: adopted upstream's rule in generalized
  form — `approved` covers exactly the one concrete publication action
  named immediately before it and expires on any target/scope/head/action
  change. Upstream's plan-only-PR merge exception was not ported, since
  there is no plan-only PR in the commit-first model.
- **Not adopted**: `instruction-compactification` (skillset maintenance
  happens on the open-network side; the closed network consumes the
  result), the OpenCode routing change and cwd/worktree delegation
  boundary (no OpenCode delegation route exists here), and
  `usage-correlation` changes (no usage tracking, per earlier decision).

## Synced 2026-08-27 — delivery-draft publication, ticket-first commits, activity-based commit granularity

Owner-requested changes to how this package's commit-first publication
interacts with the closed side's actual (human, batched) submission
process, driven by three practical problems on that side: PRs are never
opened by the tool, they're collected by a person at the end; per-phase
commits pile up faster than a human wants to batch; and work is tracked
by Jira ticket, not by commit content alone.

- **Real-PR path fully removed, not just gated behind a request.** The
  prior "When a PR is explicitly requested" section (`gh pr create` etc.)
  is gone with no exception. In its place, `publication.md` now has a
  "Delivery draft (markdown, no real PR)" section: at initiative
  completion, or on explicit request for a commit range, write a markdown
  PR description to `delivery/<TICKET-ID>-<YYYYMMDD>-<topic-slug>/PR.md`
  at the repo root. Owner-supplied real-test evidence (not unit tests)
  goes alongside it under `.../tests/`. `delivery/` is never staged or
  committed — it's local working material the human uses to batch the
  actual submission later, out of band. Owner approval of the draft's
  content is terminal; the agent takes no further action on it.
- **Commit granularity changed from one-per-phase to activity-based.**
  Per-phase commits are consolidated (squash/amend, local-unpushed-only)
  toward one impl commit, one test commit, and one fix commit per review
  round (`fix-after-review-1`, `fix-after-review-2`, ...) per initiative —
  see the new "Commit granularity" section in `publication.md`. This
  surfaced a correctness gap in the existing Gate logic: its
  auto-continuation count assumed one `Phase-gate:` trailer per commit.
  Fixed by making a consolidated commit carry multiple trailer lines (one
  per folded phase, in order) and having the Gate scan count trailer
  occurrences rather than commits — same safety property, new
  granularity.
- **Ticket-ID-first requirement, new.** Every commit message and the
  delivery draft's title must start with the Jira ticket ID. The
  controller asks the owner for it at preflight if not already given;
  a missing ticket ID is now a stop condition alongside a missing plan or
  failed gate.

Not touched: `plan-audit`, `grilled-me`, `karpathy-best-practices`,
`handoff`, `airgap-export`, `airgap-triage`, `escalation.md`, and the
plan-artifact approval flow itself (beyond picking up the same
ticket-ID-first commit-message prefix as every other commit). No
`.gitignore` change is prescribed — keeping `delivery/` untracked is
stated as the hard rule; adding a `.gitignore` entry for it is left as an
optional recommendation, not an enforced step.

A same-day audit follow-up restored one safety line the removed real-PR
section had carried ("a refusal by the host stops the action rather than
authorizing a different form of it", now attached to pushes) and fixed a
dangling "(see Stop below)" cross-reference.

## Synced 2026-08-28 — plan artifact moved out of the repository

The 2026-08-27 change left the plan-approval flow untouched, still
requiring the plan to be committed as a repository artifact with a
SHA-tied approval. That conflicts with the closed side's git policy: plan
files are not allowed in the repository, and a plan commit would pollute
the exact history a human later batches into a manual submission.

- The plan artifact now lives at
  `delivery/<TICKET-ID>-<YYYYMMDD>-<topic-slug>/plan.md`, in the same
  untracked ticket folder as `PR.md` and `tests/`, and is never staged or
  committed.
- With no commit SHA available, owner approval is tied to the plan file's
  content hash (`sha256sum`) instead. Material changes change the hash and
  invalidate the approval, preserving the exact-version binding the SHA
  used to provide. Approval evidence lives in an appended note naming the
  approved pre-note hash, or in the phase report.
- Required plan content, the L1 fast-path exemption, and every other gate
  are unchanged — only the storage location and the approval-binding
  mechanism moved.

Known trade-off, accepted: an untracked plan has no git history, so a
tampered or accidentally edited plan is only detectable by hash mismatch
against the recorded approval, not by `git log`. The hash binding is the
compensating control.

## Synced 2026-09-04 — added autonomous-goal-loop (doc + reference implementation)

Ported `autonomous-goal-loop` from `../lab` — unattended, long-running
execution toward a machine-checked goal, gated by containment, a lease,
and a ledger rather than a per-phase owner approval. Full source manifest,
pinned revision, and every documented deviation from the upstream bytes
are in `autonomous-goal-loop/PORTING.md`; summary here.

Per CLAUDE.md's "justified difficult preflight," ran one bounded
`opencode-delegate` → `sol-expert` consultation on the draft port plan
before implementing (job `890f93363dd342939f4a92604aefb69f`). It returned
STOP with four findings; three were adopted (below) and one — applying
this branch's ticket-ID/delivery-draft lifecycle to the port work itself —
was declined as a category error: that lifecycle governs work done *using*
the ported skillset on the real closed network, not this repo's own
meta-work authoring it, which stays consistent with the three prior
commits on this branch.

- **Router adapted, same pattern as `phase-gated-implementation`**: the
  `opencode-delegate` row is gone (single model — direct implementation
  under the ceremony tiers, nothing to route to); `memory-continuity`
  redirects to `handoff` plus `phase-gated-implementation`'s existing
  "Local continuity is discovery-only" invariant, since that row was
  always about the *controlling agent's own session* resuming, not the
  loop's own resume procedure (which is disk-only and self-contained
  regardless of any memory skill). The hardcoded `claude -p` headless-
  invocation example was reworded harness-neutral, matching
  `phase-gated-implementation`'s "whichever harness is active" framing.
  Both source-to-target router rows are written explicitly in the ported
  `SKILL.md`, not left as silent deletions, per the Sol-expert finding
  that flagged the original draft's bare row removal.
- **Reference implementation is real, tested code, not just docs** — the
  first skill in this package where that's true. Copied verbatim except
  for a documented, reproducibly-diffable set of deviations: the deployed
  systemd unit ships as `goal-loop-p4.service.example` with placeholder
  paths (never this dev machine's real path — the Sol-expert finding that
  blocked the first draft plan), two structural tests updated to match,
  one lab-specific test removed (`test_skill_mirrors_are_byte_identical`,
  which asserted a `.claude`/`.codex` dual-tree layout airgap-skillset
  doesn't have), and `evidence/bundle1/` kept as a required parser-
  correctness test fixture (`receipt_declaration_test.py` and one
  `bundle_b4_b6_test.py` test open it directly) with its one absolute
  path rewritten to this checkout — `evidence/bundle2/` and `bundle3/`
  stay excluded, since nothing in the kept suite reads them and a
  receipt from one machine is not evidence for another. Full pytest suite
  verified green (237/238 upstream tests — exactly upstream minus the one
  removed mirror test) after every edit, with an allowlisted-diff gate
  confirming the only differences from the pinned upstream revision are
  the ones listed in `PORTING.md`.
- **Ratchet/Search classes port as prose only** — upstream has them
  "specified, not built" too; nothing to implement on either side.

## Synced 2026-09-07 — reviewed the upstream Astra/lifecycle rewrite, narrow escalation tightening only

The open-network `phase-gated-implementation`/`opencode-delegate` skills
were rewritten again upstream (merged 2026-09-05, `agent/phase-gate-lifecycle-astra-plan`
PR #68 in `opencode_mcp`; synced into this repo's own `.claude/skills` on
the still-unmerged `agent/shared-skill-sync-20260907`) since this
package's last sync (2026-08-20). Reviewed the full diff against this
package's phase-gated-implementation; the actual change surface is much
narrower than the upstream diff's size suggests, because most of it is
real-PR/GitHub-lifecycle and multi-model-routing machinery this package
never had in the first place.

Used the Sol-expert preflight pattern again, this time with the new
Astra route itself (`agent="terra", model="openai/gpt-6-astra",
variant="high"`, job `080f1d6a032c4bf9990ff94cd6385b21`) for an
independent read on the classification below, per the user's explicit
request — one of Astra's own three valid triggers. Observability note:
the job showed zero step progress for roughly 17 minutes (no
`step_finish` event at all until completion), unlike every prior Luna/
Sol-expert delegation in this branch's history, which is recorded here as
a possible model-availability/latency characteristic of
`openai/gpt-6-astra` in this environment, not confirmed as a defect.

- **Confirmed not applicable, no port** — the combined draft-plan/
  implementation two-checkpoint PR topology, P0 verification, the exact
  `Approve merging <repository> PR #<N>?` merge prompt, draft/ready
  GitHub states, and the qualified mechanical skill-sync bundle exception
  (all real-PR mechanics; this package permanently removed real-PR
  publication on 2026-08-27/28); Luna `xhigh`/`max` effort-variant
  exceptions and their mandatory-Codex-Terra-review rule, and
  `usage-correlation.md` (both multi-model/usage-tracking machinery this
  package never had); the dual Codex/Claude router-destination columns
  (this package is a single flat tree); the 2-file → 5-file reference
  split (organizational, not a content gap — this package's leaner
  `publication.md`/`escalation.md` split already covers the same ground
  without the PR-mechanics bulk).
- **Confirmed already equivalent** — upstream's rewritten "Memory is
  discovery-only" wording matches this package's existing generalized
  invariant (checks Git/plan/HANDOFF state, treats a local note as a
  discovery lead only, never supplies approval); no change needed, and
  good confirmation the 2026-08-18 generalization was done correctly.
- **Adopted — `airgap-export` escalation tightened**
  (`references/escalation.md`), read as this package's structural analog
  to Astra (a genuinely different model, not just a different role on the
  same model — closer in spirit than equivalent, since it also crosses a
  real trust/transport boundary Astra doesn't): an explicit one-consultation-
  plus-one-follow-up cap that a new case id cannot reset (upstream has this
  for Astra; airgap-export previously only had the informal "round trips
  are manual and expensive"); an explicit statement that returned advice
  is advisory only and cannot authorize implementation, publication, scope
  expansion, a gate waiver, or an extra retry; an explicit statement that
  an export does not satisfy a required `isolated-review` (closes a real
  ambiguity `isolated-review`'s own Limits section only implied from the
  other direction); and a no-persistent-upgrade line so one escalation
  doesn't turn later phases export-by-default. Deliberately did not
  restate what `airgap-export`'s own OUT format already owns (local-attempt
  evidence, de-identification, case logging) — Astra's own review flagged
  the risk of two independently maintained definitions, and checking
  `airgap-export`/`airgap-triage`'s existing text directly confirmed no
  existing cap or authority-scope language to reconcile against, so this
  is a clean addition, not an overlap.
- **Declined — importing "explicit owner request" as a new airgap-export
  trigger.** Upstream lets the owner request Astra unconditionally; adding
  the equivalent for airgap-export would be a real trigger-list expansion
  (a policy decision), not a neutral sync, and nothing in this review
  found evidence it's needed yet.
- **Declined — a Ceremony-level effort-variant exception** analogous to
  Luna `xhigh`/`max`. Reasons: the local harness's own reasoning-effort
  knob (if any) is unverified across the three named harnesses; the
  upstream exception is a bundled package of controls including a
  mandatory independent review that has no true equivalent here
  (same-model isolated-review is explicitly weaker, per its own Limits
  section); ceremony and compute are different axes and conflating them
  risks implying more reasoning earns relaxed ceremony; and raising local
  effort alone is not a materially revised approach, so it could function
  as a disguised extra retry against the attempt cap. Revisit only if a
  concrete harness and an actual stuck Difficult-tier phase motivate it.

## Known limitation

Nothing in this package restores genuine model diversity for *routine*
review — that only happens on an airgap-export round trip. Isolated-review
will not catch a mistake the model makes identically whether it's
implementing or reviewing. Treat its "no blocker" verdict as weaker evidence
than the equivalent verdict was under the multi-model setup, and escalate
more readily than the original routing did.

Separately: the `Sequence` and Gate read-check additions above are text
mitigations for a single-model reliability question that hasn't been
tested against real use, and for an enforcement gap that has no mechanical
fix within a documentation-only skill.
