---
name: plan-audit
description: Audit an externally authored implementation plan before execution. Use when a user supplies a plan, design document, issue, PR description, decision record, migration outline, or checklist and Codex or Claude must determine whether it is safe and actionable enough for phase-gated implementation. Do not use for a plan drafted by the active agent; use grilled-me for that.
---

# Plan Audit

Audit an externally authored plan before implementation. Find the smallest
repairs needed to make the plan executable, and produce a structured handoff
for phase-gated implementation.

This is a read-only review skill. Do not edit the plan, approve proposed
boundaries, begin implementation, or silently fill consequential gaps.

## Relationship to Other Skills

- Use `plan-audit` for a plan or execution artifact authored outside the active
  agent.
- Use `grilled-me` to stress-test a plan drafted by the active agent before it
  is presented.
- Use `phase-gated-implementation` only after blocking audit findings are
  resolved and the user has approved any required phase boundaries and gates.

Do not combine these skills into one loop. A plan audit is normally a one-time
preflight for a specific artifact revision, not a review repeated at every
implementation phase.

## Authority Boundary

The audit reports evidence and proposes repairs. It does not authorize them.

- Candidate scope globs are drafts until explicitly approved by the user or
  repository owner.
- Candidate gates, sequencing changes, and risk levels are also drafts.
- Unapproved audit output must not enable automatic phase continuation.
- If the plan lacks enough information to define a narrow boundary or
  falsifiable gate, report the gap instead of inventing a broad substitute.
- Do not modify the audited artifact unless the user separately requests an
  edit.

## Establish the Audited Revision

Before reviewing, identify:

- artifact title and path, URL, issue, PR, or conversation reference;
- revision, commit, blob, version, or other stable identifier when available;
- how the content was obtained;
- related artifacts that are authoritative for scope or constraints.

If no stable revision exists, state that the audit applies only to the content
visible in the current conversation.

Treat the audit as stale when a later revision materially changes scope,
contracts, sequencing, acceptance criteria, risk, rollback, or implementation
strategy. Editorial changes alone do not require a new audit.

## Audit Procedure

### 1. State What the Plan Is Trying to Do

Summarize:

- intended outcome;
- affected system and users;
- proposed phases or major steps;
- explicit non-goals;
- success and rollback conditions.

If any of these are absent, record that before inferring intent.

### 2. Check Scope and Non-Goals

Verify that the plan:

- separates required work from follow-up work;
- names externally visible behavior that must not change;
- identifies files, components, interfaces, or systems that may change;
- excludes tempting adjacent work;
- avoids phase boundaries so broad that drift cannot be detected.

Flag a phase that requires a boundary such as `src/**` when a narrower
vertical slice should be possible.

### 3. Check Contracts

Apply the checks that match the artifact. Mark irrelevant checks as not
applicable rather than forcing every plan through a code-refactor template.

For APIs, data structures, kernels, storage, views, and low-level code, check:

- memory and element order;
- mutability and observable state;
- aliasing and view-versus-copy behavior;
- ownership and lifetime;
- synchronization and concurrency assumptions;
- error and partial-failure behavior;
- compatibility with current callers and serialized data.

For configuration, documentation, migrations, operations, or experiments,
check:

- inputs, outputs, and source of truth;
- idempotency and repeatability;
- compatibility and rollout ordering;
- rollback and recovery;
- failure visibility and observability;
- environment and permission assumptions;
- measurable comparison or baseline requirements.

If the plan freezes existing behavior with characterization tests, distinguish
an intentional guarantee from a known bug preserved only for migration safety.

### 4. Check Slicing and Sequencing

Prefer vertical phases that leave the system coherent and testable. For each
phase, ask:

- Does it produce an observable, validated result?
- Are its dependencies already available?
- Can it be rolled back or stopped without corrupting later work?
- Does it require two implementations to coexist?
- If so, is the compatibility window bounded and is one side a thin adapter?
- Does a later phase depend on evidence that an earlier phase deletes or makes
  impossible to collect?

Identify the first phase at which each sequencing problem becomes blocking.

### 5. Map Claims to Enforcement

For every important claim, determine whether it is enforced by:

- an executable test or validation command;
- build, type, schema, permission, or configuration enforcement;
- a review-only convention;
- manual observation;
- nothing.

Documentation wording is not executable enforcement. A guardrail is not a
sandbox. Call out claims whose stated confidence exceeds their enforcement.

### 6. Check Gates and Exit Criteria

Each phase should declare:

- a red gate or pre-change observation when meaningful;
- the expected failure or current behavior and why it is the right signal;
- a green validation command or inspection;
- falsifiable exit criteria;
- stop conditions;
- rollback or handoff behavior after failure;
- an attempt or iteration bound for retry loops.

Do not accept “tests pass,” “works,” or “performance improves” without naming
the relevant command, workload, baseline, or threshold.

### 7. Check Downstream Coupling

Trace important changes to:

- callers and consumers;
- adapters and compatibility layers;
- tests and fixtures;
- documentation and examples;
- build, packaging, deployment, and CI;
- monitoring, benchmarks, or operational procedures.

Report hidden consumers or coupled follow-up work that would otherwise appear
only during implementation.

### 8. Check Decision Provenance

Separate:

- decisions already approved by the owner;
- facts supported by repository evidence;
- assumptions made by the plan author;
- recommendations introduced by the audit;
- decisions that still require a human.

Do not convert a recommendation into an approved requirement through confident
wording.

## Severity and Disposition

Order findings by severity:

- **Critical**: implementation would be unsafe, destructive, or directed at the
  wrong outcome.
- **Major**: a phase cannot be executed or verified as written.
- **Moderate**: the plan is executable only with a manual gate, narrower scope,
  or clarified contract.
- **Minor**: useful precision or maintainability improvement that can be
  handled later.

For each finding include:

- evidence or exact plan location;
- what breaks and when;
- affected phase;
- smallest adequate repair;
- disposition: blocking before a named phase, manual-gate required, or
  non-blocking follow-up.

Praise only load-bearing strengths: constraints or decisions that materially
reduce risk.

## Verdict

Use exactly one:

- `ready`: no blocking findings; proposed boundaries and gates still require
  approval if the plan did not already contain approved ones.
- `ready-with-manual-gates`: execution may begin, but named phases must stop for
  a human decision or because a predicate is not mechanically decidable.
- `blocked`: implementation must not begin, or must stop before the named
  phase, until blocking findings are resolved.

## Output Format

### 1. Artifact

- identity:
- audited revision:
- content source:
- related authoritative artifacts:
- audit freshness caveat:

### 2. What the Plan Does

Concise summary of outcome, phases, success condition, and non-goals.

### 3. Load-Bearing Strengths

Only strengths that materially improve executability or safety.

### 4. Findings

List findings in severity order. For each:

- severity:
- evidence:
- affected phase:
- failure mode:
- smallest repair:
- disposition:

### 5. Sequencing and Contract Repairs

Show the minimal corrected ordering and unresolved contract decisions. Do not
rewrite the whole plan.

### 6. Phase-Gate Handoff

#### Audit verdict

`ready`, `ready-with-manual-gates`, or `blocked`

#### Global controls

- approved non-goals already present in the plan:
- unresolved human decisions:
- characterized-but-not-guaranteed behavior:
- enforcement gaps:
- stale-audit triggers specific to this plan:

#### Per-phase handoff

Repeat for each phase:

- phase identifier and title:
- dependencies and preconditions:
- candidate scope globs: `<draft / already approved in plan / unavailable>`
- red gate and expected right-reason failure:
- green validation and exit criteria:
- candidate initial risk level and evidence:
- stop rules and attempt bound:
- downstream consumers:
- blocking findings before this phase:
- required manual decisions:

### 7. Owner Decisions Required

List only unresolved decisions that materially affect scope, contracts,
sequencing, gates, or safety.

End by stating that candidate boundaries and audit recommendations remain
unapproved unless the owner explicitly approves them.

## Stop Rules

Stop and request clarification rather than guessing when:

- the artifact or authoritative revision cannot be identified;
- two sources of truth conflict;
- the intended outcome or externally visible contract is ambiguous;
- a destructive or irreversible step lacks an approved rollback;
- scope cannot be bounded narrowly enough to detect drift;
- a phase gate cannot distinguish the intended change from a wrong-reason
  result;
- the plan depends on unavailable evidence, credentials, tooling, or human
  authority.
