# Audit Procedure

Apply the checks that match the artifact; mark the rest not applicable
rather than forcing every plan through a code-refactor template.

## 1. State What the Plan Is Trying to Do

Summarize intended outcome, affected system and users, phases, explicit
non-goals, and success and rollback conditions. Record any that are absent
before inferring intent.

## 2. Check Scope and Non-Goals

The plan should separate required from follow-up work, name externally
visible behavior that must not change, identify what may change, and
exclude tempting adjacent work. Flag a boundary such as `src/**` when a
narrower vertical slice is possible: drift must stay detectable.

## 3. Check Contracts

For code and data interfaces: memory and element order, mutability,
aliasing and view-vs-copy, ownership and lifetime, concurrency, error and
partial-failure behavior, compatibility with callers and serialized data.
For configuration, docs, migrations, operations, or experiments: source of
truth, idempotency, rollout order, rollback, failure visibility,
environment and permission assumptions, and the baseline for any
comparison.

If the plan freezes existing behavior with characterization tests,
distinguish an intentional guarantee from a known bug preserved only for
migration safety.

## 4. Check Slicing and Sequencing

Prefer vertical phases that leave the system coherent and testable. For
each phase: does it produce a validated result; are dependencies ready;
can it stop or roll back without corrupting later work; if two
implementations coexist, is the window bounded and one side a thin
adapter; does a later phase need evidence an earlier one destroys?
Identify the first phase at which each problem becomes blocking.

## 5. Map Claims to Enforcement

For every important claim, record what enforces it: an executable test or
command; build, type, schema, permission, or configuration enforcement; a
review-only convention; manual observation; or nothing. Documentation is
not executable enforcement, and a guardrail is not a sandbox. Call out
claims whose stated confidence exceeds their enforcement.

## 6. Check Gates and Exit Criteria

Each phase should declare a red gate or pre-change observation when
meaningful (with why it is the right signal), a green validation,
falsifiable exit criteria, stop conditions, rollback or handoff after
failure, and a meaningful progress/stop condition for retry loops. Respect
any explicit task-specific bounds. Do not accept "tests pass",
"works", or "performance improves" without the command, workload,
baseline, or threshold.

## 7. Check Downstream Coupling

Trace important changes to callers, adapters, tests and fixtures, docs,
build/packaging/CI, and monitoring or operational procedures. Report hidden
consumers that would otherwise surface only during implementation.

## 8. Check Decision Provenance

Separate owner-approved decisions, repository-backed facts, author
assumptions, audit recommendations, and decisions still needing a human.
Do not turn a recommendation into an approved requirement through
confident wording.
