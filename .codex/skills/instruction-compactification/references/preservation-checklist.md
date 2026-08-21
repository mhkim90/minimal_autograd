# Preservation Checklist

Read this checklist immediately before deciding whether to audit or compact
instruction material. It is a decision gate, not authority to publish or alter
policy.

## Inventory and authority

- [ ] Record the approved phase, owner/controller, exact runtime(s), and exact
      paths in scope.
- [ ] Confirm whether the work is audit-only, compactification, or an explicitly
      approved delivery action. Do not infer publication, synchronization, or
      merge authority.
- [ ] Inventory each source file, its consumers, load mode, frontmatter, direct
      references, links, and known runtime variant.
- [ ] Do not inspect or modify unrelated repositories, worktrees, or artifacts.

## Classification

Classify every in-scope artifact before proposing a move or rewrite:

- [ ] **Root guide** — always-loaded agent guidance that establishes baseline
      behavior, policy, authority, or routing.
- [ ] **Skill** — on-demand instructions selected by a trigger and loaded for a
      bounded task.
- [ ] **Reference** — supporting detail linked by a root guide or skill; it is
      not silently promoted to always-loaded policy.
- [ ] Preserve the class, load boundary, trigger, and discoverability of each
      artifact unless the approved scope explicitly changes them.

## Preservation invariants

- [ ] Meaning and observable behavior remain unchanged: no softened, expanded,
      reordered, or newly implied requirement.
- [ ] Authority, approval, publication, synchronization, deletion, and stop
      boundaries remain explicit and unchanged.
- [ ] Triggers, exclusions, non-goals, required sequencing, gates, retries,
      rollback/hand-off rules, and output/reporting contracts remain covered.
- [ ] Runtime, model/route, permission, safety, and validation constraints are
      retained; concise wording must not turn a convention into authorization.
- [ ] Frontmatter names, descriptions, paths, and loader-visible conventions
      still match the artifact's runtime and directory.
- [ ] Examples and terminology still express the same contract; do not make
      byte-equality or formatting normalization a correctness requirement.

## Relocation map

For every compacted or relocated section, record a map before editing:

| Source section | Destination | Retained entry/link | Preserved contract | Reason |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

- [ ] Every removed source passage has an approved destination or is explicitly
      approved for deletion.
- [ ] The retained entry point states when the relocated detail must be read.
- [ ] A move from always-loaded guidance to an on-demand skill/reference does
      not hide a required invariant, trigger, or stop rule.
- [ ] Do not use compaction to merge root guides, skills, and references merely
      because their text is similar.

## Links and paired runtimes

- [ ] Resolve every relative link from its actual source file; verify anchors,
      filenames, and case.
- [ ] Verify links to references in both Codex and Claude skill trees when both
      runtimes are in scope.
- [ ] Compare paired runtimes for semantic coverage, not byte equality. Keep
      approved differences such as runtime-specific frontmatter, metadata, or
      routing conventions.
- [ ] Do not add Claude agents metadata unless an existing Claude convention and
      approved scope require it; Codex agent metadata follows its local
      convention.
- [ ] Record any intentional variant and its preservation rationale.

## Validation

- [ ] Inspect the final diff and confirm it contains only the approved paths.
- [ ] Run `git diff --check`.
- [ ] Run the runtime skill validator (including `quick_validate.py`) for every
      new or changed skill folder when available; record an unavailable tool as
      a blocker rather than substituting an unapproved check.
- [ ] Check frontmatter, required entry links, reference paths, and loader
      naming manually when a validator does not cover them.
- [ ] Re-read the compacted entry point and relocated reference together; verify
      that a fresh reader can still discover every required invariant.
- [ ] Do not claim semantic preservation from whitespace, line-count, or
      byte-level comparison alone.

## Stop checks

Stop and request direction if any item is true:

- [ ] Scope, authority, artifact class, or runtime pairing is missing or
      ambiguous.
- [ ] A proposed edit changes policy, authority, approval, permissions, model
      routing, or other semantics rather than presentation/load placement.
- [ ] A required rule would become less discoverable, lose its trigger, or move
      without a verified link and relocation map.
- [ ] Codex and Claude variants cannot be reconciled semantically, or a variant
      is unexplained.
- [ ] A link, frontmatter field, validator, or other loader contract fails.
- [ ] The request assumes implicit sync, deletion, publication, commit, merge,
      or byte-equality normalization.
- [ ] Unrelated files, worktrees, or user artifacts would need inspection or
      modification.
