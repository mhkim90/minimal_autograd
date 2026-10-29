---
name: instruction-compactification
description: Audit or compact always-loaded root agent guides and on-demand skills while preserving their meaning, authority, links, and runtime variants. Use when auditing or compacting instruction guides, skills, or their references; this does not authorize policy changes.
---

# Instruction Compactification

Audit or compact instruction material without changing policy, authority,
approval boundaries, or publication state. A requested policy change is a
different task; say so and get explicit approval for it.

1. **Inventory** the approved paths and classify each file as a root guide
   (always loaded), a skill (loaded on trigger), or a reference (loaded on
   demand). Keep each file's class and trigger unless the scope changes
   them.
2. **Preserve** meaning: triggers, exclusions, authority and stop rules,
   gates, ordering, and output contracts. Do not soften, expand, or imply a
   new requirement, and do not move a required rule from an always-loaded
   guide into a place where it would no longer be found.
3. **Relocate deliberately.** For every moved section, record where it went
   and keep a working link from the entry point that says when to read it.
   Do not merge guides, skills, and references just because they look
   alike.
4. **Pair runtimes.** Treat `.claude` and `.codex` copies as one artifact
   with approved differences (frontmatter, `agents/` metadata, controller
   and reviewer names). Compare meaning, not bytes, and explain any
   variant.
5. **Validate**: only approved paths changed, `git diff --check`,
   `quick_validate.py` for each changed skill, links and anchors resolve,
   and a re-read of entry point plus moved detail together. Line counts and
   byte comparison alone do not prove meaning was preserved.

Stop and ask when the scope, authority, or runtime pairing is unclear, an
edit would change semantics, a rule would lose its trigger or link, or a
validator fails. Leave commit, publication, and syncing to explicit
approval.
