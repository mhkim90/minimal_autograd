---
name: memory-continuity
description: Safely resume or hand off phased work with compact local continuity records as discovery leads, while current Git, PR, plan, handoff, and test evidence remain authoritative.
---

# Memory Continuity

Use this skill when resuming or handing off a phased task that may benefit from compact local continuity records. Use memory only as discovery leads; it never substitutes for current repository or controller evidence.

## Evidence and resume workflow

Evidence hierarchy, in this exact order: current Git/PR/test evidence first; then verified plan/HANDOFF; then memory records. Memory cannot grant approval, override a red/green gate, replace current route evidence, or decide quality/routing/publication.

Begin with current evidence checks: inspect the working tree and relevant Git/PR/test state, then verify the applicable plan or HANDOFF. If continuity is still needed and the MCP is configured, retrieve progressively: call `memory_search` for a small index, choose one relevant result, call `memory_timeline`, then call `memory_get` only for selected records. Verify every actionable memory claim against current evidence.

Build a resume capsule with: objective; approved scope; phase/gate; current evidence; active sessions; blocker/recovery; next action and required approval.

## Recording and degradation

Record explicitly only at a manual gate, blocker, attempted recovery, or requested handoff, and only with controller authorization. Use compact redacted structured data. Never auto-capture prompts, tool output, raw commands, or model reasoning.

If the MCP is missing, unconfigured, or fails, or if a record is stale, emit a non-blocking observability warning and continue from current evidence. Never fabricate memory data.

When transfer is needed, the calling agent must create or update `HANDOFF.md` only in the consumer repository, using its normal file tooling. This MCP itself never creates or edits consumer repository files. Skill distribution or synchronization is outside this local service workflow.

Do not use a failure memory record or resume lead to bypass a required current Git, PR, or test gate.
