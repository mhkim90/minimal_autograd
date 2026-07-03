---
name: handoff
description: Write or update a project handoff document so another Codex agent can continue with fresh context. Use when ending a session, switching context, preserving progress, or when the user asks for a handoff, continuation notes, or HANDOFF.md.
---

# Handoff

Write or update a handoff document so the next agent can continue without reconstructing context.

## Steps

1. Check whether `HANDOFF.md` exists in the project root.
2. If it exists, read it before updating.
3. Create or update `HANDOFF.md` with:
   - `Goal`: what the work is trying to accomplish.
   - `Current Progress`: what has been done so far.
   - `What Worked`: approaches or commands that succeeded.
   - `What Didn't Work`: failed approaches so they are not repeated.
   - `Next Steps`: clear action items for continuation.
4. Tell the user the `HANDOFF.md` path.

## Style

- Keep the handoff factual and concise.
- Include exact file paths and commands when they matter.
- Distinguish completed work from guesses or pending validation.
- Do not include unrelated conversation history.
