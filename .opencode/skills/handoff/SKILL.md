# Handoff Document Tool

Creates or updates `HANDOFF.md` to preserve context for the next agent starting with fresh state.

## Process

1. Check for existing `HANDOFF.md` to avoid losing prior work
2. Read any existing file before making updates
3. Write five sections:
   - **Goal**: What the user is trying to accomplish
   - **Completed**: Work done so far
   - **What worked**: Successful strategies and approaches
   - **What failed**: Approaches tried and abandoned (so they aren't repeated)
   - **Next actions**: Clear, concrete next steps

## Output

Saves as `HANDOFF.md` in the project root. Provide the file path so the next agent can load it immediately on start.

## Notes

- Keep each section short and scannable — this file is read cold.
- "What failed" is as important as "What worked" — it prevents repeated mistakes.
- Next actions should be concrete enough to act on without any conversation history.
