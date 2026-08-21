# Prompt Template

```text
Context: <project + phase>
Task: <exact edit/run loop>
Files/scope: <approved paths or globs>
Red gate: <check + expected failure>
Success criteria: <checks/metrics>
Safety risk: <L1-L4>
Implementation difficulty: <mechanical/economy | standard | difficult>
Routing: <agent="luna" | explicit user configured default | sol-expert | terra | sol>
Elapsed-time checkpoint / final-synthesis grace (full Sol only) / maximum wait: <phase-defined values>
Constraints:
- one bounded phase/subphase; no commit or edits outside scope
- stop after two same-blocker failures; revise materially before a third attempt
Final response: changed files, decisions, commands, blockers, session count,
retries, requested agent, job ID, session ID, bound/reported model, and route
```
