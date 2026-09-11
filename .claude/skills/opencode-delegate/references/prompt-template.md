# Prompt Template

```text
Context: <project + phase>
Task: <exact edit/run loop>
Files/scope: <approved paths or globs>
Red gate: <check + expected failure>
Success criteria: <checks/metrics>
Safety risk: <L1-L4>
Implementation difficulty: <mechanical/economy | standard | difficult>
Routing: <agent="luna" with model/variant omitted | explicit owner exact-model request: agent="terra-implementer" or agent="sol-implementer" with model/variant omitted | named Luna variant="xhigh" or "max" exception | explicit user configured default | sol-expert | fresh agent="astra-expert" with model/variant omitted | terra review | sol>
Expert escalation evidence: <role or none>; reason/question: <...>; why cheaper route is insufficient: <...>; requested/resolved model: <...>; effort: <...>; job/session: <...>; expected evidence: <...>; stop condition: <...>
Effort-variant evidence: <named complex phase; reasoning bottleneck; why clarification/decomposition is insufficient; fixed acceptance gate; relevant failure evidence or bounded Sol-expert recommendation for max; warnings>
Elapsed-time checkpoint / final-synthesis grace (full Sol only) / maximum wait: <phase-defined values>
Constraints:
- one bounded phase/subphase; no commit or edits outside scope
- stop after two same-blocker failures; revise materially before a third attempt
Final response: changed files, decisions, commands, blockers, session count,
retries, requested agent, job ID, session ID, bound/reported model, and route
```
