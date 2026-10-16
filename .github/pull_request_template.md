## What and why

<!-- One or two sentences. Link the issue. -->

## Architecture decision (AI_RULES R16)

<!-- Link the accepted ADR in the Wiki, or write "No trigger applies, because …". -->

## How it was tested

<!-- Tests added (they fail without this change). For a user-facing journey: the entry in
sprintstart-frontend/docs/testing/e2e-journeys.md and its spec, or the manual end-to-end
check that was actually performed. -->

## Checklist (AI_RULES.md)

- [ ] **AI-assisted:** an AI agent wrote part of this diff, and I have read every line of it (R23)
- [ ] R3: no new function, file or class over a limit; baseline entries I touched are removed
- [ ] R4–R6: no suppression added, no check loosened, no test skipped or weakened
- [ ] R7–R9: new ids are value types; closed sets are enums/sealed types; no nullable-as-mode
- [ ] R10/R11: each new function has one responsibility; entry points are thin
- [ ] R13/R14: I searched for existing code (where: …); nothing was copied and adapted
- [ ] R15: no swallowed exception
- [ ] R17: no comments narrating history or justifying a rule violation
- [ ] R22: one concern, ≤ 400 changed lines excluding tests and generated code
- [ ] `bash scripts/quality/ratchet.sh` passes locally

<!-- A reviewer other than the author approves. If an AI agent wrote this code, the person who
ran the agent does not count as the reviewer. -->
