# Development Process

## Branching Strategy

We use trunk-based development. All engineers branch off `main` and merge back into
`main` via pull requests. Branch names must follow the pattern `<type>/<issue-id>-<slug>`,
for example `feat/CB-142-diff-scoring` or `fix/CB-201-null-reviewer`.

Long-lived feature branches are not allowed. If a feature takes more than three days,
split it into smaller incremental PRs behind a feature flag.

## Pull Request Requirements

Before opening a PR you must:

1. Ensure all CI checks pass (lint, type-check, unit tests, integration tests)
2. Fill in the PR description template (linked in `.github/pull_request_template.md`)
3. Self-review the diff and resolve any TODO comments left in the code

A PR requires **two approvals** to merge — one from a backend engineer and one from
the engineering lead (Maya Patel) for any changes touching the review engine or the
schema. Frontend-only changes need one approval from any engineer.

## Code Review SLA

Reviewers are expected to post an initial response within **one business day**. If a
PR has not received a review after two business days, the author should ping the
`#eng-reviews` Slack channel.

## Merging

We squash-merge all PRs into `main`. The squash commit message must match the
conventional commits format: `feat:`, `fix:`, `chore:`, `docs:`, etc.

Do not merge your own PR. Always wait for approvals even if you have merge permissions.

## Releases

Releases are cut every two weeks on the Friday of the sprint end by Carlos Mendes.
The release tag format is `v<MAJOR>.<MINOR>.<PATCH>` following semantic versioning.
