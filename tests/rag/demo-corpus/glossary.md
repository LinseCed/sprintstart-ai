# Glossary

## Diff Score

The diff score is an automated metric Codebridge computes for every pull request. It
combines lines changed, number of files touched, and the cyclomatic complexity delta
of modified functions into a single number between 0 and 100. A higher diff score
indicates a riskier or more complex change and triggers additional reviewer requirements.

## Review Checklist

A review checklist is a repository-level template that lists the criteria a reviewer
must explicitly confirm before approving a PR. Checklists are defined in
`.codebridge/checklist.yml` and are rendered inside the Codebridge review UI.

## Async Thread

An async thread is a comment thread attached to a specific line or file in a PR.
Unlike GitHub inline comments, async threads have an explicit "resolved" state that
must be set by the thread author before the PR can be merged. This prevents comments
from being lost or silently ignored.

## Audit Log

The audit log is an immutable, append-only record of all review actions taken in
Codebridge: approvals, change requests, merges, and checklist completions. It is
exported as JSON and is used by enterprise customers for compliance reporting (SOC 2,
ISO 27001).

## Feature Flag

A feature flag is a runtime configuration switch that enables or disables a code path
without a deployment. Codebridge uses LaunchDarkly for flag management. New features
that require more than one PR to ship must be wrapped in a feature flag and enabled
only after all PRs land.
