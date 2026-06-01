# Welcome to Codebridge

Codebridge is a collaborative code review platform built for distributed engineering
teams. It integrates with Git providers (GitHub, GitLab, Bitbucket) and adds structured
review workflows, automated quality gates, and an audit trail on top of standard pull
requests.

## Problem We Solve

Traditional pull request tools treat code review as a simple approve-or-reject workflow.
Codebridge adds structured checklists, async discussion threads, and AI-assisted diff
summaries so that reviewers can stay in context across multiple time zones without losing
track of open decisions.

## Core Features

- **Structured review checklists**: teams define per-repository templates that reviewers
  must complete before a PR can be merged
- **Diff scoring**: every PR receives an automated complexity and risk score based on
  lines changed, files touched, and cyclomatic complexity delta
- **Async threads**: review comments are threaded and resolved explicitly, preventing
  comments from being silently ignored
- **Audit log**: all review decisions, approvals, and merges are recorded with timestamps
  and reviewer identity for compliance reporting

## Deployment

Codebridge is offered as a cloud SaaS product and as a self-hosted Docker image.
The self-hosted edition requires a PostgreSQL database (version 14 or later) and a
Redis instance for the job queue.
