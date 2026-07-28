# IntelliHomes GitHub Workflow

## Branching Strategy

- Main contains stable code.
- Feature branches use:

feature/[description]

Example:

feature/rag-document-pipeline

Branches are deleted after merging.

---

## Commit Convention

Use Conventional Commits.

Types:

- feat
- fix
- docs
- refactor
- chore

---

## Pull Request Process

Every feature is developed on its own branch.

Each Pull Request should:

- Link a GitHub Issue.
- Explain what changed.
- Receive at least one review before merging.

Reviews focus on:

- Correctness
- Readability
- Data integrity
- Documentation quality

---

## GitHub Issues

Every feature starts with a GitHub Issue.

Each issue has:

- Title
- Description
- Label
- Assignee

Issues are closed after the related Pull Request is merged.