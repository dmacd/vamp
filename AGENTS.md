# AGENTS.md

This file is for agent workflow and project-maintenance guidance. 

## Document Roles

- `PLAN.md` is the live development roadmap. Update it after each turn that
  changes implementation status, next steps, known gaps, or priorities. Keep it
  concrete and current.
- `DESIGN.md` records durable technical design choices: graph semantics,
  search interfaces, MDL behavior, parity interpretation, loading boundaries,
  and other architecture. It is not a task list.
- `STYLE.md` holds coding style and readability preferences. Keep coding rules
  there instead of duplicating them here.
- User-facing introductory material belongs in `README.md`, `TUTORIAL.md`, or
  notebooks, not in the internal planning/design docs.

## Working Principles

- This is greenfield, research development. Do not preserve compatibility 
  aliases or legacy codepaths for surfaces we control. Refactor all uses and remove stale paths.

## Test Discipline


## Git And Workspace

- The worktree may be dirty with unrelated user changes. Do not revert or stage
  unrelated files.
- Commit only explicitly intended files.
- Avoid destructive git commands unless the user explicitly requests them.
- The github CLI is NEVER logged in inside your sandbox, always check outside sandox before complaining

## GPUs

- There is a GPU on this machine, but requires you to run outside your 
  sandbox to access it. Please do so when appropriate.
