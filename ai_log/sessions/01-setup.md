# Session 01: Setup (2026-10-24)

Tool: Claude Code (Opus 5.5) in VS Code.

## Goal

Set up how AI use gets recorded in this project, starting from a clean repo.

## Key prompts

> From this message onwards, keep notes that show how I use AI:
>
> - ai_log/sessions/NN-topic.md — after each session: my goal, my key prompts (verbatim), what you proposed, what I accepted or rejected and why, and anything you got wrong and how it was caught.
> - DECISIONS.md — every design choice: options considered, evidence, what I chose, what you suggested.
> - Commits: small, never squashed; add the trailer "AI-assisted: yes" when you wrote or changed the code.
>
> Write the notes in plain words, record only what actually happened, and remind me to review them at the end of each session.

> delete old commits, start completely fresh

## What the AI did / proposed

- Found that the earlier notes files (`DECISIONS.md`, `ai_log/sessions/01-setup.md`) had been deleted from the working folder but were still in the repo's single commit. It did not restore or replace them without asking.
- Offered three options: restore the files and log this as session 02, start fresh but keep the old commit in history, or restore the files unchanged.
- After my answer, checked that the repo had no remote and only one commit, then deleted that commit and purged it from the repo, so `main` started with no history.
- Created a new `DECISIONS.md` (template only) and this log.

## Accepted / rejected

- I rejected all three options and wrote my own: delete the old commits and start completely fresh.

## What the AI got wrong

- Nothing so far.
