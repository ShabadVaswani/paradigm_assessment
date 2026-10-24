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

> this is the assessment, download the handbook, analyse how long it is and lmk

(The prompt above came with the take-home instructions PDF attached.)

## What the AI did / proposed

- Found that the earlier notes files (`DECISIONS.md`, `ai_log/sessions/01-setup.md`) had been deleted from the working folder but were still in the repo's single commit. It did not restore or replace them without asking.
- Offered three options: restore the files and log this as session 02, start fresh but keep the old commit in history, or restore the files unchanged.
- After my answer, checked that the repo had no remote and only one commit, then deleted that commit and purged it from the repo, so `main` started with no history.
- Created a new `DECISIONS.md` (template only) and this log.
- Handbook download: the PDF text only showed "(link)", so the AI found the instructions PDF in Downloads and pulled the URL out of it (a Box shared link). It then downloaded the handbook through Box's direct-download URL. The handbook isn't in the repo yet.
- Handbook analysis, using PyMuPDF and tiktoken:
  - It's the PostHog handbook, exported from HTML through headless Chrome on 2026-04-20. The file is 50 MB.
  - It has 1,076 pages, about 317k words, 2.0M characters and roughly 451k tokens. The token count uses the cl100k tokenizer, so Claude's own count will differ somewhat.
  - It contains 254 articles. Each one starts with a title and its source path (`contents/handbook/....md`). The median article is about 920 words and the largest about 7k. The biggest section is growth, with 67 articles.
  - The text extracts cleanly with no OCR. Every page has a footer to strip (date, title, `file://` URL, page N/1076). The PDF has no bookmarks or table of contents.
  - Some names and team links came out blank in the export, e.g. "handled by , , and on the team".
- Pointed out that about 450k tokens is too much to send with every question, so the app will need retrieval. This is not decided yet.

## Accepted / rejected

- I rejected all three options and wrote my own: delete the old commits and start completely fresh.

## What the AI got wrong

- In the first two commits it left a blank line between `AI-assisted: yes` and `Co-Authored-By:`. Git reads trailers only from the last paragraph, so it didn't see `AI-assisted` as a trailer. The AI caught this by checking `git log` for the trailer right after committing, and found it empty. Neither commit had been pushed, so it recreated both with the trailers in one block. Later commits use that format.
- In the analysis it ran Python with `-I`, which ignores `PYTHONIOENCODING`, so printing emoji from the handbook crashed twice on the Windows console encoding. It fixed this with `sys.stdout.reconfigure(encoding="utf-8")`. This didn't affect any results.
