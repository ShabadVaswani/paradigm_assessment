# Session 01: Setup and handbook exploration (2026-10-24)

Tool: Claude Code (Opus 5.5) in VS Code. One subagent on Claude Sonnet reviewed the images.

## Goal

Set up how AI use gets recorded in this project, starting from a clean repo. Then get the handbook and understand its size and structure well enough to decide how to split it for retrieval.

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

> 450k tokens can easily fit inside commonly available models,I think it would be a good idea to have evals later on, even though its slower , we can compare our rag pipeline with just putting the data into the context, it could be our baseline, but for now lets work on the retrieval first,
>
> yes we should remove the footer,
>
> it sounds like handbook is just collection of articles, correct me if I am wrong
>
> can you tell me which pages have blank names/team links?
>
> give me distribution in tokens of atricles/headings? then we can decide if we want to do with article based splitting or something else
>
> also use a subagent to check if any of the 78 images contain any information that people may ask about
>
> do pdf exploration inside a notebook so we can keep track of it and its saved for the future
>
> Don't ask me to review the notes; I'll tell you if something is wrong. Record decisions and session notes on your own, without being asked.

## What the AI did / proposed

### Setup

- Found that the earlier notes files (`DECISIONS.md`, `ai_log/sessions/01-setup.md`) had been deleted from the working folder but were still in the repo's single commit. It did not restore or replace them without asking.
- Offered three options: restore the files and log this as session 02, start fresh but keep the old commit in history, or restore the files unchanged.
- After my answer, checked that the repo had no remote and only one commit, then deleted that commit and purged it from the repo, so `main` started with no history.
- Created a new `DECISIONS.md` (template only) and this log.

### First look at the handbook

- Handbook download: the PDF text only showed "(link)", so the AI found the instructions PDF in Downloads and pulled the URL out of it (a Box shared link). It then downloaded the handbook through Box's direct-download URL.
- First analysis:
  - It's the PostHog handbook, exported from HTML through headless Chrome on 2026-04-20.
  - It has 1,076 pages, about 317k words and roughly 451k tokens (cl100k) including the page header and footer.
  - It contains 254 articles, each starting with a title and its source path. Some names and team links came out blank.
- Said about 450k tokens was too much to send with every question, so the app would need retrieval. See "Accepted / rejected" for my response.

### Exploration notebook (`notebooks/01_handbook_exploration.ipynb`)

- Set up a project `.venv` with `uv` and pinned the versions in `notebooks/requirements.txt`. The notebook downloads the PDF into a gitignored `data/` folder if it's missing and checks its SHA-256.
- **Header/footer:** every page has exactly 4 lines of browser header/footer, all in 8 pt Times-Roman at the page edges. They are removed by font and position. This brings the total down to about 410k tokens.
- **Structure:**
  - It is a collection of articles. The only content outside them is the document title on page 1.
  - There are 254 articles, and every one starts at the top of a new page with a title and a `contents/handbook/....md` source path. 17 titles wrap onto two or more lines.
  - Headings can be told apart by font size: H2 is 15 pt, H3 is 12.5 pt and H4 is 11 pt, all HelveticaNeue-Bold.
  - The PDF is a subset of the handbook. PostHog's website repo had 313 handbook pages at the time, so 59 are not in the PDF, mostly engineering (29) and docs-and-wizard (14).
- **Token distribution (cl100k):**
  - articles: median 1,168, p90 3,328, max 9,628. 147 are over 1k tokens and 19 over 4k.
  - H2 sections: 1,364 sections, median 173, p90 666, max 5,976. 185 are under 50 tokens.
  - H3 sections: 2,580 sections, median 114, p90 342, max 1,756. 554 are under 50 tokens.
- **What the export dropped:** compared the PDF with the markdown source at the last commit before the export (`posthog.com@ae45b60`).
  - 99.4% of normal source lines are found in the PDF text.
  - Text inside collapsed `<details>` sections is almost entirely missing (7% found), across 5 articles. In `marketing/ownership.md` ("Who can help me?"), only the questions are in the PDF, not the answers.
  - Blank names and teams: 216 `<TeamMember>`/`<SmallTeam>` mentions show as empty gaps across 67 articles and 99 pages. Another 20 are inside the dropped collapsed sections. 52 of the 216 come from lines that are nothing but blanks, so their page is approximate (it can be one page early). The full list is in the notebook and in `data/blank_mentions.csv`.
- **Images:** a Sonnet subagent viewed all 77 unique images (78 references, one image used twice) and wrote `notebooks/image_review.md`. It found 5 decorative, 60 covered by the text and 12 with information the text lacks. The three worth captioning are the communication tier list (p41, a page with no text), the sales commission ramp example (p617) and the logo do/don't (p38). The AI viewed p41 and p617 itself to check the review.

## Accepted / rejected

- I rejected all three options for the deleted notes files and wrote my own: delete the old commits and start completely fresh.
- I disagreed that ~450k tokens is too much to send. It fits in commonly available models, so full context becomes the eval baseline, and retrieval comes first. See DECISIONS.md #01.
- I agreed to remove the header/footer.
- I asked for the exploration to be in a notebook instead of throwaway scripts.
- I changed the note-keeping rule: no more reminders to review the notes. Notes and decisions get recorded without being asked.
- Still open: whether to split by article, H2 or H3.

## What the AI got wrong

- In the first two commits it left a blank line between `AI-assisted: yes` and `Co-Authored-By:`. Git reads trailers only from the last paragraph, so it didn't see `AI-assisted` as a trailer. The AI caught this by checking `git log` for the trailer right after committing, and found it empty. Neither commit had been pushed, so it recreated both with the trailers in one block.
- In the first analysis it ran Python with `-I`, which ignores `PYTHONIOENCODING`, so printing emoji from the handbook crashed twice on the Windows console encoding. Fixed with `sys.stdout.reconfigure(encoding="utf-8")`. This didn't affect any results.
- Its first image search ran recursively through all of Documents and timed out. A narrower search of Downloads found the file.
- The notebook first ran on the global Jupyter kernel, which didn't have PyMuPDF. Then `python -m jupyter nbconvert` silently used the global `jupyter-nbconvert` from PATH. Both were caught by the run failing. Fixed by creating a venv kernel and calling `python -m nbconvert` directly.
- The article splitter first merged the document title "PostHog handbook" into the first article's title, because both use the same 22 pt font. The AI caught this while reading the notebook output and fixed it.
- Its first mapping of blank names to pages missed about a third of them. Lines like "by , , and" or "EU: , NA:" have too little text to match on. It switched to checking every source line against the PDF. A spot check of page 1052 then showed that lines made only of blanks can get a page one too early. These are now marked as approximate (`page_exact`) instead of being presented as exact.
