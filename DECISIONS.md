# Decisions

One entry per design choice: the options considered, the evidence, what I chose, and what the AI suggested (if anything).

<!-- Template:

## NN. Title (YYYY-MM-DD)

- **Options:** ...
- **Evidence:** ...
- **Chosen:** ...
- **AI suggested:** ...

-->

## 01. Build retrieval first; use full-context as an eval baseline later (2026-10-24)

- **Options:** (a) put the whole handbook in the model's context on every question; (b) retrieve only the relevant pieces (RAG).
- **Evidence:** the handbook is about 410k tokens once the page header and footer are removed (cl100k count, see `notebooks/01_handbook_exploration.ipynb`). That fits in the context window of commonly available models, but every question would pay for the whole handbook and wait for it.
- **Chosen:** build the retrieval pipeline first. Later, add evals that compare it with the full-context approach, which serves as a baseline.
- **AI suggested:** retrieval. It said ~450k tokens was too much to send with every question. I pointed out that it fits in common models, so full context is a valid baseline to measure against, not something to dismiss.

## 02. Strip the browser header and footer from every page (2026-10-24)

- **Options:** keep them, strip them by regex on the text, or strip them by font and position.
- **Evidence:** every page has exactly 4 such lines: date/time and "PostHog handbook" at the top, a `file://` URL and "N/1076" at the bottom. All are 8 pt Times-Roman within 30 pt of the page edge. No handbook content uses that font.
- **Chosen:** remove them.
- **AI suggested:** removing them by font and position, which is what the notebook does.

## 03. Explore the PDF in a notebook with a pinned environment (2026-10-24)

- **Options:** ad-hoc scripts or a notebook; global Python or a project venv.
- **Evidence:** the global Jupyter kernel couldn't import PyMuPDF, so the results depended on whatever was installed on the machine.
- **Chosen:** a notebook (my request), so the exploration is kept and can be rerun.
- **AI suggested:** a project `.venv` created with `uv`, with versions pinned in `notebooks/requirements.txt`. The notebook is executed and committed with its outputs.

## 04. Don't commit the handbook PDF; download and verify it instead (2026-10-24)

- **Options:** commit the 50 MB PDF, use Git LFS, or download it on demand.
- **Evidence:** at 50 MB it is close to GitHub's 50 MB warning size. The Box link serves the file directly.
- **Chosen:** download into a gitignored `data/` folder and check its SHA-256. This was the AI's default; I haven't overruled it.
- **AI suggested:** the same.

## 05. Use the website repo as a reference to find what the PDF dropped (2026-10-24)

- **Options:** detect blank names with text patterns in the PDF, or compare against the handbook's markdown source on GitHub.
- **Evidence:** the blanks leave no trace in the PDF, with no link and no hidden text, so pattern-matching would miss cases and produce false hits. The source shows that they are `<TeamMember>` and `<SmallTeam>` components.
- **Chosen:** compare against `PostHog/posthog.com` at commit `ae45b60`, the last commit before the PDF was generated on 2026-04-20. This is for analysis only; the PDF is still the source of truth for the app.
- **AI suggested:** the same.
