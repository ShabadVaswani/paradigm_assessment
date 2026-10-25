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

## 06. Chunks of 400-800 tokens, cut along the document's structure (2026-10-24)

- **Options:** whole articles (median 1,168 tokens, 61 over 2k); H2 or H3 sections as they are (185 H2 and 554 H3 sections under 50 tokens); fixed-size windows; packing structural blocks to a token range.
- **Evidence:** the token distribution in `notebooks/01_handbook_exploration.ipynb`. Result with the chosen settings: 859 chunks, median 514 tokens, 10th percentile 366, max 768, none over 800. 113 chunks are under 400, of which 21 are whole short articles; the rest end where a section ends rather than run into the next topic.
- **Chosen:** 400-800 tokens, 1-2 sentence overlap, and a chunker that knows both structure and token counts (my spec). It never crosses articles; never cuts inside a sentence, table row, code line or image description; cuts at a heading once a chunk has 400 tokens; keeps headings with their content; splits big tables by rows, repeating the header. Every chunk starts with the article title, section path, source path and pages.
- **AI suggested:** agreed with 400-800. It applies the overlap only when a cut falls inside a section, because at a heading the previous sentences belong to another topic (17 of 859 chunks got overlap). It used headings and sentences as the "semantic" boundaries rather than embedding-similarity breakpoints: the handbook's headings already mark topic changes, and no embedding model is chosen yet. Token counts use tiktoken cl100k as a proxy until the embedding model is picked.

## 07. Leave the blank names and teams as they are (2026-10-24)

- **Options:** backfill the 216 missing names/teams from the website repo, tidy the leftover punctuation, or leave the text as extracted.
- **Evidence:** the PDF is the source of truth for this assignment; backfilling would answer from data the PDF doesn't contain.
- **Chosen:** skip them (my call). The text keeps gaps like "handled by , , and on the team".
- **AI suggested:** nothing different; it will note this as a known limitation in the README.

## 08. Images: describe them with Claude, put the descriptions in the chunks, and keep the images retrievable (2026-10-24)

- **Options:** ignore images; OCR only; describe each image with a vision model, with or without the surrounding text.
- **Evidence:** the image review in session 01 found 12 of 77 images with facts the text lacks (e.g. the communication tier list, the sales ramp example).
- **Chosen (my design):** a pipeline that describes all 77 images with Claude Sonnet 5.5, in parallel and with context; each description goes into the chunk where the image sits as `[Image <id>, page N] caption description`; each image is also its own record (`data/build/images.jsonl`, with `retrieval_text`) linked to its chunk; a `get_image` tool lets the answering model fetch the picture itself.
- **AI suggested, and I didn't object:**
  - Context = article, section path, page, ~1,500 characters before and ~800 after the image. The prompt asks for what the image adds beyond that text, a transcription of visible text, and an `adds_information` flag (52 of 77 are true).
  - Render the image's area of the page instead of extracting the raw image. The raw extraction of the p617 diagram had a transparent background and showed up as dark-on-black; the render matches what a reader sees.
  - Structured output (Pydantic schema), effort `medium`, server-side refusal fallback on the Claude API.
  - Cache descriptions in a committed file (`artifacts/image_descriptions.jsonl`) keyed by image hash and prompt version, so rebuilding the index doesn't pay for them again.

## 09. For now, describe images with Claude Code Sonnet subagents; the API path waits for the Azure key (2026-10-24)

- **Options:** wait for API access, or produce the descriptions now another way and run the API later.
- **Evidence:** no API key is connected yet.
- **Chosen:** my instruction: use our own Sonnet agents now, write the API code (key from Azure Key Vault) and verify it once the key is connected.
- **AI suggested:** the agents get exactly the prompt the API would get (`python -m handbook_qa.ingest image-tasks`), and their answers go through the same schema validation (`import-descriptions`), labelled `source: claude-code-subagent`. When the API runs, we can compare its descriptions with the agents'. The client supports both the Claude API (`LLM_PROVIDER=anthropic`) and Claude on Microsoft Foundry (`foundry`), because "Azure" could mean either; the key comes from `AZURE_KEY_VAULT_URL` + `ANTHROPIC_KEY_SECRET_NAME` via `DefaultAzureCredential`.
