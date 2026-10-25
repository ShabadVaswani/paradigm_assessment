# Session 02: Chunking and image pipeline (2026-10-24)

Tool: Claude Code (Opus 5.5) in VS Code, with six Claude Code subagents on Sonnet for the image descriptions.

## Goal

Turn the handbook PDF into retrieval data: chunks of a sensible size that follow the document's structure, and images that are described, attached to their chunks and retrievable on their own.

## Key prompts

> I think 400-800 token is a good lenght but give me your opinion on it, add 1-2 sentence overlap, we would need a semantic aware as well as token aware chunker,
> skip names if they are not present
> yeah these images are important, what would be the best way to handle them? create a pipeline which uses sonnet 5.5 to review those images, we will connect to azure later, for now use our own sonnet agents, and write the code for processing that data using anthropic api from azure key vault, you will have access to api key once I connect it to cloud, and then we can verify if its working or not, all 77 images, extract text from pdf, in parallel extract information from images in a context aware way, and attach the image information to chunks as well, make sure image caption go into their own respective chunks with their image description with page number, store images as its own retievable item linked to its chunk, which can be called as a function call which the agent can call

## What the AI did / proposed

- Built a Python package, `src/handbook_qa/`:
  - `pdf_extract.py`: PDF to articles made of headings, paragraphs, tables, code and images.
    - Paragraphs and list items are rebuilt from line positions, because the PDF puts several list items in one text block and has no bullet characters.
    - Tables come from PyMuPDF's table finder as markdown: 150 tables.
    - Code is detected by font and indent: 103 blocks.
  - `images.py`: renders each image's area of the page to PNG and records the text around it.
  - `describe.py` + `llm.py`: describes images with Claude Sonnet 5.5.
    - Structured output, effort `medium`, up to 8 requests at once.
    - The API key comes from Azure Key Vault.
    - Works with both the Claude API and Microsoft Foundry.
    - Results are cached by image hash.
  - `chunking.py`: the chunker (decision 06).
  - `tools.py`: the `get_image` tool definition and handler.
  - `ingest.py`: the command line (`download`, `extract`, `describe`, `chunk`, `all`, plus `image-tasks` / `import-descriptions` for describing without the API).
- Gave its opinion on chunk size: 400-800 is sensible for this handbook, with the overlap applied only to cuts inside a section (decision 06).
- Described all 77 images with six Sonnet subagents working from the exact prompt the API would get. It validated their output against the schema and imported it into `artifacts/image_descriptions.jsonl`.
- **Result:**
  - 254 articles became 859 chunks: median 514 tokens, maximum 768.
  - All 77 images are described, placed in their chunks with page numbers, and linked from `data/build/images.jsonl`.
  - 22 tests pass.
- **Not done yet:**
  - The API path has not been run, because there's no key yet. The tests only check the request it builds, using a fake client.
  - There is no retrieval or answering yet.

## Accepted / rejected

- I set the chunk range (400-800), the 1-2 sentence overlap, "skip missing names", and the image design: Sonnet 5.5, context-aware, descriptions in chunks, images as their own items, and a tool to fetch them.
- I haven't reviewed the AI's own additions yet: overlap only inside sections, rendering instead of raw extraction, the committed description cache, and Foundry support.

## What the AI got wrong, and how it was caught

- The first extraction merged two list items into one ("…move the task forward. No Status for 7+ days…") and left a space inside hyphen-wrapped words ("art- board", "right- hand"). It caught both by reading the extracted text. The hyphen fix was simple. The merge can't be settled from line positions alone, because the second item starts with a word too short to show a line break. It added a rule for list lines (sentence end followed by a capital letter means a new item) and a test for it.
- One of its chunker tests expected two chunks from sections too small to split. The test failed. The chunker had correctly merged them, so the test was wrong and the AI fixed the test.
- Two of the six subagents described some images without reading the surrounding text: one image in one batch, eight in another, against the instructions. Both said so in their reports. The AI sent them back to redo those images with the context, and they did. Their first attempt to read the task files with Python had failed on Windows text encoding.
- The subagents shared one scratch folder. One agent's helper script overwrote another's, and one agent ran the other's script, rewriting some of the other batch's result files. The agent reported this. Before importing, the AI checked every caption against the section its image sits in, and looked at one of the affected images (p866) and its description directly. All matched.
- Noted, not yet handled: some dollar amounts in the PDF were rendered as maths and lost their spaces, e.g. on p21 "$50,000 per year - they only need to register once…" became "50, 000peryear − theyonlyneedtoregisteronce…".
