# Evaluation set

`handbook_eval.jsonl` has 40 questions about the PostHog handbook, each with a reference answer and the evidence that
supports it. It measures retrieval now (`notebooks/08_retrieval_eval.ipynb`) and will be part of the end-to-end eval.

| type | questions | what it tests |
|---|---|---|
| existing | 11 | the earlier spot-check questions with a known answer |
| reworded | 9 | everyday wording that avoids the handbook's own terms |
| multi-section | 8 | an answer that needs 2-3 sections of one article (2-3 chunks) |
| multi-article | 12 | an answer that needs 2-3 articles |

Each line looks like:

```json
{"id": "C1-03", "question": "...", "type": "reworded", "answer": "...",
 "evidence": [{"article": "contents/handbook/growth/sales/contract-rules.md", "quote": "...", "supports": "..."}],
 "origin": "..."}
```

## How it was built

The answer key was written by reading the handbook, not by running the app's search, so it doesn't inherit the
retrieval system's blind spots.

1. `tools/dump_articles.py` writes every article as plain text, exactly as the chunks render it, in four areas.
2. Four Claude Code subagents (Opus), one per area, read those files and wrote questions with verbatim quotes as
   evidence. They were told not to read the app's code, chunks or vectors.
3. `tools/validate_draft.py` checked that every quote is verbatim, on one line, 6-30 words and unique in the
   handbook.
4. The main session reviewed all 40 questions and replaced one (A-09) whose evidence turned out to sit in a single
   chunk. `tools/merge_drafts.py` produced this file.

## How it is scored

Evidence is stored as quotes, not chunk IDs, so the key survives re-chunking. At evaluation time each quote is mapped
to the chunk(s) of its article that contain it by string matching (`src/handbook_qa/evaluation.py`). An image
description quote also matches the image's own record.

A question's **expected chunks** are the distinct places its quotes live. **Chunk recall@k** is the fraction of them
found in the top k results, averaged over questions.

## Caveats

- 40 questions is enough to see large differences, not small ones. Each question is 2.5% of the score.
- The same questions are being used to choose between methods. Tuning many settings against them would overfit, so
  a held-out set should be added before the final numbers are reported.
- Some answers depend on image descriptions (A-01, A-04, B-03, C1-01); those test the image pipeline too.
