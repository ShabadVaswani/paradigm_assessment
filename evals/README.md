# Evaluation set

`handbook_eval.jsonl` has 100 questions about the PostHog handbook. Each has a reference answer and the evidence
behind it, and is labelled `dev` or `holdout`:

- **dev (40):** used during development to compare and choose methods: the reranker, k and the dynamic cutoffs.
  Scores on these are optimistic.
- **holdout (60):** written afterwards and used only in the final face-off (`notebooks/12_final_faceoff.ipynb`). They
  were committed to git before any system answered them.

| type | dev | holdout | what it tests |
|---|---|---|---|
| existing / single | 11 | 14 | answered from one place |
| reworded | 9 | 12 | everyday wording that avoids the handbook's own terms |
| multi-section | 8 | 9 | an answer that needs 2-3 sections of one article |
| multi-article | 12 | 15 | an answer that needs 2-3 articles |
| not-covered | 0 | 10 | the handbook doesn't answer it; the right response is to say so |

Each line looks like:

```json
{"id": "C1-03", "question": "...", "type": "reworded", "answer": "...",
 "evidence": [{"article": "contents/handbook/growth/sales/contract-rules.md", "quote": "...", "supports": "..."}],
 "origin": "...", "split": "dev"}
```

Not-covered questions have no evidence. Instead they have `notes`: the terms searched across the whole handbook
(`absence_check`), and what the handbook does say nearby (`near_miss`).

## How it was built

The answer key was written by reading the handbook, not by running the app's search, so it doesn't inherit the
retrieval system's blind spots.

1. `tools/dump_articles.py` writes every article as plain text, exactly as the chunks render it, in four areas.
2. Claude Code subagents (Opus), one per area, read those files and wrote questions with verbatim quotes as
   evidence. They were told not to read the app's code, chunks or vectors.
   - **dev:** written in session 02. The 11 "existing" questions are earlier spot-check questions, re-keyed by
     reading.
   - **holdout:** written later, with the dev questions shown to the subagents so they wouldn't be repeated.
3. **Not-covered questions** were checked by searching all four areas with several phrasings. The main session then
   re-checked each one with its own synonym searches.
4. `tools/validate_draft.py` checks:
   - every quote is verbatim, on one line, 6-30 words and unique in the handbook;
   - multi-article questions use 2+ articles;
   - not-covered questions have no evidence and 3+ search terms;
   - ids and questions are new.
5. The main session reviewed every question:
   - **dev:** A-09 was replaced because its evidence sat in a single chunk.
   - **holdout:** B-17 was relabelled from multi-section to single, for the same reason.
6. `tools/merge_drafts.py` produced this file.

## How it is scored

Evidence is stored as quotes, not chunk IDs, so the key survives re-chunking. At evaluation time each quote is mapped
to the chunk(s) of its article that contain it by string matching (`src/handbook_qa/evaluation.py`). An image
description quote also matches the image's own record.

- **Chunk recall@k:** a question's *expected chunks* are the distinct places its quotes live. Chunk recall@k is the
  fraction of them found in the top k results, averaged over questions. Not-covered questions have no expected
  chunks and are left out of recall.
- **Answers:** judged by Claude Opus 5.5 against the reference answer (`src/handbook_qa/judge.py`). For not-covered
  questions, the right behaviour is to say plainly that the handbook doesn't cover it.

## Caveats

- 60 held-out questions can show large differences, not small ones; each question is 1.7% of the holdout score.
- Some answers depend on image descriptions; those test the image pipeline too.
- A few reference answers rest on handbook passages that contradict each other. The judges flagged these, and they
  are noted in the session logs.
