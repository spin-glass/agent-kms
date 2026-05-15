From the tail of the following session transcript, extract up to **3
lessons** that a *future session, working on a different task in this
project*, would benefit from knowing in advance. Output JSON array only.

Format: [{{"text": "lesson body (1-3 sentences)", "confidence": 0.0-1.0}}]

Selection criteria (ALL must hold):
1. **Reusable** — applies to a different target (different feature / bug /
   module), not just the one in this session.
2. **Concrete** — contains literal values, API names, file kinds, or wrong
   assumptions that were corrected.
3. **Grounded** — reflects an actual failure / correction / discovery in
   THIS session (NOT proposed / planned work).

Reject entirely (do NOT output if any apply):
- ❌ Git / PR / CI / branch procedure (already in project conventions).
- ❌ Vague advice ("verify X", "be careful of Y").
- ❌ One-off typos / single-shot operational mistakes.
- ❌ Implementation details of abandoned experiments.
- ❌ Internal evaluation metrics (coverage / MRR / NDCG / ranking).
- ❌ Single design decisions (renames, deletions).
- ❌ Notes about in-progress work or next-up todos.

Forbidden vocabulary: skeleton, placeholder, defer, minimal, scope-narrow,
dummy. **Confidence < 0.85 must NOT be output.** Return ``[]`` if nothing
qualifies.

**Output language**: Match the transcript's primary natural language. If
the transcript is mostly Japanese, write each ``text`` field in natural,
idiomatic Japanese — proper particles, no translated-Chinese phrasing,
no missing okurigana. Code identifiers, file paths, and English technical
terms (API names, command flags, library names) stay in their original
form. If the transcript is mostly English, write in English.

transcript:
{transcript_tail}

JSON only: /no_think
