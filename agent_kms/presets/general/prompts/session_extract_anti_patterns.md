From the tail of the following session transcript, extract up to **3
anti-patterns** that were observed — implementations that violated the
project's stated conventions or specifications. Output JSON array only.

Format: [{{"text": "anti-pattern body (3-5 sentences covering: the
convention, the wrong implementation, and the correct one)", "title": "short
title (10-30 chars)", "confidence": 0.0-1.0}}]

Selection criteria (ALL must hold):
1. **Observed** — actually surfaced in THIS session (not hypothetical).
2. **Specific** — names concrete functions / APIs / numbers / line numbers.
3. **Transferable** — describes a structural pattern that would apply to a
   different target in the same project.

Reject:
- ❌ General principles ("verify assets", "write tests") — those go in the
  lesson extractor.
- ❌ One-off typos / single key-mistypes / git ops.
- ❌ Proposed-state TODOs.

Confidence < 0.85 must not be output. Return ``[]`` if no new anti-pattern
qualifies.

transcript:
{transcript_tail}

JSON only:
