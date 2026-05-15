The following is a planning query an AI agent issues when starting a task.
Rewrite it as a richer retrieval query that explicitly enumerates the
universal aspects implied by the original (asset / dependency verification,
faithful adherence to specifications, project conventions, phase workflow,
completion verification, plus any topic-specific signals already present).

Constraints:
- Preserve every topic-specific term in the original query.
- Enumerate 5-8 universal aspects in plain prose.
- Return a single paragraph, 200-400 characters.
- Do NOT wrap in JSON or code fences.
- Forbidden vocabulary: skeleton, placeholder, dummy.

planning query:
{query}

expanded query:
