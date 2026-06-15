You are creating evaluation data for a retrieval-augmented generation (RAG) system.

Read the passage below and write:
- one factual question that can be answered using only information stated in the passage
- a concise reference answer to that question, using only information from the passage

Respond with a single JSON object and nothing else - no markdown, no code fences, no extra commentary - in exactly this format:
{{"question": "<question text>", "reference": "<answer text>"}}

Passage:
{passage}
