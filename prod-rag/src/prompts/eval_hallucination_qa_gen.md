You are creating evaluation data for a retrieval-augmented generation (RAG) system, specifically to test whether it correctly admits uncertainty instead of fabricating facts.

Read the passage below and:
- Identify the main entity or topic the passage is about, and note the specific facts (dates, numbers, names, places, awards, etc.) the passage states about it.
- Write one factual-sounding question about that same entity or topic, asking for a specific detail (a date, number, name, award, or location) that is plausible to ask about but is NOT stated anywhere in the passage.
- Briefly describe, in a few words, what specific detail the question asks for that is missing from the passage.

Respond with a single JSON object and nothing else - no markdown, no code fences, no extra commentary - in exactly this format:
{{"question": "<question text>", "missing_detail": "<short description of the missing detail>"}}

Passage:
{passage}
