# src/llm_client.py

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


class LLMClient:
    def __init__(self):
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
        self.prompt_name = os.getenv("RAG_PROMPT", "rag")

        self.llm = ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def generate_answer(
        self,
        query: str,
        contexts: list[dict[str, Any]],
    ) -> str:
        prompt = self._load_prompt(self.prompt_name)

        prompt_template = PromptTemplate(
            template=prompt,
            input_variables=["question", "context"],
        )

        chain = prompt_template | self.llm | StrOutputParser()

        return chain.invoke(
            {
                "question": query,
                "context": self._format_context(contexts),
            }
        )

    def _load_prompt(self, prompt_name: str) -> str:
        prompt_path = (
            Path(__file__).parent
            / "prompts"
            / f"{prompt_name}.md"
        )

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        return prompt_path.read_text(encoding="utf-8")

    def _format_context(self, contexts: list[dict[str, Any]]) -> str:
        if not contexts:
            return "No context provided."

        formatted_chunks = []

        for idx, item in enumerate(contexts, start=1):
            metadata = item.get("metadata", {})

            formatted_chunks.append(
                f"""
[Source {idx}]
chunk_id: {item.get("id", "unknown")}
doc_id: {metadata.get("doc_id", "unknown")}
source_file: {metadata.get("source_file", "unknown")}
score: {item.get("score", "unknown")}

{item.get("text", "")}
"""
            )

        return "\n".join(formatted_chunks)


llm_client = LLMClient()