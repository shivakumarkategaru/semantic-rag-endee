import os
import logging
from typing import List, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIGenerator:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        logger.info("OpenAI ready. Model: %s", self.model)

    def generate(self, question: str, context_texts: List[str]) -> str:
        context = "\n".join([f"[{i+1}] {t}" for i, t in enumerate(context_texts[:3])])
        prompt = f"""Answer using only this context:
{context}

Question: {question}
Answer:"""
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Answer based on context only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.2,
        )
        return r.choices[0].message.content.strip()