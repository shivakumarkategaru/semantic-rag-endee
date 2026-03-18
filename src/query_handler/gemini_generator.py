import os
import logging
from typing import List
from google import genai

logger = logging.getLogger(__name__)


class GeminiGenerator:
    def __init__(self, model: str = "gemini-2.0-flash", api_key: str = None):
        self.model_name = model
        api = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api)
        logger.info("Gemini ready. Model: %s", model)

    def generate(self, question: str, context_texts: List[str]) -> str:
        context = "\n".join([f"[{i+1}] {t}" for i, t in enumerate(context_texts[:3])])
        prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say I do not know.

Context:
{context}

Question: {question}
Answer:"""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text.strip()