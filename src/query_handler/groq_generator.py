import os
import logging
from typing import List
from groq import Groq

logger = logging.getLogger(__name__)


class GroqGenerator:
    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str = None):
        self.model_name = model
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.chat_history = [
            {"role": "system", "content": "You are a helpful, smart, and friendly AI assistant. You can help with anything — programming, general knowledge, writing, introductions, explanations, advice, and more. Always understand the user's intent and give a clear, helpful, human-like response. Remember the full conversation history."}
        ]
        logger.info("Groq ready. Model: %s", model)

    def generate(self, question: str, context_texts: List[str]) -> str:
        context = "\n".join([f"[{i+1}] {t}" for i, t in enumerate(context_texts[:3])]) if context_texts else ""

        user_message = question
        if context:
            user_message = f"{question}\n\n(Relevant context: {context})"

        self.chat_history.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.chat_history,
            max_tokens=800,
            temperature=0.7,
        )

        answer = response.choices[0].message.content.strip()
        self.chat_history.append({"role": "assistant", "content": answer})
        return answer
