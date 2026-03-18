import os
import sys
import logging
sys.path.insert(0, '.')

logging.disable(logging.CRITICAL)

from src.pipeline import RAGPipeline
from src.query_handler.groq_generator import GroqGenerator

pipeline = RAGPipeline()
pipeline.ingest_file('data/knowledge_base.json')
pipeline.engine._openai_gen = GroqGenerator(model='llama-3.1-8b-instant')

print("\n" + "="*70)
print("        🤖 AI Assistant powered by Groq + Llama 3.1")
print("="*70)
print("  Type your question and press Enter.")
print("  Type 'history' to see previous questions.")
print("  Type 'quit' to exit.")
print("="*70 + "\n")

while True:
    q = input("You: ").strip()

    if not q:
        continue
    if q.lower() in ('quit', 'exit'):
        print("\nGoodbye! 👋\n")
        break
    if q.lower() == 'history':
        history = pipeline.engine._openai_gen.chat_history
        if not history:
            print("\n  No questions asked yet.\n")
        else:
            print("\n  📜 Chat History:")
            for i, item in enumerate(history, 1):
                if item['role'] == 'user':
                    print(f"  You: {item['content']}")
            print()
        continue

    r = pipeline.ask(q)
    print("\n" + "─"*70)
    print(f"  🤖 {r.answer}")
    print("─"*70 + "\n")