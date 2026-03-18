from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import logging
logging.disable(logging.CRITICAL)

sys.path.insert(0, '.')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.pipeline import RAGPipeline
from src.query_handler.groq_generator import GroqGenerator

app = Flask(__name__)
CORS(app)

print("Starting RAG Pipeline...")
pipeline = RAGPipeline()
pipeline.ingest_file('data/knowledge_base.json')
pipeline.engine._openai_gen = GroqGenerator(model='llama-3.1-8b-instant')
print("Pipeline ready!")

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    response = pipeline.ask(question)
    sources = [
        {
            'text': s.text[:200],
            'source': s.source,
            'score': round(s.score, 3),
            'chunk_index': s.chunk_index
        }
        for s in response.sources
    ]
    return jsonify({
        'answer': response.answer,
        'sources': sources,
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(port=5000, debug=False)
