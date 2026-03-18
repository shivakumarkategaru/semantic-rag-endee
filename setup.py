from setuptools import setup, find_packages

setup(
    name="semantic-rag-endee",
    version="1.0.0",
    description="Semantic RAG system powered by Endee vector database",
    author="Your Name",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers>=2.7.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "endee>=0.1.0",
    ],
    extras_require={
        "llm": ["transformers>=4.40.0"],
        "dev": ["pytest>=8.0.0", "pytest-cov>=5.0.0", "black>=24.0.0", "ruff>=0.4.0"],
    },
)
