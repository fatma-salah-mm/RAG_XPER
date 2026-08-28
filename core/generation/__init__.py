"""
core/generation package for RAG_XPER.

Contains LLM interfaces (Gemini & Ollama) and the high-level RAG orchestrator.
"""
from core.generation.llm_interface import BaseLLM, GeminiLLM, OllamaLLM
from core.generation.rag_orchestrator import RAGOrchestrator

__all__ = [
    "BaseLLM",
    "GeminiLLM",
    "OllamaLLM",
    "RAGOrchestrator",
]
