"""
rag_xper.core.generation.llm_interface

Backend-ready LLM interface supporting Google Gemini (modern SDK) and local Ollama.
Includes automatic sub-batching, rate-limit backoff, and timeouts.
"""
from __future__ import annotations

import abc
import concurrent.futures
import re
import time
from typing import List, Optional

from rag_xper.core.exceptions import LLMGenerationError, LLMTimeoutError
from rag_xper.utils.logger import get_logger

logger = get_logger(__name__)


def _extract_retry_delay(exc: Exception, default_delay: float = 20.0) -> float:
    msg = str(exc)
    match = re.search(r"retry in\s+([\d\.]+)\s*s", msg, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) + 1.0
        except ValueError:
            pass
    match = re.search(r"retryDelay['\":\s]+(\d+)", msg, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) + 1.0
        except ValueError:
            pass
    return default_delay


class BaseLLM(abc.ABC):
    """Abstract interface every LLM provider must implement."""

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class GeminiLLM(BaseLLM):
    """Google Gemini implementation of BaseLLM."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.5-flash-lite",
        embedding_model: str = "gemini-embedding-001",
        timeout_seconds: int = 30,
        max_retries: int = 3,
        temperature: float = 0.2,
    ) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._embedding_model = embedding_model
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._temperature = temperature
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            self._use_modern_client = True
        except ImportError:
            import google.generativeai as genai
            self._genai = genai
            self._genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(model_name)
            self._use_modern_client = False

        logger.info(
            "GeminiLLM ready (model=%s, embedding_model=%s, modern_sdk=%s)",
            model_name, embedding_model, self._use_modern_client,
        )

    def generate(self, prompt: str) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 2):
            future = self._executor.submit(self._call_model, prompt)
            try:
                return future.result(timeout=self._timeout)
            except concurrent.futures.TimeoutError as exc:
                future.cancel()
                logger.warning("Gemini call timed out after %ds (attempt %d/%d)", self._timeout, attempt, self._max_retries + 1)
                raise LLMTimeoutError(f"Gemini generation exceeded {self._timeout}s timeout") from exc
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
                    delay = _extract_retry_delay(exc, default_delay=20.0)
                    logger.warning("Gemini rate limit (429) hit. Sleeping %.1fs before retry (%d/%d)...", delay, attempt, self._max_retries + 1)
                    time.sleep(delay)
                else:
                    logger.warning("Gemini call failed on attempt %d/%d: %s", attempt, self._max_retries + 1, exc)
                    time.sleep(1.0)

        raise LLMGenerationError(f"Gemini generation failed after retries: {last_exc}") from last_exc

    def _call_model(self, prompt: str) -> str:
        if self._use_modern_client:
            from google.genai import types
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=self._temperature),
            )
            text = getattr(response, "text", None)
            if not text:
                raise LLMGenerationError("Gemini returned an empty response")
            return text
        else:
            response = self._model.generate_content(
                prompt,
                generation_config={"temperature": self._temperature},
            )
            if not getattr(response, "text", None):
                raise LLMGenerationError("Gemini returned an empty response")
            return response.text

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        sub_batch_size = 32
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), sub_batch_size):
            sub_texts = texts[i : i + sub_batch_size]
            sub_embeddings = self._embed_batch_with_retry(sub_texts)
            all_embeddings.extend(sub_embeddings)
            if i + sub_batch_size < len(texts):
                time.sleep(0.5)

        return all_embeddings

    def _embed_batch_with_retry(self, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                if self._use_modern_client:
                    res = self._client.models.embed_content(
                        model=self._embedding_model,
                        contents=texts,
                    )
                    if hasattr(res, "embeddings") and res.embeddings:
                        return [e.values for e in res.embeddings]
                    if hasattr(res, "embedding") and res.embedding:
                        return [res.embedding.values]
                    raise LLMGenerationError("Gemini returned no embeddings")
                else:
                    vectors: List[List[float]] = []
                    model_name = (
                        self._embedding_model
                        if self._embedding_model.startswith("models/")
                        else f"models/{self._embedding_model}"
                    )
                    for text in texts:
                        result = self._genai.embed_content(
                            model=model_name,
                            content=text,
                            task_type="retrieval_document",
                        )
                        vectors.append(result["embedding"])
                    return vectors
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
                    delay = _extract_retry_delay(exc, default_delay=22.0)
                    logger.warning("Gemini rate limit (429) on embed. Sleeping %.1fs before retry...", delay)
                    time.sleep(delay)
                else:
                    logger.warning("Gemini embedding call failed on attempt %d/%d: %s", attempt, max_retries, exc)
                    if attempt < max_retries:
                        time.sleep(2.0)

        raise LLMGenerationError(f"Gemini embedding call failed after {max_retries} retries: {last_exc}") from last_exc


class OllamaLLM(BaseLLM):
    """Local Ollama implementation of BaseLLM."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.1",
        embedding_model: str = "nomic-embed-text",
        timeout_seconds: int = 60,
        max_retries: int = 1,
        temperature: float = 0.2,
    ) -> None:
        import ollama

        self._client = ollama.Client(host=base_url, timeout=timeout_seconds)
        self._model_name = model_name
        self._embedding_model = embedding_model
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._temperature = temperature
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def generate(self, prompt: str) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 2):
            future = self._executor.submit(self._call_model, prompt)
            try:
                return future.result(timeout=self._timeout)
            except concurrent.futures.TimeoutError as exc:
                future.cancel()
                raise LLMTimeoutError(f"Ollama generation exceeded {self._timeout}s timeout") from exc
            except Exception as exc:
                last_exc = exc

        raise LLMGenerationError(f"Ollama generation failed after retries: {last_exc}") from last_exc

    def _call_model(self, prompt: str) -> str:
        response = self._client.chat(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self._temperature},
        )
        content = getattr(response.message, "content", None) if hasattr(response, "message") else None
        if content is None and isinstance(response, dict):
            content = response.get("message", {}).get("content")
        if not content:
            raise LLMGenerationError("Ollama returned an empty response")
        return content

    def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self._client.embed(model=self._embedding_model, input=texts)
            embeddings = getattr(response, "embeddings", None)
            if embeddings is None and isinstance(response, dict):
                embeddings = response.get("embeddings")
            if not embeddings:
                raise LLMGenerationError("Ollama returned no embeddings")
            return embeddings
        except Exception as exc:
            raise LLMGenerationError(f"Ollama embedding call failed: {exc}") from exc
