# Self-Correcting RAG with Iterative Retrieval Refinement

[![CI](https://github.com/YOUR_USERNAME/self-correcting-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/self-correcting-rag/actions)
![Python](https://img.shields.io/badge/python-3.11-blue)

> A production-ready RAG system that detects retrieval failures and
> autonomously refines queries before generating answers.

---

## The problem it solves

Standard RAG systems are blind to retrieval failures. If search returns
irrelevant chunks, the LLM generates a hallucinated answer using that
bad context. This system detects poor retrieval and automatically
retries with a better query before answering.

---