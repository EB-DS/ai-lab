# AI for Healthcare

## Evidence-Grounded Healthcare Question Answering with Open-Weight LLMs

Project 3 of the AI Research Laboratory investigates how open-weight large language models can be combined with retrieval-augmented generation (RAG) to produce more reliable, evidence-grounded answers to healthcare questions.

The project is designed simultaneously as:

- a graduate-level AI teaching project;
- a practical healthcare AI application;
- a reproducible research platform;
- a potential foundation for future peer-reviewed research.

## Research Question

> How much does retrieval-augmented generation improve factual correctness, evidence grounding, and reliability of an open-weight LLM for healthcare question answering, and what latency and computational trade-offs accompany those improvements?

## Research Hypotheses

**H1:** RAG-assisted generation will produce more evidence-grounded healthcare answers than LLM-only generation.

**H2:** Improved retrieval will increase retrieval relevance and answer grounding compared with a standard RAG pipeline.

**H3:** Improvements in grounding and retrieval quality will introduce measurable computational costs, including retrieval and end-to-end latency.

## Experimental Design

The project will compare three systems using the same healthcare question set:

1. **LLM Only** — answers questions without retrieved evidence.
2. **Standard RAG** — uses a baseline semantic retrieval pipeline.
3. **Improved RAG** — uses an enhanced retrieval and/or evidence-selection strategy.

The comparison will evaluate factual correctness, answer relevance, retrieval relevance, evidence grounding, citation support, unsupported claims, latency, and token usage.
