# Production LLM Serving & API Engineering

## Overview

This project explores how to turn a locally hosted large language model into a production-style API service.

The system is being built incrementally:

1. FastAPI service foundation
2. Health and model discovery endpoints
3. OpenAI-style chat completion endpoint
4. Backend abstraction
5. Configuration-driven backend selection
6. Automated API tests
7. Transformers-based local LLM backend
8. GPU inference
9. Streaming responses
10. Concurrency and performance testing

## Current Architecture

Client
  |
  v
FastAPI
  |
  v
Backend Interface
  |
  +--> Mock Backend
  |
  +--> Transformers Backend (next)

## Current Endpoints

- `GET /`
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

## Automated Tests

The current API test suite validates:

- root endpoint
- health endpoint
- model listing
- chat completion
- invalid request handling

Current status: **5 tests passed**.

## Current Backend

The API currently uses a mock backend so that the HTTP contract,
validation, configuration, and testing infrastructure can be developed
without requiring GPU compute.

The next milestone is connecting the same API to a real Hugging Face
Transformers model.
