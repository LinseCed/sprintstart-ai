# Ingestion System Test Document

This repository contains a multi-format file parser designed for AI ingestion pipelines. The goal is to transform raw files into structured chunks that can later be used in retrieval-augmented generation systems.

## Purpose

The system is responsible for:
- Detecting file types automatically
- Parsing content into chunks
- Adding metadata for downstream processing
- Ensuring compatibility with different formats like Markdown, JSON, and plain text

## Notes

This markdown file is intentionally extended with additional paragraphs to exceed the 512 character threshold required for testing chunk splitting behavior.

The parser should treat this file as a single text block when below threshold, but this version ensures that the input exceeds that limit so that edge cases in chunking logic can be validated properly.

Further filler text is added here to increase total size and simulate realistic documentation that might be found in production repositories.