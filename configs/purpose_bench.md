# Wiki Research Direction

## Core Positioning
Structured knowledge base compiled from Wikipedia paragraphs in a multi-hop QA
benchmark (HotpotQA / MuSiQue / 2WikiMultiHopQA). Used as the retrieval backend
for downstream multi-hop question answering.

## Key Knowledge Domains
General encyclopedic knowledge: people, organizations, places, events,
artistic and literary works, scientific concepts, and the relations between
them.

## Ingestion Focus
- Priority extraction: key facts, entity relationships, temporal information,
  causal connections.
- Moderate extraction: background context, categorical information.
- Skip: redundant information already captured in other pages.

## Target Use Case
Multi-hop question answering — questions that require combining information
from multiple paragraphs or entities to derive the answer.
