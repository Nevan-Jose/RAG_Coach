# RAG Coach Memory System

A retrieval augmented generation (RAG) pipeline that gives an AI coach memory of past sessions.

## The problem it solves

Coaching AI assistants have no memory between sessions. Every conversation starts from scratch. This system fixes that by retrieving relevant past session transcripts and giving them to Claude as context before answering.

## How it works

1. Past session transcripts are stored and embedded using a local AI model
2. A question is asked in natural language
3. The most relevant transcript is retrieved using dot product similarity
4. That transcript is passed to Claude as context
5. Claude answers the question grounded in real session history

## Tech used

- Python
- sentence-transformers (all-MiniLM-L6-v2)
- numpy
- Anthropic Claude API

## How to run
Add your Anthropic API key to a .env file:
ANTHROPIC_API_KEY=your-key-here
## Example

Question: "What has the user committed to doing about stress?"

Retrieved session: Handle stress better

Claude's answer: Based on the transcript, the user committed to taking a 10-minute walk every day.
