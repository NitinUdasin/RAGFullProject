# Generation

The final LLM call that produces an answer from the assembled context. Supports Anthropic Claude, OpenAI, and local models through a unified provider interface.

## Pipeline Position

```
[ContextAssembly] → [Generation] → [Response / PageIndex]
```

## Provider Interface

All providers implement a common async interface:

```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system: str = "") -> str: ...

    @abstractmethod
    async def stream(self, prompt: str, system: str = "") -> AsyncIterator[str]: ...
```

## Provider Implementations

### Anthropic Claude

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-6",          # or claude-sonnet-4-6, claude-haiku-4-5
    max_tokens=1024,
    system=system_prompt,
    messages=[{"role": "user", "content": prompt}],
)
answer = response.content[0].text
```

### OpenAI

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ],
)
answer = response.choices[0].message.content
```

### Local (Ollama)

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0)
answer = llm.invoke(prompt)
```

## Streaming with FastAPI

```python
from fastapi.responses import StreamingResponse

async def stream_answer(query: str):
    async def event_generator():
        async for chunk in llm_provider.stream(prompt):
            yield f"data: {chunk}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

LangGraph streaming uses `astream_events`:

```python
async for event in graph.astream_events(inputs, version="v2"):
    if event["event"] == "on_chat_model_stream":
        yield event["data"]["chunk"].content
```

## Generation Parameters

| Parameter | Recommended | Effect |
|---|---|---|
| `temperature` | 0.0–0.2 | Lower = more factual / deterministic |
| `max_tokens` | 512–2048 | Cap output length |
| `top_p` | 0.9 | Nucleus sampling (leave at default unless tuning) |

For RAG, keep temperature low (0–0.2) to reduce hallucination.

## Hallucination Guardrails

- Instruct the model to say "I don't know" if the answer is not in the context
- Add a post-generation faithfulness check node in the LangGraph pipeline
- Optionally use structured output to force citation of source IDs
