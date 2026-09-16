# app/context.py
from app.loop import call_llm, final_text

CONTEXT_PROMPT = """Here is a document:
<document>
{document}
</document>

Here is a chunk from it:
<chunk>
{chunk}
</chunk>

In one or two sentences, situate this chunk within the document so it can be
understood on its own. Answer with the sentences only."""


async def situate(document: str, chunk_text: str) -> str:
    prompt = CONTEXT_PROMPT.format(document=document, chunk=chunk_text)
    resp = await call_llm([{"role": "user", "content": prompt}], tools=[])
    return final_text(resp).strip()
