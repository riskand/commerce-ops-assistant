# evals/optimize.py
"""Ask a model to improve a prompt, then make it prove the improvement.

This is the evaluator-optimizer pattern, pointed at the
prompts themselves. The model proposes; it does not decide. Nothing it
writes becomes active -- the new version lands inactive, and the only
thing that promotes it is a better score on the same fixed question set
every other version was measured against.

The gate below matters more than the generation. A rewrite that drops a
placeholder does not fail loudly: str.format on the old values would
raise, but a proposal that merely renames {context} to {passages} would
sail into the registry and break every caller at once."""
import argparse, asyncio, re

from sqlalchemy import select

from app.config import settings
from app.db import Session
from app.loop import call_llm
from app.models import PromptVersion
from app.prompts import PromptError, declared, get_version, render_template

ASK = """Here is a prompt template used in a retrieval-augmented answering
system. Rewrite it to get more accurate, better-grounded answers.

Keep every {{placeholder}} exactly as it is: the surrounding code fills
them by name and renaming or removing one breaks it. Reply with the
rewritten template and nothing else.

{template}"""

FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.S)


def accept_proposal(current: str, proposed: str) -> str:
    """A model's reply is untrusted text. Pull the template out of the
    fence it probably added, refuse anything still carrying a fence
    marker, then refuse anything whose placeholders do not match the
    template it was asked to improve.

    Refusing on a leftover backtick marker is what makes the extraction
    safe to get wrong: a reply whose backtick fence this regex does not
    recognise fails loudly instead of storing the prose. It is not a
    general prose filter -- a reply with no fence at all, or one fenced
    with tildes, still passes if its placeholders match, because a
    preamble does not change the placeholder set. The eval set is the
    backstop: a proposal carrying an apology in front of the template
    scores badly and never gets activated."""
    text = proposed.strip()
    match = FENCE.search(text)
    if match:
        text = match.group(1).strip("\n")
    if "```" in text:
        raise PromptError("proposal still contains a code fence marker")
    try:
        got = declared(text)
    except ValueError as exc:
        raise PromptError(f"proposal is not a valid template: {exc}") from exc
    want = declared(current)
    if want != got:
        lost = ", ".join(sorted(want - got)) or "none"
        added = ", ".join(sorted(got - want)) or "none"
        raise PromptError(
            f"proposal changed the template contract "
            f"(dropped: {lost}; invented: {added})")
    # The contract check passes on {} because declared() drops empty field
    # names, so prove the template actually renders before storing it.
    # Otherwise the failure surfaces later as an IndexError inside every
    # request that uses the prompt, long after anyone could connect it to
    # this proposal.
    try:
        render_template(text, {name: "" for name in want})
    except (PromptError, IndexError, KeyError, ValueError) as exc:
        raise PromptError(f"proposal does not render: {exc}") from exc
    return text


async def propose(name: str) -> int:
    current = await get_version(name)
    resp = await call_llm(
        [{"role": "user", "content": ASK.format(template=current.template)}],
        tools=[])
    raw = "".join(b["text"] for b in resp["content"] if b["type"] == "text")
    template = accept_proposal(current.template, raw)
    async with Session() as session, session.begin():
        highest = (await session.execute(
            select(PromptVersion.version)
            .where(PromptVersion.name == name)
            .order_by(PromptVersion.version.desc()).limit(1))).scalar_one()
        session.add(PromptVersion(name=name, version=highest + 1,
                                  template=template,
                                  variables=sorted(declared(template)),
                                  active=False))
    return highest + 1


async def main(name: str) -> None:
    version = await propose(name)
    print(f"proposed {name} v{version}, inactive.")
    print(f"score it:  python -m evals.prompts --name {name}")
    print(f"ship it:   curl -X POST {settings.app_url}/prompts/{name}/activate "
          f"-H 'content-type: application/json' -d '{{\"version\": {version}}}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="rag_answer")
    a = p.parse_args()
    asyncio.run(main(a.name))
