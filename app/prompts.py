# app/prompts.py
"""Prompts are rows, not string literals.

An f-string in source has no version, no diff, no rollback, and no way to
answer "which wording produced last month's eval score". A row has all
four. The cost is this module: one indirection between a caller and the
text it sends.

Rendering is strict in both directions -- a value the template never
declared is as much a bug as a placeholder with no value -- because the
failure it prevents is silent. str.format ignores extra keys, so a typo
in a caller's key would leave the placeholder unfilled and ship
"{question}" to the model inside an otherwise plausible prompt.

The template comes from the registry and the values come from users, and
that order is not negotiable: str.format on a user-supplied template
grants attribute access, so a template is trusted input and a value
never is."""
from string import Formatter

from sqlalchemy import select

from app.db import Session
from app.models import PromptVersion


class PromptError(RuntimeError):
    pass


def declared(template: str) -> set[str]:
    """The placeholder names a template actually uses. Formatter().parse
    is the same parser str.format uses, so this cannot drift from what
    rendering will look for."""
    return {name for _, name, _, _ in Formatter().parse(template) if name}


def render_template(template: str, values: dict) -> str:
    want = declared(template)
    have = set(values)
    if want != have:
        missing = ", ".join(sorted(want - have)) or "none"
        extra = ", ".join(sorted(have - want)) or "none"
        raise PromptError(
            f"template variables do not match values "
            f"(missing: {missing}; unexpected: {extra})")
    return template.format(**values)


async def get_version(name: str,
                      version: int | None = None) -> PromptVersion:
    """version=None means whichever row is active. Passing a version is
    for evaluation, where pinning beats mutating the active row and
    racing every other consumer of this table."""
    where = [PromptVersion.name == name]
    where.append(PromptVersion.version == version if version is not None
                 else PromptVersion.active.is_(True))
    async with Session() as session:
        row = (await session.execute(
            select(PromptVersion).where(*where))).scalar_one_or_none()
    if row is None:
        which = (f"version {version}" if version is not None
                 else "active version")
        raise PromptError(f"prompt {name!r} has no {which}")
    return row


async def render(name: str, values: dict,
                 version: int | None = None) -> str:
    row = await get_version(name, version)
    return render_template(row.template, values)
