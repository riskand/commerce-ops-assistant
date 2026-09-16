# app/routes/prompts.py
import difflib

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, update

from app.db import Session
from app.models import PromptVersion
from app.prompts import PromptError, get_version

router = APIRouter(tags=["prompts"], prefix="/prompts")


class Activate(BaseModel):
    version: int


@router.get("")
async def list_prompts() -> list[dict]:
    async with Session() as session:
        rows = (await session.execute(
            select(PromptVersion).order_by(PromptVersion.name,
                                           PromptVersion.version))
        ).scalars().all()
    out: dict[str, dict] = {}
    for row in rows:
        entry = out.setdefault(row.name, {"name": row.name, "versions": 0,
                                          "active_version": None})
        entry["versions"] += 1
        if row.active:
            entry["active_version"] = row.version
    return list(out.values())


@router.get("/{name}")
async def show_prompt(name: str) -> list[dict]:
    async with Session() as session:
        rows = (await session.execute(
            select(PromptVersion).where(PromptVersion.name == name)
            .order_by(PromptVersion.version.desc()))).scalars().all()
    if not rows:
        raise HTTPException(404, f"no prompt named {name!r}")
    return [{"version": r.version, "active": r.active,
             "variables": r.variables, "template": r.template,
             "created_at": r.created_at.isoformat()} for r in rows]


@router.get("/{name}/diff")
async def diff_prompt(name: str, a: int, b: int) -> dict:
    try:
        left, right = await get_version(name, a), await get_version(name, b)
    except PromptError as exc:
        raise HTTPException(404, str(exc)) from exc
    lines = difflib.unified_diff(
        left.template.splitlines(keepends=True),
        right.template.splitlines(keepends=True),
        fromfile=f"{name} v{a}", tofile=f"{name} v{b}")
    return {"diff": "".join(lines)}


@router.post("/{name}/activate")
async def activate(name: str, req: Activate) -> dict:
    """Deactivate then activate, in one transaction. The partial unique
    index means the database would reject the second UPDATE if the first
    had not landed, so the ordering is enforced rather than assumed."""
    try:
        await get_version(name, req.version)
    except PromptError as exc:
        raise HTTPException(404, str(exc)) from exc
    async with Session() as session, session.begin():
        await session.execute(
            update(PromptVersion).where(PromptVersion.name == name)
            .values(active=False))
        result = await session.execute(
            update(PromptVersion)
            .where(PromptVersion.name == name,
                   PromptVersion.version == req.version)
            .values(active=True))
        if result.rowcount != 1:
            raise HTTPException(
                404, f"prompt {name!r} has no version {req.version}")
    return {"name": name, "active_version": req.version}
