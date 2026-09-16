# tests/test_prompts_schema.py
"""The registry's invariant is a database constraint, not a convention:
two active versions of one prompt must be impossible, not merely
discouraged. Application code that 'always deactivates first' is one
forgotten branch away from a prompt with two active rows and a render
that picks whichever the planner returns first."""
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db import Session
from app.models import PromptVersion

pytestmark = pytest.mark.integration


async def test_version_one_of_each_prompt_is_seeded_and_active():
    async with Session() as session:
        rows = (await session.execute(
            select(PromptVersion).where(PromptVersion.active.is_(True))
        )).scalars().all()
    active = {r.name: r for r in rows}
    assert set(active) >= {"rag_answer", "eval_judge"}
    assert all(r.version == 1 for r in active.values())


async def test_a_second_active_version_is_refused_by_the_database():
    async with Session() as session:
        row = (await session.execute(
            select(PromptVersion).where(PromptVersion.name == "rag_answer")
        )).scalars().first()
        session.add(PromptVersion(name="rag_answer", version=999,
                                  template=row.template,
                                  variables=row.variables, active=True))
        with pytest.raises(IntegrityError):
            await session.commit()


async def test_stored_variables_match_the_template_placeholders():
    from app.prompts import declared
    async with Session() as session:
        rows = (await session.execute(select(PromptVersion))).scalars().all()
    for row in rows:
        assert set(row.variables) == declared(row.template), row.name
