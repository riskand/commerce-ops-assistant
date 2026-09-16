# tests/test_domain_models.py
import sqlalchemy

from app.models import Order, SyncError


def test_both_tables_are_tenant_scoped():
    """Every tenant-facing table filters on tenant_id, so it must be indexed."""
    for model in (Order, SyncError):
        assert "tenant_id" in model.__table__.c
        assert model.__table__.c.tenant_id.index is True


def test_order_ref_is_indexed_because_tools_look_up_by_it():
    assert Order.__table__.c.order_ref.index is True


def test_sync_errors_are_indexed_by_channel():
    assert SyncError.__table__.c.channel.index is True
