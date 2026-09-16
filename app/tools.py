# app/tools.py
def get_order(order_id: str) -> dict:
    """Canned. mcp_server/ points the equivalent tools at a real database."""
    return {
        "order_id": order_id,
        "status": "partially_synced",
        "channel": "channel-b",
        "line_items": [{"sku": "SKU-99", "qty": 2, "price": 24.50}],
        "sync_state": {"channel-a": "ok",
                       "channel-b": "error: SKU not found on channel"},
    }

GET_ORDER_SCHEMA = {
    "name": "get_order",
    "description": "Look up one order by ID. Returns status, channel and sync state.",
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "e.g. SO-1042"},
        },
        "required": ["order_id"],
    },
}
