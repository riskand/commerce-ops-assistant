# evals/corpus.py
import asyncio, json, pathlib

import httpx

from app.config import settings

TENANT = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"   # the demo tenant, Seller A
RETURNS = "aaaaaaaa-0000-4000-8000-000000000001"
SHIPPING = "aaaaaaaa-0000-4000-8000-000000000002"
SELLER = "aaaaaaaa-0000-4000-8000-000000000003"
SYNC = "aaaaaaaa-0000-4000-8000-000000000004"

DOCUMENTS = {
    RETURNS:
        "Returns policy. Buyers may return an item within 30 days of "
        "delivery. Refunds are issued within 30 days of an approved return "
        "request. Returns requested after the 30-day window are declined.",
    SHIPPING:
        "Shipping policy. Standard delivery is three business days. "
        "Shipping to remote postcodes takes an extra 5 business days.",
    SELLER:
        "Seller obligations. Sellers must respond to buyer messages within "
        "24 hours of receipt.",
    SYNC:
        "Channel sync error codes. Error 8541 means the SKU in the order "
        "does not exist on the destination channel; relist the SKU on that "
        "channel and retry the sync. Error 8542 means the channel rejected "
        "the price as outside its allowed range.",
}

# (question, expected_answer, must_cite_doc)
QA = [
    ("How long do buyers have to return an item?",
     "30 days from delivery", RETURNS),
    ("When is a refund issued after a return is approved?",
     "within 30 days of an approved return request", RETURNS),
    ("Can a buyer return an item after 40 days?",
     "No, the window is 30 days", RETURNS),
    ("What happens to a return request that is never approved?",
     "no refund is issued; refunds follow an approved return", RETURNS),
    ("How long does shipping take to remote postcodes?",
     "an extra 5 business days", SHIPPING),
    ("Is delivery to a metropolitan address faster than to a remote one?",
     "yes; remote postcodes take 5 extra business days", SHIPPING),
    ("What is the extra delay for remote delivery?",
     "5 business days", SHIPPING),
    ("How long is standard delivery?",
     "three business days", SHIPPING),
    ("How quickly must a seller reply to a buyer message?",
     "within 24 hours", SELLER),
    ("Is a two-day reply to a buyer acceptable?",
     "No, sellers must respond within 24 hours", SELLER),
    ("What is the seller response time obligation?",
     "24 hours", SELLER),
    ("Which policy governs buyer messages?",
     "the 24-hour seller response rule", SELLER),
]

# questions no document answers; a correct reply says so
UNANSWERABLE = [
    "What commission does the platform take on each sale?",
    "Can I sell alcohol on the platform?",
    "What does sync error 9001 mean?",
    "Do you offer international shipping?",
    "How do I change the bank account my payouts go to?",
]
NOT_ANSWERED = "the documents do not say"

QA_PATH = "evals/qa.jsonl"


async def wait_until_done(client: httpx.AsyncClient, document_id: str) -> dict:
    while True:
        r = await client.get(f"/documents/{document_id}")
        r.raise_for_status()
        doc = r.json()
        if doc["status"] in ("ready", "failed"):
            return doc
        await asyncio.sleep(1)


async def main() -> None:
    async with httpx.AsyncClient(base_url=settings.app_url,
                                 timeout=120) as client:
        for document_id, text in DOCUMENTS.items():
            r = await client.post("/ingest", json={
                "tenant_id": TENANT, "document_id": document_id, "text": text})
            r.raise_for_status()
        for document_id in DOCUMENTS:
            doc = await wait_until_done(client, document_id)
            print(document_id, doc["status"], doc["chunks"])
            if doc["status"] == "failed":
                raise SystemExit(doc["error"])

    rows = [{"tenant_id": TENANT, "question": q, "expected_answer": a,
             "must_cite_doc": doc} for q, a, doc in QA]
    rows += [{"tenant_id": TENANT, "question": q,
              "expected_answer": NOT_ANSWERED, "must_cite_doc": None}
             for q in UNANSWERABLE]
    pathlib.Path(QA_PATH).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {QA_PATH}: {len(rows)} rows")


if __name__ == "__main__":
    asyncio.run(main())
