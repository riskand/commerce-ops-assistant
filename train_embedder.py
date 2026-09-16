# train_embedder.py
"""LoRA fine-tune of the embedding model on questions this corpus answers.

    python -m evals.make_pairs --out evals/pairs.jsonl     # build the set
    python train_embedder.py --out models/bge-ai-lab-v1    # train
    EMBEDDING_MODEL=models/bge-ai-lab-v1 python reindex.py --yes
    python -m evals.run --out results.json                 # measure

LoRA rather than a full fine-tune, and the reason is arithmetic rather
than fashion. bge-large is 335M parameters; Adam keeps a fp32 copy plus
two moments, so a full fine-tune needs roughly 16 bytes per parameter --
about 5.4GB of optimizer state before a single activation. That does not
fit on a 4GB card. A LoRA adapter trains a few million parameters
instead, leaves the base frozen, and fits with room for a batch size
large enough to matter -- which it must, because the loss below learns
from the other examples in the batch.
"""
import argparse
import json
import pathlib

from datasets import Dataset
from sentence_transformers import (SentenceTransformer,
                                   SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.losses import MultipleNegativesRankingLoss

from app.embedder import MODEL


def load_pairs(path: str, eval_fraction: float = 0.1):
    rows = [json.loads(l) for l in
            pathlib.Path(path).read_text().splitlines() if l.strip()]
    pairs = [{"anchor": r["question"], "positive": r["positive"]}
             for r in rows]
    cut = max(1, int(len(pairs) * eval_fraction))
    # A held-out slice of the *synthetic* pairs, used only to watch the
    # loss fall. The real verdict is evals/qa.jsonl, which this file has
    # never read and the model has never seen.
    return Dataset.from_list(pairs[cut:]), Dataset.from_list(pairs[:cut])


def main(args) -> None:
    from peft import LoraConfig

    model = SentenceTransformer(args.base)
    model.add_adapter(LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        # The attention projections: where a retrieval model's notion of
        # "related" actually lives. Adapting the feed-forward layers as
        # well doubles the trainable parameters for very little on a
        # corpus this size.
        target_modules=["query", "key", "value"]))
    model.max_seq_length = args.max_seq_length

    train, dev = load_pairs(args.pairs)
    print(f"{len(train)} training pairs, {len(dev)} held out")

    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            output_dir=args.out + "-checkpoints",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            # AMP, not pure float16: the master weights stay fp32 and a
            # gradient scaler keeps small gradients from flushing to
            # zero. fp16 rather than bf16 because Turing has no native
            # bf16 tensor cores -- torch.cuda.is_bf16_supported() answers
            # True on a GTX 1650 because it can emulate it, which is a
            # slower way to get the same answer. Measure, do not trust
            # the capability flag.
            fp16=True,
            # Recompute activations in the backward pass instead of
            # keeping them. It buys memory with time -- roughly a third
            # slower -- and on a small card it is what makes a batch size
            # large enough for in-batch negatives possible at all.
            gradient_checkpointing=args.gradient_checkpointing,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
        ),
        train_dataset=train,
        eval_dataset=dev,
        # In-batch negatives: every other positive in the batch is treated
        # as a negative for this anchor. That is why batch size is a
        # quality setting here and not just a speed one -- a batch of 4
        # gives the model three wrong answers to rank against, and a batch
        # of 64 gives it sixty-three.
        loss=MultipleNegativesRankingLoss(model),
    )
    trainer.train()

    # Saves the adapter, not a merged copy of the base model: a few
    # megabytes rather than 1.3GB, and SentenceTransformer(path) loads it
    # back as long as peft is installed -- which is why peft is in
    # requirements-local.txt beside sentence-transformers.
    #
    # Merging with merge_and_unload() is the other option and it is NOT
    # available here: sentence-transformers injects the adapter into the
    # existing BertModel in place rather than wrapping it in a PeftModel,
    # so model[0].auto_model is still a BertModel and has no such method.
    # That call is the first thing most people try, this comment is here
    # because it cost an afternoon, and the adapter is the better artifact
    # anyway -- you can keep six of them and swap.
    model.save_pretrained(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=MODEL)
    p.add_argument("--pairs", default="evals/pairs.jsonl")
    p.add_argument("--out", default="models/bge-ai-lab-v1")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-seq-length", type=int, default=256)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--gradient-checkpointing", action="store_true")
    main(p.parse_args())
