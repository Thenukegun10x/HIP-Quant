"""Create the fixed text stream used by the same-backend HQ/IQ quality test.

The pre-existing long-form corpus is stored as token IDs, whereas llama.cpp's
perplexity tool accepts text.  Each 512-token source window is individually
round-tripped through the source tokenizer before being joined with explicit
newlines.  llama.cpp then records the *actual* token stream in its base-logit
file; subsequent HQ and IQ runs consume that saved stream rather than
re-tokenizing, which locks all three GGUF evaluations to identical tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    fragments: list[str] = []
    sample_manifest: list[dict[str, str]] = []
    for sample in corpus["samples"]:
        input_ids = [int(token) for token in sample["input_ids"]]
        fragment = tokenizer.decode(
            input_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        round_trip = tokenizer.encode(fragment, add_special_tokens=False)
        if round_trip != input_ids:
            raise ValueError(
                f"Tokenizer round-trip mismatch for {sample['id']}: "
                f"expected {len(input_ids)} tokens, got {len(round_trip)}"
            )
        fragments.append(fragment)
        sample_manifest.append(
            {
                "id": sample["id"],
                "domain": sample["domain"],
                "input_ids_sha256": sample["input_ids_sha256"],
            }
        )

    text = "\n\n".join(fragments) + "\n"
    encoded = tokenizer.encode(text, add_special_tokens=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "three_way_quality_text_v1",
                "source_corpus": str(args.corpus.resolve()),
                "source_suite": corpus["suite"],
                "source_window_count": len(fragments),
                "source_window_tokens": corpus["sequence_length"],
                "join_separator": "\\n\\n",
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "hf_token_count_with_bos": len(encoded),
                "samples": sample_manifest,
                "note": (
                    "The saved llama.cpp base-logit file, not this provisional HF token count, "
                    "is the authoritative evaluation token stream."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] wrote {args.output} ({len(text):,} UTF-8 characters)")
    print(f"[done] source windows={len(fragments)}, HF tokens with BOS={len(encoded):,}")


if __name__ == "__main__":
    main()
