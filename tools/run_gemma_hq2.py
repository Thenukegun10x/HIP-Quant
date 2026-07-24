"""Run a Gemma 4 hybrid HQ2 archive through Transformers on ROCm.

The first invocation streams base BF16 tensors and the persistent HQ2 MLP
archive into a hybrid model.  It does not quantize any weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

import hq2


def _sample_next(logits: "torch.Tensor", temperature: float) -> "torch.Tensor":
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Local original Gemma 4 checkpoint directory")
    parser.add_argument("--archive", required=True, help="HQ2 MLP .hq2 archive")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    model = hq2.load_gemma4_hq2(args.base, args.archive)
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    # This explicit prefill/cache loop is deliberately used instead of the
    # current development Gemma 4 unified ``generate()`` wrapper.  It maps
    # directly to the verified inference contract: one prefill then one
    # packed-HQ2 decode step per generated token.
    result = model(**inputs, use_cache=True)
    generated = [inputs["input_ids"]]
    next_token = _sample_next(result.logits[:, -1, :], args.temperature)
    generated.append(next_token)
    cache = result.past_key_values
    for _ in range(1, args.max_new_tokens):
        result = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        next_token = _sample_next(result.logits[:, -1, :], args.temperature)
        generated.append(next_token)
    print(tokenizer.decode(torch.cat(generated, dim=1)[0], skip_special_tokens=True))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
