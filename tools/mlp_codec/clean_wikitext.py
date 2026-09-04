"""Normalize WikiText-2 raw into ordinary prose.

The corpus distributed by `scripts/get-wikitext-2.sh` is the Moses-tokenized
text with the original spacing preserved: punctuation is space-separated and
intra-word hyphens are escaped as ` @-@ `.  A modern tokenizer sees that as
heavily out-of-distribution text, which drives base perplexity into the
hundreds and pushes the model into a regime where quantization deltas no longer
reflect normal use.

Undoing the tokenization gives a corpus that is still exactly the held-out
WikiText-2 test split, just written the way the model expects to read it.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# The @-@ family encodes characters that Moses split out of the middle of a
# token.  These must be rejoined without surrounding spaces.
_ESCAPES = {" @-@ ": "-", " @,@ ": ",", " @.@ ": "."}

# Punctuation that Moses pushed onto its own token and that should re-attach to
# the preceding word.
_ATTACH_LEFT = [",", ".", ";", ":", "!", "?", "%", ")", "]", "}", "'s", "n't", "'re", "'ve", "'ll", "'d", "'m"]
_ATTACH_RIGHT = ["(", "[", "{", "$", "#"]


def clean(text: str) -> str:
    for token, replacement in _ESCAPES.items():
        text = text.replace(token, replacement)

    for punct in _ATTACH_LEFT:
        text = text.replace(" " + punct, punct)
    for punct in _ATTACH_RIGHT:
        text = text.replace(punct + " ", punct)

    # Quotes alternate open/close; Moses spaced both sides of every " token.
    out, inside = [], False
    for char in text:
        if char == '"':
            out.append(('"' if not inside else '"'))
            inside = not inside
        else:
            out.append(char)
    text = "".join(out)
    text = re.sub(r' "\s*', ' "', text)
    text = re.sub(r'\s*" ', '" ', text)

    # Headings arrive as " = = Title = = "; keep them as markdown-ish headings.
    text = re.sub(r"^ (=+) (.*?) \1 *$", lambda m: f"{m.group(1)} {m.group(2).strip()} {m.group(1)}", text, flags=re.M)

    # Collapse the leading space Moses put on every line, and squeeze the runs
    # of blank lines the raw file uses as paragraph separators.
    text = re.sub(r"^ +", "", text, flags=re.M)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    raw = args.source.read_text(encoding="utf-8")
    cleaned = clean(raw)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(cleaned, encoding="utf-8")

    print(f"{args.source.name}: {len(raw):,} -> {len(cleaned):,} chars")
    print(f"  ' @-@ ' occurrences remaining: {cleaned.count(' @-@ ')}")
    print(f"  wrote {args.destination}")


if __name__ == "__main__":
    main()
