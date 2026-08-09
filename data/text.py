"""Char-level text loader. ASCII byte-level tokenization.

ponytail: no tokenizer deps (BPE/SentencePiece add complexity for
the same math). Each byte → one token, vocab_size = 256 (ASCII range).
"""
import os
from typing import Tuple

import numpy as np


ASCII_VOCAB_SIZE = 256


def load_text(path: str) -> Tuple[np.ndarray, int]:
    """Load a text file as a uint8 array of byte values.

    Returns (token_ids, vocab_size). vocab_size is fixed at 256
    regardless of which bytes actually appear (matches ticket 07 spec:
    vocab_size = 256 for char-level).
    """
    with open(path, "rb") as f:
        raw = f.read()
    ids = np.frombuffer(raw, dtype=np.uint8)
    return ids, ASCII_VOCAB_SIZE


# Bundled clean Shakespeare excerpt (~5KB). Defaults to data/shakespeare.txt.
# Tiny but real text — not random — so the trainer has structure to learn.
_BUNDLED_SHAKESPEARE = b"""\
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die-to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream-ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause-there's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
Th'oppressor's wrong, the proud man's contumely,
The pangs of dispriz'd love, the law's delay,
The insolence of office, and the spurns
That patient merit of th'unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would these fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry
And lose the name of action. Soft you now,
The fair Ophelia! Nymph, in thy orisons
Be all my sins remember'd.
"""


def _ensure_bundled_shakespeare() -> str:
    """If data/shakespeare.txt doesn't exist, write the bundled excerpt.
    Returns the path."""
    path = os.path.join(os.path.dirname(__file__), "shakespeare.txt")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(_BUNDLED_SHAKESPEARE)
    return path


# Ensure bundled file exists on import (idempotent)
DEFAULT_PATH = _ensure_bundled_shakespeare()
