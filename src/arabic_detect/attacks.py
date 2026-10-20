"""Character- and word-level perturbations that make ChatGPT Arabic text look human-typed.

Six attack sets are built from the out-of-domain ChatGPT answers, each the same size as the original:

1. random character replacement - swap letters for letters of the same skeleton (dots moved/removed)
2. homoglyph substitution       - swap letters for visually identical code points (Persian/Urdu forms)
3. phonetic substitutions       - misspellings that follow pronunciation (ذ/ز, ث/س, ة/ه, hamza forms...)
4. keyboard layout errors       - a letter replaced by a horizontal neighbour on the Arabic keyboard
5. word segmentation errors     - words wrongly joined, split, or with the article/conjunction detached
6. hybrid                       - all of the above applied in sequence

Each attack takes a ``rate`` (probability per eligible character or word) and a ``seed`` so that the
attack sets are reproducible.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence

# Letters that share a rasm (base shape) and differ only in dots.
SKELETON_GROUPS: Sequence[str] = ("بتثني", "جحخ", "دذ", "رز", "سش", "صض", "طظ", "عغ", "فق", "هة")

# Visually identical (or nearly so) code points from the Arabic block's Persian/Urdu letters.
HOMOGLYPHS: Dict[str, str] = {
    "ي": "ی",  # U+064A -> U+06CC Farsi yeh
    "ى": "ی",  # U+0649 -> U+06CC
    "ك": "ک",  # U+0643 -> U+06A9 keheh
    "ه": "ە",  # U+0647 -> U+06D5 ae
    "ة": "ۃ",  # U+0629 -> U+06C3 teh marbuta goal
    "ا": "ٱ",  # U+0627 -> U+0671 alef wasla
    "٠": "۰", "١": "۱", "٢": "۲", "٣": "۳", "٧": "۷", "٨": "۸", "٩": "۹",
}

# Common spelling errors driven by pronunciation (MSA and dialectal speech).
PHONETIC: Dict[str, Sequence[str]] = {
    "ذ": ("ز", "د"),
    "ث": ("س", "ت"),
    "ظ": ("ض", "ز"),
    "ض": ("ظ", "د"),
    "ق": ("ء", "ك"),
    "ة": ("ه",),
    "ى": ("ي", "ا"),
    "أ": ("ا",),
    "إ": ("ا",),
    "آ": ("ا",),
    "ئ": ("ء", "ي"),
    "ؤ": ("ء", "و"),
    "ص": ("س",),
    "ط": ("ت",),
}

# Standard Arabic (101) keyboard, letter rows only.
KEYBOARD_ROWS: Sequence[str] = ("ضصثقفغعهخحجد", "شسيبلاتنمكط", "ئءؤرىةوزظ")

ARTICLE_PREFIXES: Sequence[str] = ("وال", "بال", "فال", "كال", "ال", "و")

ARABIC_LETTERS = set("ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي")


def _keyboard_neighbours() -> Dict[str, List[str]]:
    neighbours: Dict[str, List[str]] = {}
    for row in KEYBOARD_ROWS:
        for i, ch in enumerate(row):
            neighbours[ch] = [row[j] for j in (i - 1, i + 1) if 0 <= j < len(row)]
    return neighbours


KEYBOARD_NEIGHBOURS = _keyboard_neighbours()
SKELETON_ALTERNATIVES: Dict[str, List[str]] = {
    ch: [o for o in group if o != ch] for group in SKELETON_GROUPS for ch in group
}


def _substitute(text: str, table: Dict[str, Sequence[str]], rate: float, rng: random.Random) -> str:
    out = []
    for ch in text:
        options = table.get(ch)
        if options and rng.random() < rate:
            out.append(rng.choice(list(options)))
        else:
            out.append(ch)
    return "".join(out)


def random_char_replacement(text: str, rate: float = 0.1, seed: Optional[int] = None) -> str:
    return _substitute(text, SKELETON_ALTERNATIVES, rate, random.Random(seed))


def homoglyph_substitution(text: str, rate: float = 0.1, seed: Optional[int] = None) -> str:
    table = {k: (v,) for k, v in HOMOGLYPHS.items()}
    return _substitute(text, table, rate, random.Random(seed))


def phonetic_substitution(text: str, rate: float = 0.1, seed: Optional[int] = None) -> str:
    return _substitute(text, PHONETIC, rate, random.Random(seed))


def keyboard_errors(text: str, rate: float = 0.1, seed: Optional[int] = None) -> str:
    return _substitute(text, KEYBOARD_NEIGHBOURS, rate, random.Random(seed))


def word_segmentation_errors(text: str, rate: float = 0.1, seed: Optional[int] = None) -> str:
    """Join a word with the next one, split it in two, or detach its article/conjunction prefix."""
    rng = random.Random(seed)
    words = text.split(" ")
    out: List[str] = []
    i = 0
    while i < len(words):
        word = words[i]
        arabic = any(c in ARABIC_LETTERS for c in word)
        if arabic and rng.random() < rate:
            choice = rng.randrange(3)
            if choice == 0 and i + 1 < len(words):  # join with the next word
                out.append(word + words[i + 1])
                i += 2
                continue
            prefix = next((p for p in ARTICLE_PREFIXES if word.startswith(p) and len(word) > len(p) + 1), None)
            if choice == 1 and prefix:  # detach the prefix
                out.extend([prefix, word[len(prefix):]])
                i += 1
                continue
            if len(word) >= 4:  # split inside the word
                cut = rng.randint(2, len(word) - 2)
                out.extend([word[:cut], word[cut:]])
                i += 1
                continue
        out.append(word)
        i += 1
    return " ".join(out)


AttackFn = Callable[..., str]

BASE_ATTACKS: Dict[str, AttackFn] = {
    "random_char": random_char_replacement,
    "homoglyph": homoglyph_substitution,
    "phonetic": phonetic_substitution,
    "keyboard": keyboard_errors,
    "word_segmentation": word_segmentation_errors,
}


def hybrid(text: str, rate: float = 0.1, seed: Optional[int] = None) -> str:
    """Apply every attack in turn, each with its own derived seed."""
    for k, fn in enumerate(BASE_ATTACKS.values()):
        text = fn(text, rate=rate, seed=None if seed is None else seed * 31 + k)
    return text


ATTACKS: Dict[str, AttackFn] = {**BASE_ATTACKS, "hybrid": hybrid}


def apply_attack(name: str, texts: Sequence[str], rate: float = 0.1, seed: int = 0) -> List[str]:
    """Attack every text; text ``i`` uses seed ``seed + i`` so single rows can be regenerated."""
    if name not in ATTACKS:
        raise KeyError(f"unknown attack {name!r}; choose from {sorted(ATTACKS)}")
    fn = ATTACKS[name]
    return [fn(t, rate=rate, seed=seed + i) for i, t in enumerate(texts)]
