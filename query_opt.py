"""Local (no LLM) query optimisation: filler removal, stutter dedupe, small-talk detection.

NOTE: Devanagari matras Python ke \\w me nahi aate, isliye yahan \\w use nahi kiya -
tokenization whitespace + explicit punctuation set se hoti hai.
"""
import re

_PUNCT = re.compile(r"[!-/:-@\[-`{-~।॥…“”‘’]+")
_EDGE = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~।॥…“”‘’"

_FILLERS = {
    "um", "umm", "uh", "uhh", "er", "erm", "hmm", "hmmm", "ah", "okay", "ok", "please", "kindly",
    "actually", "basically",
    "कृपया", "प्लीज", "हम्म", "हम्", "अरे", "यार", "ओके",
}
_GREET = {"hi", "hii", "hello", "helo", "hey", "namaste", "namaskar", "नमस्ते", "नमस्कार", "हेलो", "हैलो", "हाय"}
_SMALLTALK = _GREET | {
    "thanks", "thank", "you", "bye", "goodbye", "good", "morning", "afternoon", "evening", "night",
    "how", "are", "ok", "okay", "who", "what", "is", "your", "name",
    "धन्यवाद", "शुक्रिया", "थैंक्स", "थैंक", "यू", "बाय", "अलविदा", "ठीक", "है", "हैं", "हो",
    "आप", "कैसे", "क्या", "तुम", "तुम्हारा", "नाम", "आपका",
}
_PHRASES = re.compile(
    r"\b(?:(?:can|could|would) you(?: please)?(?: tell me| help me)?"
    r"|i (?:want|would like|wanted) to (?:know|ask)(?: about)?"
    r"|i have a (?:question|doubt)|you know|i mean|sort of|kind of)\b",
    re.IGNORECASE,
)


def norm_text(s: str) -> str:
    return " ".join(_PUNCT.sub(" ", (s or "").lower()).split())


def truncate_words(s: str, n: int) -> str:
    words = s.split()
    if len(words) <= n:
        return s
    head = n * 2 // 3
    return " ".join(words[:head] + words[-(n - head):])


def compress_query(text: str, max_words: int = 10_000) -> str:
    s = _PHRASES.sub(" ", text or "")
    out, prev = [], None
    for tok in s.split():
        core = tok.strip(_EDGE).lower()
        if not core or core in _FILLERS:
            continue
        if core == prev:          # stutter: "the the"
            continue
        out.append(tok)
        prev = core
    if len(out) > 3 and out[0].strip(_EDGE).lower() in _GREET:
        out = out[1:]
    res = truncate_words(" ".join(out), max_words)
    return res or (text or "").strip()


def is_smalltalk(q: str) -> bool:
    toks = norm_text(q).split()
    return 0 < len(toks) <= 6 and all(t in _SMALLTALK for t in toks)


# ------------------------------------------------------------------ V3 additions
_CACHE_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and", "or", "with", "what", "how",
    "do", "does", "can", "i", "you", "my", "me", "it", "this", "that", "be", "at", "by", "from", "as", "tell", "about",
    "में", "की", "के", "का", "है", "हैं", "से", "को", "और", "कि", "क्या", "कैसे", "मुझे", "आप", "बताइए", "बताओ",
    "mujhe", "kya", "hai", "ka", "ki", "ke", "me", "batao", "bataiye",
}
_PRONOUNS = {
    "it", "its", "this", "that", "they", "them", "those", "these", "he", "she", "one",
    "इसका", "इसकी", "इसके", "उसका", "उसकी", "उसके", "यह", "वह", "ये", "वो", "इसमें", "उसमें", "इसे", "उसे",
    "iska", "iski", "iske", "uska", "uski", "uske", "isme", "usme", "yeh", "ye", "woh", "wo", "isko", "usko",
}


def cache_key_text(q: str) -> str:
    """Retrieval/web cache ke liye aggressive normalisation: filler/stopwords hatao, unique tokens sort karo.
    'return policy kya hai' aur 'what is the return policy' jaise rephrase ek hi key par aate hain.
    (Sirf retrieval/web cache ke liye - answer cache normal norm_text use karta hai.)"""
    toks = {t for t in norm_text(q).split() if t not in _CACHE_STOP and t not in _FILLERS}
    return " ".join(sorted(toks)) or norm_text(q)


def needs_history(q: str) -> bool:
    """Chhoti query ya pronoun wali query ('iska price?') hi previous turn par depend karti hai."""
    toks = norm_text(q).split()
    return len(toks) < 5 or any(t in _PRONOUNS for t in toks)


def latin_ratio(text: str) -> float:
    letters = [c for c in (text or "") if c.isalpha()]
    return sum(c.isascii() for c in letters) / len(letters) if letters else 1.0


_ENTITY = re.compile(r"\b\w*\d\w*\b|\b[A-Z][a-zA-Z]{3,}\b")


def missing_entities(original: str, summary: str) -> list[str]:
    """Summarise ke baad numbers/IDs/dawai ke naam (Capitalised words) kho gaye ho to wapas dene ke liye."""
    s = (summary or "").lower()
    out, seen = [], set()
    for m in _ENTITY.finditer(original or ""):
        e = m.group(0)
        if e.lower() not in s and e.lower() not in seen:
            seen.add(e.lower())
            out.append(e)
    return out
