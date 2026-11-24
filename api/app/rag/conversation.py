"""Conversational retrieval: turning a follow-up ("explain it in detail", "tell clerly about this") into a
standalone search question, deciding when an earlier answer's evidence may be reused, and spotting broad
questions that need more than one chunk.

Why: a follow-up embedded on its own ("tell clerly about this allobrate the explination") or glued to the
previous question matches no chunk well enough (bge-m3 similarity 0.41-0.44, threshold 0.50), so the agent
answered "I don't know" without even asking the model. The previous question and answer say what "this" is.

Rules kept on purpose:
- A standalone question costs no extra model call (only questions that look like follow-ups are rewritten).
- The rewrite is a search query only. The answer still comes from retrieved, authorized chunks with citations
  and the claim check; an earlier answer is never a source.
- The rewrite uses the same model as the answer, through the same local-only rules (no cloud call in a
  local-only run), and is counted in tokens and cost as its own step.
"""

import re
from difflib import get_close_matches

# Words that point back at the conversation, or ask to transform the previous answer.
REFERENCES = {"it", "its", "this", "that", "these", "those", "they", "them", "their", "he", "she", "him", "her",
              "same", "previous", "earlier", "again", "further", "second", "third", "fourth", "fifth", "continue"}
TRANSFORMS = {
    "detail": ("detail", "detailed", "elaborate", "elaborately", "clearly", "clear", "explain", "explanation",
               "expand", "deeper", "depth", "fully", "properly", "briefly", "brief"),
    "simplify": ("simple", "simply", "simplify", "simpler", "easy", "easier", "layman", "beginner", "kid", "child"),
    "example": ("example", "examples", "instance", "illustrate", "sample", "demo"),
    "summary": ("summary", "summarise", "summarize", "summarised", "summarized", "recap", "points", "bullets",
                "bullet", "tldr", "short", "shorter"),
    "translate": ("translate", "translation", "telugu", "hindi", "english", "తెలుగులో", "हिंदी", "हिन्दी"),
    "continue": ("continue", "continued"),  # "go on" is matched as a phrase; "tell me more" is short anyway
}
_VOCAB = sorted({w for words in TRANSFORMS.values() for w in words} | REFERENCES)
WORD = re.compile(r"[a-zA-Zऀ-ॿఀ-౿]+")

# "in Telugu", "telugu lo", "తెలుగులో", "in Hindi", "हिंदी में" -> the answer language the person asked for.
LANGUAGE_REQUEST = [
    ("te", re.compile(r"\b(in|into|to)\s+telugu\b|\btelugu\s*(lo|lone|language)\b|తెలుగులో|తెలుగు\s*లో", re.I)),
    ("hi", re.compile(r"\b(in|into|to)\s+hindi\b|\bhindi\s*(me|mein|mai|language)\b|हिंदी\s*में|हिन्दी\s*में", re.I)),
    ("en", re.compile(r"\b(in|into|to)\s+english\b", re.I)),
]

# Questions about a whole document or project: they need several sections, not the single closest chunk.
BROAD = re.compile(
    r"\b(what\s+is\s+(this|the|it)\b[\w\s]{0,30}\babout|what\s+does\s+(this|the|it)\b[\w\s]{0,30}\bdo|overview|"
    r"entire|whole|everything|all\s+about|main\s+(features?|points?|ideas?|parts?|components?)|features?|"
    r"architecture|components?|how\s+does\s+(it|this|the\s+\w+)\s+work|how\s+it\s+works|in\s+detail|detailed|"
    r"elaborate|summar(y|ise|ize)|explain\s+(it|this|the\s+(project|document|system|app)))\b", re.I)


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text)]


def _normalised(text: str) -> list[str]:
    """Words with common misspellings mapped to the vocabulary ("clerly" -> clearly, "explination" ->
    explanation, "allobrate" -> elaborate), so informal follow-ups are still recognised."""
    out = []
    for w in words(text):
        if w in _VOCAB or len(w) < 6:
            out.append(w)
        else:
            close = get_close_matches(w, _VOCAB, n=1, cutoff=0.8)
            out.append(close[0] if close else w)
    return out


def transforms(question: str) -> set[str]:
    ws = set(_normalised(question))
    text = question.lower()
    found = {kind for kind, vocab in TRANSFORMS.items() if ws & set(vocab)}
    if "go on" in text:
        found.add("continue")
    return found


def requested_language(question: str) -> str | None:
    for code, pattern in LANGUAGE_REQUEST:
        if pattern.search(question):
            return code
    return None


def is_follow_up(question: str, history: list) -> bool:
    """True if the question probably depends on the conversation: it refers back (it, this, the second ...),
    asks to transform the previous answer (explain, simplify, example, summarise, translate, continue), or is
    too short to stand alone. Only then is a rewrite (one extra model call) worth it."""
    if not any(role == "user" for role, *_ in history):
        return False
    ws = _normalised(question)
    if len(ws) <= 4:
        return True
    return bool(set(ws) & REFERENCES) or bool(transforms(question))


def is_broad(question: str) -> bool:
    return BROAD.search(question) is not None


# "this project", "the document", "the app": a subject the question names only by pointing at it.
POINTED = re.compile(r"\b(this|that|the|these|those)\s+(project|document|doc|file|pdf|app|application|system|"
                     r"platform|product|tool|report|policy|repo|repository|website|site|page)s?\b", re.I)
_POINTERS = {"it", "its", "this", "that", "these", "those", "they", "them", "their"}


def needs_subject(question: str) -> bool:
    """A question that does not say what it is about: it points (its features, this project) or asks about the
    whole thing (overview, architecture, main features). At the start of a chat there is no earlier message to
    say what "it" is, so the project's own source names are searched with it."""
    return bool(set(words(question)) & _POINTERS) or POINTED.search(question) is not None or is_broad(question)


def subject_query(question: str, titles: list[str]) -> str:
    """The question followed by the names of the project's sources ("BharatAgent Studio Build Handoff")."""
    names = []
    for t in titles:
        name = re.sub(r"\.(pdf|docx|md|markdown|txt|csv|xlsx)$", "", t.strip(), flags=re.I)
        name = re.sub(r"[_\s]+", " ", name.replace("-", " ")).strip()[:80]
        if name and name.lower() not in {n.lower() for n in names}:
            names.append(name)
    return f"{question} {'; '.join(names)}" if names else question


REWRITE_SYSTEM = (
    "You turn the user's latest message into ONE standalone search query for searching their documents.\n"
    "Rules:\n"
    "1. Replace words like it, this, that, they, 'the second one' with what they refer to in the conversation.\n"
    "2. Fix spelling mistakes and informal English.\n"
    "3. Drop instructions about the answer's form (in detail, simple words, five points, in Telugu, continue): "
    "keep only WHAT is being asked about.\n"
    "4. Write the query in English, at most 25 words. Do not answer it.\n"
    "5. If the latest message is about a new, unrelated topic, keep that topic and ignore the earlier ones.\n"
    "Reply with the query only, on one line, without quotes."
)


def rewrite_prompt(question: str, history: list) -> str:
    """The last few turns (bounded: 4 messages, 500 characters each) and the latest message."""
    lines = []
    for role, text, *_ in history[-4:]:
        lines.append(f"{'User' if role == 'user' else 'Assistant'}: {' '.join(text.split())[:500]}")
    return "Conversation so far:\n" + "\n".join(lines) + f"\n\nLatest message: {question}\n\nStandalone search query:"


def clean_rewrite(text: str, question: str) -> str | None:
    """The model's query, or None if it is unusable (empty, an apology, an answer instead of a query)."""
    text = "".join(ch if ch.isprintable() or ch in "\n" else " " for ch in text)  # no control characters
    line = next((ln.strip() for ln in text.strip().splitlines() if ln.strip()), "")
    line = re.sub(r"^(standalone\s+)?(search\s+)?query\s*:\s*", "", line, flags=re.I).strip().strip("\"'`")
    if not line or len(line) > 300 or line.lower() == question.lower().strip():
        return None if not line or len(line) > 300 else line
    if re.match(r"(?i)(i\s+(don'?t|do not|cannot|can'?t)|sorry)", line):
        return None
    return line
