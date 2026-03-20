import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import emoji
    _HAS_EMOJI = True
except Exception:
    _HAS_EMOJI = False

try:
    from dictionary import dictionary as EXTERNAL_DICTIONARY  # type: ignore
    if not isinstance(EXTERNAL_DICTIONARY, dict):
        print("Warning: imported 'dictionary' is not a dict. Using empty mapping.")
        EXTERNAL_DICTIONARY = {}
except Exception:
    EXTERNAL_DICTIONARY = {}

# offensive_words.py -> variable: offensive_words (list)
try:
    from offensive_words import offensive_words as EXTERNAL_OFFENSIVE_WORDS  # type: ignore
    if not isinstance(EXTERNAL_OFFENSIVE_WORDS, (list, tuple, set)):
        print("Warning: imported 'offensive_words' is not a list/tuple/set. Using empty list.")
        EXTERNAL_OFFENSIVE_WORDS = []
    EXTERNAL_OFFENSIVE_WORDS = list(EXTERNAL_OFFENSIVE_WORDS)
except Exception:
    EXTERNAL_OFFENSIVE_WORDS = []

# slangs.txt -> parsed into SLANGS_MAP
SLANGS_FILE = Path("slangs.txt")


def load_slangs_file(path: Path) -> Dict[str, str]:
    slangs: Dict[str, str] = {}
    if not path.exists():
        return slangs
    with path.open("r", encoding="utf8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            for sep in ("\t", "=>", ":", "="):
                if sep in line:
                    k, v = line.split(sep, 1)
                    slangs[k.strip().casefold()] = v.strip().casefold()
                    break
            else:
                slangs[line.casefold()] = ""
    return slangs


SLANGS_MAP = load_slangs_file(SLANGS_FILE)


try:
    from location import places as EXTERNAL_PLACES  # type: ignore
    if not isinstance(EXTERNAL_PLACES, (list, tuple, set)):
        print("Warning: imported 'places' is not a list/tuple/set. Using empty list.")
        EXTERNAL_PLACES = []
    EXTERNAL_PLACES = list(EXTERNAL_PLACES)
except Exception:
    EXTERNAL_PLACES = []


DICT_MAP_RAW: Dict[str, str] = {}
for k, v in EXTERNAL_DICTIONARY.items():
    if not isinstance(k, str):
        continue
    val = v if isinstance(v, str) else str(v)
    DICT_MAP_RAW[k.casefold()] = val.casefold()

# Build sets
OFFENSIVE_SET = {w.casefold() for w in EXTERNAL_OFFENSIVE_WORDS if isinstance(w, str)}

# Prepare place replacers: map any listed place -> "<location>"
PLACES_LIST = [p for p in (str(p) for p in EXTERNAL_PLACES) if p]
PLACES_CASEFOLDED = [p.casefold() for p in PLACES_LIST]


# --- Helpers ----------------------------------------------------------------

def normalize_nfkc(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text)


def normalize_casefold(text: str) -> str:
    return text.casefold()


def build_replacer_for_word_key(k: str, v: str) -> Tuple[re.Pattern, str]:
    pat = re.compile(r"\b" + re.escape(k) + r"\b", flags=re.IGNORECASE)
    return pat, v


def build_replacer_for_nonword_key(k: str, v: str) -> Tuple[re.Pattern, str]:
    pat = re.compile(r"(?<!\w)" + re.escape(k) + r"(?!\w)", flags=re.IGNORECASE)
    return pat, v


def split_and_build_dict_replacers(mapping: Dict[str, str]) -> Tuple[List[Tuple[re.Pattern, str]], List[Tuple[re.Pattern, str]]]:
    items = sorted(mapping.items(), key=lambda kv: -len(kv[0]))
    nonword_replacers: List[Tuple[re.Pattern, str]] = []
    word_replacers: List[Tuple[re.Pattern, str]] = []
    for k, v in items:
        if re.fullmatch(r'\w+', k):  # purely word chars
            word_replacers.append(build_replacer_for_word_key(k, v))
        else:
            nonword_replacers.append(build_replacer_for_nonword_key(k, v))
    return nonword_replacers, word_replacers


def build_place_replacers(places: List[str], tag: str = "<location>") -> List[Tuple[re.Pattern, str]]:
    unique = sorted({p for p in places if p}, key=lambda x: -len(x))
    replacers: List[Tuple[re.Pattern, str]] = []
    for p in unique:
        pat = re.compile(r"\b" + re.escape(p) + r"\b", flags=re.IGNORECASE)
        replacers.append((pat, tag))
    return replacers


# Build replacers
NONWORD_DICT_REPLACERS, WORD_DICT_REPLACERS = split_and_build_dict_replacers(DICT_MAP_RAW)


def build_word_replacer(mapping: Dict[str, str]) -> List[Tuple[re.Pattern, str]]:
    items = sorted(mapping.items(), key=lambda kv: -len(kv[0]))
    replacers: List[Tuple[re.Pattern, str]] = []
    for k, v in items:
        if not k:
            continue
        if re.fullmatch(r'\w+', k):
            pat = re.compile(r"\b" + re.escape(k) + r"\b", flags=re.IGNORECASE)
        else:
            pat = re.compile(r"(?<!\w)" + re.escape(k) + r"(?!\w)", flags=re.IGNORECASE)
        replacers.append((pat, v))
    return replacers


SLANG_REPLACERS = build_word_replacer(SLANGS_MAP)
PLACES_REPLACERS = build_place_replacers(PLACES_CASEFOLDED, tag="<location>")


# --- Processing functions ---------------------------------------------------

def mask_offensive_words(text: str, offensive_set: set, mask_token: str = "<offensive>") -> str:
    if not offensive_set:
        return text
    words = sorted(offensive_set, key=lambda x: -len(x))
    escaped = [re.escape(w) for w in words if w]
    if not escaped:
        return text
    pattern = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)
    return pattern.sub(mask_token, text)


def apply_replacers(text: str, replacers: List[Tuple[re.Pattern, str]]) -> str:
    for pat, repl in replacers:
        text = pat.sub(repl, text)
    return text


# ----------------- NEW: Devanagari removal helpers --------------------------

def contains_devanagari(token: str) -> bool:
    return bool(re.search(r'[\u0900-\u097F]', token))


def remove_devanagari_tokens(text: str) -> str:
    if not isinstance(text, str):
        return ""
    tokens = text.split()
    kept = []
    for tok in tokens:
        stripped = re.sub(r"^[^\w']+|[^\w']+$", "", tok)
        if not stripped:
            continue
        if contains_devanagari(stripped):
            continue
        kept.append(stripped)
    return " ".join(kept)


def replace_numbers(text: str, number_tag: str = "<number>") -> str:
    if not isinstance(text, str):
        return text
    text = re.sub(r'\b\d{1,3}(?:[,\.\d]*\d)?\b', number_tag, text)
    text = re.sub(r'[\u0966-\u096F]+(?:[,\.\u0966-\u096F]*[\u0966-\u096F]+)*', number_tag, text)
    return text


# --- The pipeline -----------------------------------------------------------

def preprocessing_pipeline(text: str) -> str:
   
    if not isinstance(text, str):
        return ""

    text = normalize_nfkc(text)
    text = normalize_casefold(text)

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r';', '', text)

    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # aaa -> aa
    text = re.sub(r'\b(\w+)( \1){2,}\b', r'\1', text)  # reduce repeated words (3+ -> 1)

    if NONWORD_DICT_REPLACERS:
        text = apply_replacers(text, NONWORD_DICT_REPLACERS)

    if _HAS_EMOJI:
        text = emoji.demojize(text, delimiters=(" ", " "))
        # convert :emoji_name: -> emoji_name (so word replacers can match)
        text = re.sub(r":([a-z0-9_+-]+):", r"\1", text, flags=re.IGNORECASE)
    else:
        text = re.sub(
            r"[\U0001F600-\U0001F64F"
            r"\U0001F300-\U0001F5FF"
            r"\U0001F680-\U0001F6FF"
            r"\U0001F1E0-\U0001F1FF"
            r"]+", "", text)

    if PLACES_REPLACERS:
        text = apply_replacers(text, PLACES_REPLACERS)

    text = replace_numbers(text, number_tag="<number>")

    text = remove_devanagari_tokens(text)

    if WORD_DICT_REPLACERS:
        text = apply_replacers(text, WORD_DICT_REPLACERS)

    if SLANG_REPLACERS:
        text = apply_replacers(text, SLANG_REPLACERS)

    if OFFENSIVE_SET:
        text = mask_offensive_words(text, OFFENSIVE_SET, mask_token="<offensive>")

    text = re.sub(r"[^\w\s']", '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


if __name__ == "__main__":
    samples = [
        "I have 123 apples and 1,234.56 rupees",
        "नेपाली अंक १२३४५ र mixed १२३ and 456",
        "This was a a a aaaa good experience :)",
        "I am so happy ^_^ and love you <3 😂",
        "Mixed: hello म sathii :)",              
    ]
    for s in samples:
        processed = preprocessing_pipeline(s)
        print("ORIGINAL:", s)
        print("PROCESSED:", processed)
        print("-" * 60)
