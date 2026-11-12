"""Word-internal features: characters, hashed prefixes/suffixes, and word shape.

Kinyarwanda marks noun class and verb agreement with prefixes (umu-, aba-, igi-, ibi-,
ku-, gu-, ba-, ya- ...) and tense/aspect with endings (-ye, -ze, -ga, -a), so the
first and last few characters carry most of the POS signal for unseen words.
"""
import unicodedata
import zlib

CHAR_PAD, CHAR_UNK, CHAR_BOW, CHAR_EOW = 0, 1, 2, 3
MAX_WORD_CHARS = 32          # long words keep their first and last 15 characters
PREFIX_LENS = (1, 2, 3, 4, 5)
SUFFIX_LENS = (1, 2, 3, 4)
AFFIX_BUCKETS = 200003       # hashed; id 0 is padding
N_AFFIX = len(PREFIX_LENS) + len(SUFFIX_LENS)
N_SHAPES = 9                 # 0 is padding


def build_char_vocab(sentence_lists):
    chars = sorted({c for sents in sentence_lists for s in sents for w in s["words"] for c in w})
    vocab = {"<pad>": CHAR_PAD, "<unk>": CHAR_UNK, "<bow>": CHAR_BOW, "<eow>": CHAR_EOW}
    for c in chars:
        vocab.setdefault(c, len(vocab))
    return vocab


def char_ids(word, vocab):
    if len(word) > MAX_WORD_CHARS - 2:
        half = (MAX_WORD_CHARS - 2) // 2
        word = word[:half] + word[-half:]
    return [CHAR_BOW] + [vocab.get(c, CHAR_UNK) for c in word] + [CHAR_EOW]


def _bucket(key):
    return 1 + zlib.crc32(key.encode("utf-8")) % (AFFIX_BUCKETS - 1)


def affix_ids(word):
    w = word.lower()
    ids = []
    for n in PREFIX_LENS:
        ids.append(_bucket(f"p{n}:{w[:n]}" if len(w) >= n else f"p{n}:<short>"))
    for n in SUFFIX_LENS:
        ids.append(_bucket(f"s{n}:{w[-n:]}" if len(w) >= n else f"s{n}:<short>"))
    return ids


def shape_id(word):
    if all(unicodedata.category(c).startswith(("P", "S")) for c in word):
        return 1                                    # punctuation / symbol
    if all(c.isdigit() or c in ".,:/-" for c in word) and any(c.isdigit() for c in word):
        return 2                                    # number
    if any(c.isdigit() for c in word):
        return 3                                    # mixed with digits (DC10, 2020s)
    if "@" in word or word.lower().startswith(("http", "www.")):
        return 4                                    # email / url
    if len(word) > 1 and word.isupper():
        return 5                                    # ACRONYM
    if word[0].isupper():
        return 6                                    # Capitalized
    if word.islower():
        return 7                                    # lowercase
    return 8                                        # other
