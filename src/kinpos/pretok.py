"""Split raw Kinyarwanda text the way MasakhaPOS does.

- curly quotes/apostrophes are normalized to ' and "
- contractions are split: n'abasaga -> n ' abasaga, y'agaciro -> y ' agaciro
- every punctuation/symbol character is its own token (1,2 -> 1 , 2; 10% -> 10 %)
- URLs and e-mail addresses stay whole (tagged SYM by the guideline)
"""
import re

_NORMALIZE = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "“": '"', "”": '"', "«": '"', "»": '"',
                            "–": "-", "—": "-"})
_TOKEN = re.compile(
    r"(?:https?://|www\.)\S+[^\s.,;:!?)\]\"']"      # URL
    r"|[\w.+-]+@[\w-]+\.[\w.-]*\w"                   # e-mail
    r"|\w+"                                          # word / number
    r"|[^\w\s]",                                     # any single punctuation / symbol
    re.UNICODE)


def pretokenize(text):
    return _TOKEN.findall(" ".join(text.translate(_NORMALIZE).split()))
