"""Typo-tolerant keyword matching against the tag vocabulary (rapidfuzz).

Each keyword is expanded to the database tags whose similarity with it is at
least a cutoff. The similarity is rapidfuzz `fuzz.ratio` (0-100, from the Indel
edit distance). The cutoff follows the fuzziness slider f in [0, 1]:
    cutoff = 100 - f * (100 - FUZZY_MIN_TAG_SIMILARITY)
so f = 0 keeps exact tags only and f = 1 accepts e.g. 'sunst' -> 'sunset' (91)
or 'beech' -> 'beach' (80).
"""
from rapidfuzz import fuzz, process

from .. import config
from .database import normalize_keyword


class TagMatcher:
    def __init__(self, conn):
        self.vocab = [row[0] for row in conn.execute("SELECT DISTINCT tag FROM tags")]
        self.vocab_set = set(self.vocab)

    def expand(self, keyword, fuzziness):
        """Return {tag: similarity} for one keyword (exact match first when it exists)."""
        word = normalize_keyword(keyword)
        if not word:
            return {}
        exact = {word: 100.0} if word in self.vocab_set else {}
        if fuzziness <= 0:
            return exact
        cutoff = 100 - fuzziness * (100 - config.FUZZY_MIN_TAG_SIMILARITY)
        matches = process.extract(word, self.vocab, scorer=fuzz.ratio, score_cutoff=cutoff,
                                  limit=config.FUZZY_MAX_EXPANSIONS)
        return {**exact, **{tag: float(score) for tag, score, _ in matches if tag not in exact}}
