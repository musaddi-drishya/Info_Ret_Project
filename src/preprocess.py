"""Text preprocessing: tokenization, stopword removal and Porter stemming.

Everything is implemented from scratch (no NLTK / spaCy) so that every step
of the pipeline is visible and reproducible.
"""
import re
import unicodedata
from functools import lru_cache

# SMART-style English stopword list (trimmed to common function words).
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because been before being below
between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each
few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself
him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't
my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll
we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself yourselves also may might must shall will
one two us via etc ie eg re subject write wrote writes article
""".split())

TOKEN_RE = re.compile(r"[a-z][a-z0-9]+")


def tokenize(text):
    """NFKC-normalise (e.g. Unicode 'math bold' letters used in campus posts -> ASCII), lowercase,
    and split into alphanumeric tokens starting with a letter (len >= 2)."""
    return TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower())


class PorterStemmer:
    """Porter (1980) stemming algorithm, steps 1a-5b."""

    VOWELS = frozenset("aeiou")

    def _cons(self, w, i):
        ch = w[i]
        if ch in self.VOWELS:
            return False
        if ch == "y":
            return i == 0 or not self._cons(w, i - 1)
        return True

    def _m(self, stem):
        """Measure: number of VC sequences in the stem."""
        n, i, L = 0, 0, len(stem)
        while i < L and self._cons(stem, i):
            i += 1
        while i < L:
            while i < L and not self._cons(stem, i):
                i += 1
            if i >= L:
                break
            n += 1
            while i < L and self._cons(stem, i):
                i += 1
        return n

    def _has_vowel(self, stem):
        return any(not self._cons(stem, i) for i in range(len(stem)))

    def _double_c(self, w):
        return len(w) >= 2 and w[-1] == w[-2] and self._cons(w, len(w) - 1)

    def _cvc(self, w):
        if len(w) < 3:
            return False
        if not (self._cons(w, len(w) - 3) and not self._cons(w, len(w) - 2) and self._cons(w, len(w) - 1)):
            return False
        return w[-1] not in "wxy"

    def _replace(self, w, suffix, repl, min_m):
        if w.endswith(suffix):
            stem = w[: len(w) - len(suffix)]
            if self._m(stem) > min_m:
                return stem + repl, True
            return w, True
        return w, False

    def _step1ab(self, w):
        if w.endswith("sses"):
            w = w[:-2]
        elif w.endswith("ies"):
            w = w[:-2]
        elif w.endswith("ss"):
            pass
        elif w.endswith("s"):
            w = w[:-1]

        if w.endswith("eed"):
            if self._m(w[:-3]) > 0:
                w = w[:-1]
            return w
        for suf in ("ed", "ing"):
            if w.endswith(suf) and self._has_vowel(w[: -len(suf)]):
                w = w[: -len(suf)]
                if w.endswith(("at", "bl", "iz")):
                    w += "e"
                elif self._double_c(w) and w[-1] not in "lsz":
                    w = w[:-1]
                elif self._m(w) == 1 and self._cvc(w):
                    w += "e"
                return w
        return w

    def _step1c(self, w):
        if w.endswith("y") and self._has_vowel(w[:-1]):
            return w[:-1] + "i"
        return w

    STEP2 = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"),
             ("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
             ("ization", "ize"), ("ation", "ate"), ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
             ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
             ("logi", "log")]
    STEP3 = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"),
             ("ful", ""), ("ness", "")]
    STEP4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion",
             "ou", "ism", "ate", "iti", "ous", "ive", "ize"]

    def _step2(self, w):
        for suf, rep in self.STEP2:
            w2, hit = self._replace(w, suf, rep, 0)
            if hit:
                return w2
        return w

    def _step3(self, w):
        for suf, rep in self.STEP3:
            w2, hit = self._replace(w, suf, rep, 0)
            if hit:
                return w2
        return w

    def _step4(self, w):
        for suf in self.STEP4:
            if w.endswith(suf):
                stem = w[: -len(suf)]
                if suf == "ion" and not (stem and stem[-1] in "st"):
                    continue
                if self._m(stem) > 1:
                    return stem
                return w
        return w

    def _step5(self, w):
        if w.endswith("e"):
            stem = w[:-1]
            m = self._m(stem)
            if m > 1 or (m == 1 and not self._cvc(stem)):
                w = stem
        if self._m(w) > 1 and self._double_c(w) and w.endswith("l"):
            w = w[:-1]
        return w

    def stem(self, w):
        if len(w) <= 2:
            return w
        w = self._step1ab(w)
        w = self._step1c(w)
        w = self._step2(w)
        w = self._step3(w)
        w = self._step4(w)
        return self._step5(w)


_stemmer = PorterStemmer()


@lru_cache(maxsize=200_000)
def stem(word):
    return _stemmer.stem(word)


def preprocess(text, use_stemming=True, remove_stopwords=True):
    """Full pipeline: text -> list of index terms."""
    toks = tokenize(text)
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    if use_stemming:
        toks = [stem(t) for t in toks]
    return toks
