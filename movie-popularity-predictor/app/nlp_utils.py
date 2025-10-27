"""Utilities for ensuring required NLTK resources are available.

This module centralises the logic for downloading and loading the
WordNet lemmatiser and English stop-word list.  Both the training
script and the FastAPI application import these helpers so that we
have a single place responsible for verifying the corpora exist.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Set, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_REQUIRED_CORPORA = ("wordnet", "stopwords")

_FALLBACK_STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
}


class _IdentityLemmatizer:
    def lemmatize(self, token: str) -> str:
        return token


def ensure_nltk_data() -> None:
    """Ensure the corpora needed by the project are available.

    If the resources are missing (for example in a fresh Codespaces
    environment) they are downloaded quietly so that the rest of the
    application can continue to start up without manual intervention.
    """

    for corpus in _REQUIRED_CORPORA:
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            try:
                nltk.download(corpus, quiet=True)
            except Exception:
                # Offline environments (e.g. CI, Codespaces without public
                # internet) can raise errors during the download.  We swallow
                # them so that the fallback logic in ``get_language_tools``
                # can take over.
                continue


@lru_cache(maxsize=1)
def get_language_tools() -> Tuple[WordNetLemmatizer, Set[str]]:
    """Return a shared WordNet lemmatiser and stop-word set."""

    ensure_nltk_data()

    try:
        lemmatiser = WordNetLemmatizer()
        # Accessing WordNet lazily; if the corpus is missing this will raise
        # a LookupError that we handle below.
        lemmatiser.lemmatize("test")
    except LookupError:
        lemmatiser = _IdentityLemmatizer()

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set(_FALLBACK_STOPWORDS)

    return lemmatiser, stop_words

