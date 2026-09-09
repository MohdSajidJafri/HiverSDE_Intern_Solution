"""
Text cleaning and normalization pipeline for customer support tweets.
Preserves semantic content while standardizing mentions, links, and whitespace.
"""

import re
import html
from typing import Dict, Any


class TextNormalizer:
    """Normalizes raw tweet text while retaining diagnostic signals."""

    HANDLE_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    WHITESPACE_PATTERN = re.compile(r"\s+")
    # Twitter artifact symbols and escaped entities
    HTML_ENTITY_PATTERN = re.compile(r"&[a-zA-Z0-9#]+;")

    def __init__(self, replace_urls_with: str = "http://link", strip_handles: bool = True):
        self.replace_urls_with = replace_urls_with
        self.strip_handles = strip_handles

    def normalize(self, text: str) -> str:
        """
        Cleans and normalizes text string.
        - Decodes HTML entities (&amp; -> &, &gt; -> >)
        - Normalizes Unicode spaces and control characters
        - Replaces URLs with placeholder token
        - Strips @mentions if requested
        - Collapses excess whitespace
        """
        if not text or not isinstance(text, str):
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Remove or replace URLs
        if self.replace_urls_with:
            text = self.URL_PATTERN.sub(self.replace_urls_with, text)
        else:
            text = self.URL_PATTERN.sub("", text)

        # Strip user/brand handles
        if self.strip_handles:
            text = self.HANDLE_PATTERN.sub("", text)

        # Collapse whitespace and strip
        text = self.WHITESPACE_PATTERN.sub(" ", text).strip()

        return text

    def extract_features(self, raw_text: str) -> Dict[str, Any]:
        """Extracts structural and conversational meta-features from raw text."""
        if not raw_text or not isinstance(raw_text, str):
            return {
                "has_question": False,
                "has_url": False,
                "has_mention": False,
                "char_length": 0,
                "word_count": 0,
                "has_exclamation": False,
                "normalized_text": "",
            }

        norm = self.normalize(raw_text)
        return {
            "has_question": "?" in raw_text,
            "has_url": bool(self.URL_PATTERN.search(raw_text)),
            "has_mention": bool(self.HANDLE_PATTERN.search(raw_text)),
            "char_length": len(raw_text),
            "word_count": len(norm.split()),
            "has_exclamation": "!" in raw_text,
            "normalized_text": norm,
        }
