"""
Unit tests for IntentDiscovery.
"""

import pytest
from src.hiver_agent.nlp.clustering import IntentDiscovery


def test_intent_discovery_clusters():
    messages = [
        "My app keeps crashing when I open it on Windows 10",
        "Spotify app freezes on startup and exits immediately",
        "Desktop app crash on launch after latest update",
        "Music stops playing after 5 seconds every track",
        "Playback pauses automatically and shuffle is broken",
        "Song skips and repeat button does not work properly",
        "I was charged twice for Spotify Premium this month",
        "Why did my subscription renewal bill me 9.99 two times?",
        "Cancel my premium subscription billing issue",
        "Cannot download offline songs playlist greyed out",
        "Offline downloads not working on mobile device",
        "Sync offline music failed error message",
        "Cannot connect to bluetooth speaker in car",
        "CarPlay bluetooth connection dropping constantly",
        "Chromecast device not showing up on Spotify Connect",
        "Someone hacked my account and changed email address",
        "Cannot login password reset link not received",
        "Account access locked need security verification",
        "All my saved playlists and songs disappeared",
        "My library is empty where did my albums go?",
        "Playlist organization feature request",
        "Please add explicit lyrics toggle filter to settings",
        "Is Spotify down for everyone right now server error 500",
        "Spotify server outage service status down"
    ]

    discoverer = IntentDiscovery(n_clusters=4, random_state=42)
    results = discoverer.discover_clusters(messages)

    assert results["total_messages"] == len(messages)
    assert len(results["clusters"]) == 4
    for cluster in results["clusters"]:
        assert cluster["size"] > 0
        assert len(cluster["top_terms"]) > 0
