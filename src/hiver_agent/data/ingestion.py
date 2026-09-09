"""
Data ingestion module for Customer Support on Twitter (TWCS).
Supports streaming range requests, chunked reading, and local caching.
"""

import io
import os
import urllib.request
from pathlib import Path
from typing import Optional, Generator
import pandas as pd
from tqdm import tqdm


class DatasetIngestion:
    """Manages downloading, streaming, and caching of the TWCS dataset."""

    DEFAULT_URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"

    def __init__(self, cache_dir: Path = Path("data/raw")):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_csv_path = self.cache_dir / "twcs.csv"

    def stream_sample_bytes(self, max_bytes: int = 31457280, url: Optional[str] = None) -> bytes:
        """
        Streams a byte chunk from the remote LFS endpoint via HTTP Range header.
        Default 30MB allows inspecting ~170,000 tweets quickly.
        """
        target_url = url or self.DEFAULT_URL
        headers = {"Range": f"bytes=0-{max_bytes}"}
        req = urllib.request.Request(target_url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        # Cut to last newline to preserve complete CSV records
        last_nl = data.rfind(b"\n")
        return data[:last_nl] if last_nl != -1 else data

    def load_sample_df(self, max_bytes: int = 31457280) -> pd.DataFrame:
        """Loads a representative byte sample directly into a pandas DataFrame."""
        if self.local_csv_path.exists():
            # Read first N rows from local file
            return pd.read_csv(self.local_csv_path, nrows=100000, on_bad_lines="skip")
        
        raw_bytes = self.stream_sample_bytes(max_bytes=max_bytes)
        return pd.read_csv(io.BytesIO(raw_bytes), on_bad_lines="skip")

    def download_sample_to_cache(self, max_bytes: int = 52428800, filename: str = "twcs_sample.csv") -> Path:
        """Downloads a 50MB sample to local cache for fast offline access."""
        out_path = self.cache_dir / filename
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        
        raw_bytes = self.stream_sample_bytes(max_bytes=max_bytes)
        with open(out_path, "wb") as f:
            f.write(raw_bytes)
        return out_path
