import re
import json
from typing import Any, Dict
import pandas as pd

def parse_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a Discord style message dictionary.

    Extracts the message id, timestamp, content with any urls stripped, a
    list of urls and simplified embed metadata.
    """
    content = message.get("content", "") or ""
    urls = re.findall(r"https?://\S+", content)
    content_without_urls = re.sub(r"https?://\S+", "", content).strip()

    embeds_metadata = []
    for embed in message.get("embeds", []):
        metadata = {
            "url": embed.get("url"),
            "title": embed.get("title"),
            "description": embed.get("description"),
            "thumbnail_url": embed.get("thumbnail", {}).get("url") if embed.get("thumbnail") else None,
            "author_name": embed.get("author", {}).get("name") if embed.get("author") else None,
        }
        embeds_metadata.append(metadata)

    return {
        "id": message.get("id"),
        "timestamp": message.get("timestamp"),
        "content": content_without_urls,
        "urls": urls,
        "embeds": embeds_metadata,
    }


def create_combined_content(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``combined_content`` column containing ``embeds`` and ``content``."""
    df = df.copy()
    df["combined_content"] = df["embeds"].astype(str) + " " + df["content"].astype(str)
    return df
