"""
LLM-assisted topic-chapter segmentation.
Supports both .srt and plaintext transcripts.

Capabilities:
    1. Detect rough topical shifts
    2. Title & summarize each segment

Implementation:
    1. First splits transcript into rough "windows" based on time or every n lines
    2. Sends chunks to LLM to detect topic titles
"""

import re
from datetime import timedelta
from .llm_client import ollama_generate
from typing import Any

def _parse_srt(text: str) -> list[tuple[float, float, Any]]:
    """
    Parses an .srt string to a list of (start_seconds, end_seconds, text)
    """
    pattern = re.compile(
        r"(\d+)\s+(\d{2}:\d{2}:\d{2}[,.]\d{3})\s-->\s(\d{2}:\d{2}:\d{2}[,.]\d{3})\s+(.*?)\s*(?=\n\d+\n|\Z)",
        re.DOTALL
    )
    matches = pattern.findall(text)
    output = []
    for _, start_str, end_str, content in matches:
        start = _timestamp_to_seconds(start_str)
        end = _timestamp_to_seconds(end_str)
        output.append((start, end, content.strip().replace("\n", " ")))
    return output


def _timestamp_to_seconds(timestamp: str) -> float:
    """Convert a timestamp to its value in seconds."""
    timestamp = timestamp.replace(",", ".")
    hours, mins, secs = timestamp.split(":")
    sec = float(hours) * 3600 + float(mins) * 60 + float(secs)
    return sec


def _is_srt(text: str) -> bool:
    """Identify whether a file is an .srt file and return true iff the file contains the "-->" pattern
    prominent in .srt files."""
    return "-->" in text and re.search(r"\d{2}:\d{2}:\d{2}", text) is not None


def generate_chapters(path: str, model: str, max_tokens: int = 2048, min_chunk_seconds: int = 120) -> list[dict[str, str | Any]]:
    """
    Return a list of chapters.
    Each chapter contains the following keys:
        - "start": start time of this chapter
        - "end": end time of this chapter
        - "title": title of this chapter
        - "summary": highlights of this chapter's contents
    """

    with open(path, "r", encoding="utf-8") as f:
        t = f.read()

    # Parse the file as .srt if possible.
    if _is_srt(t):
        entries = _parse_srt(t)
    else:
        # Then the text is plaintext.
        # Treat each line as if it's worth 5s.
        entries = []
        for idx, line in enumerate(t.splitlines()):
            seconds = idx * 5  # Add a dummy 5s to the 'timestamp' of each new line.
            entries.append((seconds, seconds + 5, line.strip()))

    # Merge into coarse windows.
    windows = []
    current_chunk = ""
    current_start = entries[0][0] if entries else 0
    last_time = current_start

    for start, end, txt in entries:
        if (end - current_start >= min_chunk_seconds) and current_chunk:
            windows.append((current_start, last_time, current_chunk.strip()))
            current_chunk = ""
            current_start = start
        current_chunk += " " + txt
        last_time = end

    # Combine.
    if current_chunk:
        windows.append((current_start, last_time, current_chunk.strip()))

    output = []
    for (start, end, chunk) in windows:
        prompt = (
            "You are an assistant analyzing a transcript chunk.\n"
            "Chunk:\n"
            f"{chunk}\n\n"
            "1. Provide a short title (max 6 words) capturing the topic of this chunk.\n"
            "2. Provide a 1–2 sentence summary of what is being discussed."
            "Condition: Your response should be exactly in the form {chunk title}: {chunk summary}."
            "Condition: Return plaintext only. Do NOT use Markdown, bold, bullet points, or any special formatting.\n"
            "Condition: Do NOT provide anything else other than the short title and 1-2 sentence summary (e.g, no 'here are the requested outputs)"
        )
        response = ollama_generate(prompt=prompt, model=model, max_tokens=max_tokens)

        # Parse assuming the form Title: ...\nSummary: ..."
        lines = response.splitlines()
        title = lines[0].strip()
        summary = " ".join(l.strip() for l in lines[1:])
        output.append({"start": start, "end": end, "title": title, "summary": summary})

    return output
