from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    text: str
    start: int
    end: int


def split_parent_child(
    text: str,
    parent_id: str,
    parent_token_limit: int = 512,
    child_token_limit: int = 256,
    child_overlap_tokens: int = 50,
) -> list[Chunk]:
    """Create parent and child chunks from a semantic segment.

    This is a simple length-based implementation. The real pipeline should
    replace it with token-aware semantic splitting.
    """
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    cursor = 0
    text_len = len(text)
    child_index = 0

    while cursor < text_len:
        end = min(cursor + child_token_limit, text_len)
        child = text[cursor:end]
        chunks.append(
            Chunk(
                chunk_id=f"{parent_id}:child:{child_index}",
                parent_id=parent_id,
                text=child,
                start=cursor,
                end=end,
            )
        )
        child_index += 1
        if end >= text_len:
            break
        cursor = max(cursor + child_token_limit - child_overlap_tokens, cursor + 1)

    return chunks


def extract_parent_chunks(
    segments: list[tuple[str, str]],
    parent_token_limit: int = 512,
) -> list[tuple[str, str]]:
    """Return parent chunks for semantic segments."""
    return [(segment_id, text) for segment_id, text in segments if text.strip()]
