from qdrant_client.models import PointStruct

def format_points_for_llm(points: list) -> str:
    blocks = []

    for idx, p in enumerate(points, start=1):
        payload = p.payload

        book = payload.get("book_name", "Unknown Book")
        part = payload.get("part_name", "NA")
        chapter = payload.get("chapter_name", "Unknown Chapter")
        position = payload.get("chunk_number", "-")
        text = payload.get("chunk_content", "").strip()

        block = f"""
### Source {idx}
Book: {book}
Part: {part}
Chapter: {chapter}
Chunk Index: {position}

Content:
{text}
        """.strip()

        blocks.append(block)

    return "\n\n".join(blocks)