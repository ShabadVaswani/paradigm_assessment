"""Tools the answering model can call.

`get_image` lets the model look at a handbook image that a retrieved chunk refers to (`[Image img-..., page N]`).
The chunk already carries the image's description; the tool is for when the model needs the picture itself, e.g.
to read a diagram, a screenshot or a chart more closely than the description allows.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from .images import ImageRecord, load_image_index

GET_IMAGE_TOOL = {
    "name": "get_image",
    "description": (
        "Fetch an image from the PostHog handbook by its ID. Retrieved handbook excerpts mention images as "
        "'[Image <image_id>, page N] <description>'. Call this when answering depends on what the image shows and "
        "the description is not enough, for example to read exact labels, steps, numbers or a diagram's layout. "
        "Returns the image together with its page, article, section, the excerpt it belongs to, and its stored "
        "description."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "image_id": {"type": "string", "description": "The image ID from the excerpt, e.g. img-p0041-x258."},
        },
        "required": ["image_id"],
        "additionalProperties": False,
    },
}


class ImageStore:
    """Image records by ID, loaded from the index written by `ingest chunk`."""

    def __init__(self, index_path: Path, root: Path) -> None:
        self.root = root
        self.records = {r.image_id: r for r in load_image_index(index_path)}

    def get(self, image_id: str) -> ImageRecord | None:
        return self.records.get(image_id)


def run_get_image(store: ImageStore, tool_input: dict) -> tuple[list[dict], bool]:
    """Execute a `get_image` call. Returns (tool_result content blocks, is_error)."""
    image_id = str(tool_input.get("image_id", "")).strip()
    record = store.get(image_id)
    if record is None:
        return [{"type": "text", "text": f"No image with ID {image_id!r}. Use an ID exactly as it appears in an "
                                         "excerpt, e.g. img-p0041-x258."}], True
    path = store.root / record.file
    if not path.exists():
        return [{"type": "text", "text": f"Image {image_id} is indexed but its file is missing; "
                                         "run `python -m handbook_qa.ingest extract`."}], True

    meta = {
        "image_id": record.image_id,
        "appears_in": [{"page": o.page, "article": o.article_title, "source": o.article_path,
                        "section": " > ".join(o.heading_path), "chunk_id": o.chunk_id} for o in record.occurrences],
        "caption": record.description["caption"] if record.description else None,
        "description": record.description["description"] if record.description else None,
    }
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": json.dumps(meta, ensure_ascii=False)},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
    ], False
