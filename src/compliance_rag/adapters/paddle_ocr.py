from __future__ import annotations

from pathlib import Path

from paddleocr import PaddleOCR

from compliance_rag.adapters.interfaces import TableResult


class PaddleOCRClient:
    def __init__(self) -> None:
        self.engine = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            use_gpu=False,
            show_log=False,
        )

    def extract_tables(self, image_path: Path, page: int) -> list[TableResult]:
        result = self.engine.ocr(str(image_path), cls=True)
        markdown = "\n".join(
            " | ".join(item[1][0] for item in line)
            for line in result or []
        )
        return [TableResult(page=page, markdown=markdown)]
