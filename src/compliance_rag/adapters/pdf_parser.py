from __future__ import annotations

from pathlib import Path

import pymupdf

from compliance_rag.adapters.interfaces import ParsedDocument


class PyMuPdfParser:
    def __init__(self, image_dir: Path | None = None) -> None:
        self.image_dir = image_dir or Path("artifacts/images")

    def parse(self, path: Path) -> ParsedDocument:
        self.image_dir.mkdir(parents=True, exist_ok=True)
        document_id = path.stem
        pages: list[str] = []
        images: list[str] = []
        metadata: dict[str, object] = {}

        with pymupdf.open(path) as pdf:
            metadata["page_count"] = pdf.page_count
            metadata["title"] = pdf.metadata.get("title")
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                pages.append(f"<!-- page {page_number} -->\n{text}")
                for image in page.get_images(full=True):
                    xref = image[0]
                    pix = pymupdf.Pixmap(pdf, xref)
                    if pix.n - pix.alpha >= 4:
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    image_name = f"{document_id}_p{page_number}_{xref}.png"
                    image_path = self.image_dir / image_name
                    pix.save(image_path)
                    images.append(str(image_path))
                    pix = None

        return ParsedDocument(
            source_path=path,
            document_id=document_id,
            text="\n".join(pages),
            metadata=metadata,
            images=images,
        )
