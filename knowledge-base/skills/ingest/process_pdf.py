"""Convert a PDF file to Markdown and save image attachments."""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from marker.converters.pdf import PdfConverter  # type: ignore[import-not-found]
from marker.models import create_model_dict  # type: ignore[import-not-found]
from marker.output import text_from_rendered  # type: ignore[import-not-found]

_PDF_SUFFIX = ".pdf"


def convert_pdf(pdf_path: Path, output_dir: Path) -> None:
    """Convert a PDF to Markdown and save image attachments.

    Parameters
    ----------
    pdf_path : Path
        Path to the input PDF file.
    output_dir : Path
        Directory where the ``.md`` file and ``attachments/`` folder are written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    models = create_model_dict()
    converter = PdfConverter(artifact_dict=models)
    rendered = converter(str(pdf_path))
    text, _ext, images = text_from_rendered(rendered)

    text = re.sub(r"!\[\]\(([^/)]+\.jpeg)\)", r"![](attachments/\1)", text)

    # 1. Normalize line endings (convert Windows \r\n to Unix \n)
    text = text.replace("\r\n", "\n")

    # 2. Remove invisible trailing spaces at the end of lines
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    md_path = output_dir / f"{pdf_path.stem}.md"
    md_path.write_text(text, encoding="utf-8")
    if images:
        attachments_dir = output_dir / "attachments"
        attachments_dir.mkdir(exist_ok=True)
        for img_name, img in images.items():
            img.save(attachments_dir / img_name)


def llm_markdown_correction(path: Path) -> None:
    """Let gemini correct formatting problems with extracted md."""
    if path.exists():
        cmd = [
            "gemini",
            "--prompt",
            f"The markdown ({path.name}) was extracted from a pdf and therefore has some formatting issues. Reorder blocks if necessary and fix the latex math formulas.",
            "--yolo",
        ]
        subprocess.run(cmd, cwd=path.parent, shell=True, stdout=sys.stdout, stderr=sys.stderr, timeout=600, check=False)  # noqa: S602


def main() -> None:
    """Run the PDF-to-Markdown CLI."""
    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown and save image attachments.",
    )
    parser.add_argument("pdf", type=Path, help="Path to the input PDF file.")

    args = parser.parse_args()
    pdf_path: Path = args.pdf
    output_dir: Path = pdf_path.parent / pdf_path.stem
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)  # noqa: T201
        sys.exit(1)
    if pdf_path.suffix.lower() != _PDF_SUFFIX:
        print(  # noqa: T201
            f"Error: expected a {_PDF_SUFFIX} file, got: {pdf_path.suffix}",
            file=sys.stderr,
        )
        sys.exit(1)
    convert_pdf(pdf_path, output_dir)

    shutil.move(pdf_path, output_dir / pdf_path.name)

    llm_markdown_correction((output_dir / pdf_path.name).with_suffix(".md").absolute())

    print(f"Converted: {pdf_path.name} -> {output_dir}")  # noqa: T201


if __name__ == "__main__":
    main()
