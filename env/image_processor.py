"""
Optional OCR image processor.

If Tesseract (pytesseract + Pillow) is installed, this module converts
an image file or base64-encoded image into OCR text that the environment
can consume as ``ocr_text``.

If Tesseract is NOT installed, all public functions raise
``OCRNotAvailableError`` with a helpful installation message.
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore

    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


class OCRNotAvailableError(RuntimeError):
    """Raised when pytesseract / Pillow are not installed."""

    _MESSAGE = (
        "OCR support requires Tesseract + Python bindings. Install with:\n"
        "  pip install pytesseract Pillow\n"
        "  # Then install Tesseract binary:\n"
        "  # Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
        "  # Ubuntu:  sudo apt-get install tesseract-ocr\n"
        "  # macOS:   brew install tesseract"
    )

    def __init__(self) -> None:
        super().__init__(self._MESSAGE)


def is_ocr_available() -> bool:
    """Return True if pytesseract + Pillow are importable."""
    return _OCR_AVAILABLE


def ocr_from_file(path: str | Path, language: str = "eng") -> str:
    """
    Run Tesseract OCR on an image file and return extracted text.

    Args:
        path: Path to the image file (JPEG, PNG, TIFF, etc.).
        language: Tesseract language code (default: 'eng').

    Returns:
        Extracted text string.

    Raises:
        OCRNotAvailableError: if pytesseract / Pillow are not installed.
        FileNotFoundError: if the image file does not exist.
    """
    if not _OCR_AVAILABLE:
        raise OCRNotAvailableError()

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    image = Image.open(file_path)
    text: str = pytesseract.image_to_string(image, lang=language)
    logger.debug("OCR extracted %d characters from %s", len(text), file_path)
    return text.strip()


def ocr_from_base64(
    data: str,
    mime_type: str = "image/jpeg",
    language: str = "eng",
) -> str:
    """
    Run Tesseract OCR on a base64-encoded image.

    Args:
        data: Base64-encoded image content (without data: URI prefix).
        mime_type: MIME type hint (informational; Pillow auto-detects format).
        language: Tesseract language code (default: 'eng').

    Returns:
        Extracted text string.

    Raises:
        OCRNotAvailableError: if pytesseract / Pillow are not installed.
    """
    if not _OCR_AVAILABLE:
        raise OCRNotAvailableError()

    try:
        raw_bytes = base64.b64decode(data)
    except Exception as exc:
        raise ValueError(f"Failed to decode base64 image data: {exc}") from exc

    image = Image.open(io.BytesIO(raw_bytes))
    text: str = pytesseract.image_to_string(image, lang=language)
    logger.debug("OCR extracted %d characters from base64 payload (%s)", len(text), mime_type)
    return text.strip()


def ocr_from_image_input(image_input: object, language: str = "eng") -> Optional[str]:
    """
    Dispatch OCR based on an ``ImageInput`` model.

    Args:
        image_input: An ``models.schemas.ImageInput`` instance.
        language: Tesseract language code.

    Returns:
        Extracted text or None if no input provided.
    """
    # Lazy import to avoid circular imports
    from models.schemas import ImageInput  # noqa: PLC0415

    assert isinstance(image_input, ImageInput)

    if image_input.base64_data:
        return ocr_from_base64(image_input.base64_data, image_input.mime_type, language)
    if image_input.file_path:
        return ocr_from_file(image_input.file_path, language)
    return None
