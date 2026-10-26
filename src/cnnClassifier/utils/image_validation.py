"""Checks an uploaded file really is an acceptable image, before the model sees it.

The Content-Type header is written by the client, so it proves nothing. These
checks look at the actual bytes instead.
"""

import io

from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_PIXELS = 16_000_000  # e.g. 4000 x 4000; guards against "decompression bombs"
ALLOWED_FORMATS = {"JPEG", "PNG"}

# what Pillow raises for bytes that are not (or not fully) an image
_NOT_AN_IMAGE = (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError)


class ImageValidationError(Exception):
    """Raised for a bad upload. `status_code` is the HTTP status to answer with."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def validate_image(data: bytes) -> None:
    """Raise ImageValidationError unless `data` is a reasonably sized JPEG or PNG."""
    if not data:
        raise ImageValidationError("The uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageValidationError(
            f"File is too large (limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
            status_code=413,
        )

    # step 1: read only the header, which is cheap and tells us the format and size
    try:
        with Image.open(io.BytesIO(data)) as img:
            image_format = img.format
            width, height = img.size
    except _NOT_AN_IMAGE as e:
        raise ImageValidationError("File is not a valid image") from e

    if image_format not in ALLOWED_FORMATS:
        raise ImageValidationError("Only JPEG and PNG images are accepted")
    if width * height > MAX_PIXELS:
        raise ImageValidationError(
            f"Image dimensions are too large ({width}x{height})", status_code=422
        )

    # step 2: read the whole file to detect corruption or truncation
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except _NOT_AN_IMAGE as e:
        raise ImageValidationError("File is not a valid image") from e
