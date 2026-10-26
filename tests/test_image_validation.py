import pytest

from cnnClassifier.utils.image_validation import (
    MAX_UPLOAD_BYTES,
    ImageValidationError,
    validate_image,
)
from tests.conftest import image_bytes


def status_of(data: bytes) -> int:
    with pytest.raises(ImageValidationError) as error:
        validate_image(data)
    return error.value.status_code


def test_accepts_png():
    validate_image(image_bytes("PNG"))


def test_accepts_jpeg():
    validate_image(image_bytes("JPEG"))


def test_empty_file_is_400():
    assert status_of(b"") == 400


def test_random_bytes_are_400():
    assert status_of(b"definitely not an image") == 400


def test_gif_is_400():
    assert status_of(image_bytes("GIF")) == 400


def test_too_many_bytes_is_413():
    assert status_of(b"x" * (MAX_UPLOAD_BYTES + 1)) == 413


def test_too_many_pixels_is_422():
    assert status_of(image_bytes("PNG", size=(5000, 5000), mode="1")) == 422


def test_truncated_png_is_400():
    valid = image_bytes("PNG", size=(200, 200))
    assert status_of(valid[: len(valid) // 2]) == 400
