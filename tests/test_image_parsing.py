"""
Image Parsing Tests (Bonus)

Tests for image_parser.py:
- Image validation (format, size)
- Base64 encoding
- MIME type detection
- ParsedContractPage structure validation

Note: Actual LLM parsing tests require API keys and are marked as integration tests.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_parser import (
    validate_image,
    encode_image,
    get_image_media_type,
    SUPPORTED_FORMATS,
    MAX_IMAGE_SIZE_MB
)
from models import ParsedContractPage, Section, Clause


class TestImageValidation:
    """Tests for image validation functions."""

    def test_supported_formats(self):
        """Test that all expected formats are supported."""
        expected_formats = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        assert SUPPORTED_FORMATS == expected_formats

    def test_max_image_size(self):
        """Test that max image size is set correctly."""
        assert MAX_IMAGE_SIZE_MB == 20  # OpenAI limit

    def test_validate_nonexistent_file(self):
        """Test validation fails for nonexistent file."""
        is_valid, error_msg = validate_image("/nonexistent/path/image.png")
        
        assert is_valid is False
        assert "not found" in error_msg.lower()

    def test_validate_unsupported_format(self, tmp_path):
        """Test validation fails for unsupported format."""
        # Create a fake file with unsupported extension
        fake_file = tmp_path / "document.pdf"
        fake_file.write_text("fake content")
        
        is_valid, error_msg = validate_image(str(fake_file))
        
        assert is_valid is False
        assert "unsupported" in error_msg.lower()

    def test_validate_valid_png(self, tmp_path):
        """Test validation passes for valid PNG file."""
        # Create a small fake PNG file
        png_file = tmp_path / "test.png"
        # PNG magic bytes + minimal data
        png_file.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
        
        is_valid, error_msg = validate_image(str(png_file))
        
        assert is_valid is True
        assert error_msg == ""

    def test_validate_valid_jpeg(self, tmp_path):
        """Test validation passes for valid JPEG file."""
        jpeg_file = tmp_path / "test.jpg"
        jpeg_file.write_bytes(b'\xff\xd8\xff' + b'\x00' * 100)
        
        is_valid, error_msg = validate_image(str(jpeg_file))
        
        assert is_valid is True

    def test_validate_file_too_large(self, tmp_path):
        """Test validation fails for files exceeding size limit."""
        large_file = tmp_path / "large.png"
        # Create file larger than MAX_IMAGE_SIZE_MB
        large_file.write_bytes(b'\x89PNG' + b'\x00' * (21 * 1024 * 1024))  # 21 MB
        
        is_valid, error_msg = validate_image(str(large_file))
        
        assert is_valid is False
        assert "too large" in error_msg.lower()


class TestImageEncoding:
    """Tests for image encoding functions."""

    def test_encode_image_returns_base64(self, tmp_path):
        """Test that encode_image returns valid base64 string."""
        import base64
        
        test_file = tmp_path / "test.png"
        original_content = b'\x89PNG\r\n\x1a\n' + b'test image data'
        test_file.write_bytes(original_content)
        
        encoded = encode_image(str(test_file))
        
        # Should be valid base64
        decoded = base64.b64decode(encoded)
        assert decoded == original_content

    def test_encode_image_string_output(self, tmp_path):
        """Test that encode_image returns a string, not bytes."""
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b'\x89PNG\r\n\x1a\n')
        
        encoded = encode_image(str(test_file))
        
        assert isinstance(encoded, str)


class TestMimeTypeDetection:
    """Tests for MIME type detection."""

    def test_png_mime_type(self):
        """Test PNG MIME type detection."""
        assert get_image_media_type("image.png") == "image/png"
        assert get_image_media_type("IMAGE.PNG") == "image/png"

    def test_jpeg_mime_type(self):
        """Test JPEG MIME type detection."""
        assert get_image_media_type("photo.jpg") == "image/jpeg"
        assert get_image_media_type("photo.jpeg") == "image/jpeg"

    def test_gif_mime_type(self):
        """Test GIF MIME type detection."""
        assert get_image_media_type("animation.gif") == "image/gif"

    def test_webp_mime_type(self):
        """Test WebP MIME type detection."""
        assert get_image_media_type("modern.webp") == "image/webp"

    def test_unknown_defaults_to_png(self):
        """Test that unknown extension defaults to PNG."""
        assert get_image_media_type("file.unknown") == "image/png"


class TestParsedContractPageStructure:
    """Tests for ParsedContractPage model structure."""

    def test_valid_page_structure(self):
        """Test creating a valid ParsedContractPage."""
        page = ParsedContractPage(
            page_number=1,
            sections=[
                Section(
                    id="I",
                    title="AGREEMENT DETAILS",
                    content="Preamble text",
                    clauses=[
                        Clause(id="1.1", title="Scope", content="The Provider shall...")
                    ],
                    subsections=[]
                )
            ],
            raw_content="Full page text"
        )
        
        assert page.page_number == 1
        assert len(page.sections) == 1
        assert page.sections[0].id == "I"
        assert len(page.sections[0].clauses) == 1

    def test_page_without_sections(self):
        """Test page with empty sections (valid for partial pages)."""
        page = ParsedContractPage(
            page_number=1,
            sections=[],
            raw_content="Just raw text, no structure detected"
        )
        
        assert page.page_number == 1
        assert len(page.sections) == 0
        assert page.raw_content is not None

    def test_nested_subsections(self):
        """Test sections with nested subsections."""
        page = ParsedContractPage(
            page_number=1,
            sections=[
                Section(
                    id="I",
                    title="MAIN SECTION",
                    content="",
                    clauses=[],
                    subsections=[
                        Section(
                            id="I.A",
                            title="SUBSECTION",
                            content="Subsection content",
                            clauses=[],
                            subsections=[]
                        )
                    ]
                )
            ],
            raw_content=""
        )
        
        assert len(page.sections[0].subsections) == 1
        assert page.sections[0].subsections[0].id == "I.A"


class TestTestContractsExist:
    """Tests to verify test contract images exist."""

    @pytest.fixture
    def test_contracts_dir(self):
        """Get path to test contracts directory."""
        return Path(__file__).parent.parent / "data" / "test_contracts"

    def test_pair1_original_exists(self, test_contracts_dir):
        """Test that pair_1/original folder exists with images."""
        original_dir = test_contracts_dir / "pair_1" / "original"
        
        if original_dir.exists():
            images = list(original_dir.glob("*.png")) + list(original_dir.glob("*.jpg"))
            assert len(images) >= 1, "pair_1/original should have at least 1 image"
        else:
            pytest.skip("Test contracts not available")

    def test_pair1_amendment_exists(self, test_contracts_dir):
        """Test that pair_1/amendment folder exists with images."""
        amendment_dir = test_contracts_dir / "pair_1" / "amendment"
        
        if amendment_dir.exists():
            images = list(amendment_dir.glob("*.png")) + list(amendment_dir.glob("*.jpg"))
            assert len(images) >= 1, "pair_1/amendment should have at least 1 image"
        else:
            pytest.skip("Test contracts not available")

    def test_pair2_exists(self, test_contracts_dir):
        """Test that pair_2 folder exists with original and amendment."""
        pair2_dir = test_contracts_dir / "pair_2"
        
        if pair2_dir.exists():
            assert (pair2_dir / "original").exists(), "pair_2/original should exist"
            assert (pair2_dir / "amendment").exists(), "pair_2/amendment should exist"
        else:
            pytest.skip("Test contracts not available")


# =============================================================================
# INTEGRATION TESTS (require API keys - marked for separate execution)
# =============================================================================

@pytest.mark.integration
class TestImageParsingIntegration:
    """
    Integration tests that require OpenAI API.
    Run with: pytest -m integration tests/test_image_parsing.py
    """

    @pytest.fixture
    def test_image_path(self):
        """Get path to a test image."""
        return Path(__file__).parent.parent / "data" / "test_contracts" / "pair_1" / "original" / "page_01.png"

    @pytest.mark.skip(reason="Requires API key and makes actual API calls")
    def test_parse_single_image(self, test_image_path):
        """Test parsing a single contract image."""
        # This test would require actual API calls
        # Skipped by default to avoid API costs
        pass

    @pytest.mark.skip(reason="Requires API key and makes actual API calls")
    def test_parse_folder(self):
        """Test parsing all images in a folder."""
        # This test would require actual API calls
        # Skipped by default to avoid API costs
        pass
