"""
Image Parser for Contract Documents

Functions for:
- Image validation (format, size)
- Encoding (base64)
- Multimodal API calls (GPT-4o) with proper message formatting
- Vision-specific prompts for contract parsing
- Extracts structured text preserving document hierarchy (sections, subsections, clauses)
- Handles various image qualities (scanned, photographed, different resolutions)
"""

import os
import json
import base64
from typing import Optional
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import ValidationError

from models import ParsedContractPage

# Load environment variables
load_dotenv(override=True)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI_MODEL = os.getenv("OPENAI_MODEL")
USE_OPEN_ROUTER = os.getenv("USE_OPEN_ROUTER", "false").lower() == "true"

# Initialize client (same API key for both OpenAI and OpenRouter)
if USE_OPEN_ROUTER:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    # OpenRouter requires provider prefix (e.g., "openai/gpt-4o")
    AI_MODEL = f"openai/{AI_MODEL}"
else:
    client = OpenAI(api_key=OPENAI_API_KEY)


# Supported image formats
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MAX_IMAGE_SIZE_MB = 20  # OpenAI limit


def validate_image(image_path: str) -> tuple[bool, str]:
    """
    Validate image file format and size.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(image_path)
    
    # Check file exists
    if not path.exists():
        return False, f"Image file not found: {image_path}"
    
    # Check format
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        return False, f"Unsupported format: {path.suffix}. Supported: {SUPPORTED_FORMATS}"
    
    # Check size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        return False, f"Image too large: {size_mb:.1f}MB (max {MAX_IMAGE_SIZE_MB}MB)"
    
    return True, ""


def encode_image(image_path: str) -> str:
    """
    Encode an image file as a base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Get the MIME type for an image based on extension."""
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return media_types.get(ext, "image/png")


SYSTEM_PROMPT = """You are a document parsing assistant specialized in extracting content from scanned legal documents.

Extract the content from the image and return a JSON object with this structure:
{
    "page_number": <integer>,
    "sections": ["Section I Title", "Section II Title", ...],
    "content": "<full text content preserving structure>"
}

Rules:
1. Preserve document hierarchy (sections, subsections, clauses)
2. Remove formatting markers (**, *, etc.) from the output
3. Convert numerical values in words to digits when clear (e.g., "fifteen thousand" → 15000)
4. Use "|" to represent unrecognizable characters
5. Maintain paragraph breaks with newlines
6. Include section numbers and titles in the sections array
"""


def parse_contract_image(
    image_path: str,
    page_number: int = 1
) -> Optional[ParsedContractPage]:
    """
    Parse a contract image and extract structured content.
    
    Args:
        image_path: Path to the contract image file
        page_number: Page number for this image (1-indexed)
        model: Optional model override
        
    Returns:
        ParsedContractPage object with validated content, or None on error
    """
    # Validate image
    is_valid, error_msg = validate_image(image_path)
    if not is_valid:
        print(f"Image validation error: {error_msg}")
        return None
    
    print(f"Encoding image: {image_path}")
    base64_img = encode_image(image_path)
    media_type = get_image_media_type(image_path)
    
    print(f"Sending image to LLM ({AI_MODEL})...")
    
    try:
        completion = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_img}"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"Extract the content from this contract page (page {page_number})."
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        # Parse the response
        response_content = completion.choices[0].message.content
        parsed_json = json.loads(response_content)
        
        # Ensure page_number is set correctly
        parsed_json["page_number"] = page_number
        
        # Validate with Pydantic model
        validated_page = ParsedContractPage.model_validate(parsed_json)
        print(f"✅ Page {page_number} parsed and validated successfully")
        return validated_page
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None
    except ValidationError as e:
        print(f"Validation error: {e}")
        return None
    except Exception as e:
        print(f"Error parsing image: {e}")
        return None


def parse_contract_folder(
    folder_path: str
) -> list[ParsedContractPage]:
    """
    Parse all contract images in a folder.
    
    Args:
        folder_path: Path to folder containing contract page images
        model: Optional model override
        
    Returns:
        List of ParsedContractPage objects sorted by page number
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder_path}")
        return []
    
    # Find all image files and sort by name
    image_files = sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in SUPPORTED_FORMATS
    ])
    
    if not image_files:
        print(f"No images found in: {folder_path}")
        return []
    
    print(f"Found {len(image_files)} images in {folder_path}")
    
    parsed_pages = []
    for i, image_file in enumerate(image_files, start=1):
        page = parse_contract_image(str(image_file), page_number=i, model=AI_MODEL)
        if page:
            parsed_pages.append(page)
    
    return parsed_pages


# CLI entry point for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python image_parser.py <image_path_or_folder>")
        sys.exit(1)
    
    target = sys.argv[1]
    
    if Path(target).is_dir():
        pages = parse_contract_folder(target)
        print(f"\nParsed {len(pages)} pages")
        for page in pages:
            print(f"  Page {page.page_number}: {len(page.sections)} sections, {len(page.content)} chars")
    else:
        page = parse_contract_image(target)
        if page:
            print(f"\nParsed page {page.page_number}")
            print(f"Sections: {page.sections}")
            print(f"Content preview: {page.content[:200]}...")
