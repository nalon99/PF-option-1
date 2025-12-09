"""
Image Parser for Contract Documents

Functions for:
- Image validation (format, size)
- Encoding (base64)
- Multimodal API calls (GPT-4o) with proper message formatting
- Vision-specific prompts for contract parsing
- Extracts structured text preserving document hierarchy (sections, subsections, clauses)
- Handles various image qualities (scanned, photographed, different resolutions)
- Langfuse tracing integration for monitoring
"""

import datetime
import os
import json
import base64
from typing import List, Optional
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import ValidationError

from models import ParsedContractPage
from tracing import TracingSession

# Load environment variables from project root .env file
ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_FILE, override=True)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI_MODEL = os.getenv("OPENAI_MODEL")
USE_OPEN_ROUTER = os.getenv("USE_OPEN_ROUTER", "false").lower() == "true"

# Keep original model name for Langfuse cost tracking (e.g., "gpt-4o")
LANGFUSE_MODEL_NAME = AI_MODEL

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


# Hierarchical extraction prompt for legal documents
SYSTEM_PROMPT = """You are a document parsing assistant specialized in extracting content from scanned legal documents.

Extract the content preserving the HIERARCHICAL STRUCTURE. Return a JSON object with this exact structure:

{
    "page_number": <integer>,
    "sections": [
        {
            "id": "<section number, e.g., 'I', 'II', '1', '2'>",
            "title": "<section title, e.g., 'TERM AND TERMINATION'>",
            "content": "<optional preamble text before clauses>",
            "clauses": [
                {
                    "id": "<clause number, e.g., '1.1', '2.3'>",
                    "title": "<clause title if present, e.g., 'Scope of Work'>",
                    "content": "<full clause text>"
                }
            ],
            "subsections": []
        }
    ],
    "raw_content": "<full page text as fallback>"
}

Rules:
1. PRESERVE document hierarchy: Section → Clauses → Subclauses
2. Section IDs are typically Roman numerals (I, II, III) or numbers (1, 2, 3)
3. Clause IDs are typically decimal numbers (1.1, 1.2, 2.1)
4. Remove formatting markers (**, *, etc.) from the output
5. Convert numerical values in words to digits when clear (e.g., "fifteen thousand" → 15000)
6. Use "|" to represent unrecognizable characters
7. If a section continues from a previous page, still include it with available content
8. Include raw_content as a fallback with the full page text
"""


def parse_contract_image(
    image_path: str,
    page_number: int,
    session: TracingSession
) -> Optional[ParsedContractPage]:
    """
    Parse a contract image and extract hierarchical structured content.
    
    Args:
        image_path: Path to the contract image file
        page_number: Page number for this image (1-indexed)
        session: TracingSession for Langfuse monitoring (REQUIRED)
        
    Returns:
        ParsedContractPage object with validated hierarchical content, or None on error
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
    # Add a parameter to identify contract type (original/amendment)
    contract_type = "original" if "original" in image_path.lower() else (
        "amendment" if "amendment" in image_path.lower() else "unknown"
    )
    span_name = f"parse_{contract_type}_contract"
    # Create generation for LLM call tracing (tracks model, tokens, cost)
    span = session.create_generation(
        name=span_name,
        model=LANGFUSE_MODEL_NAME,
        input_data={"image_path": str(image_path), "page_number": page_number},
        metadata={
            "session_id": getattr(session, 'session_id', None),
            "contract_paid_id": getattr(session, 'contract_id', None),
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
        }
    )
    
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
                            "text": f"Extract the hierarchical content from this contract page (page {page_number})."
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        # Update trace with usage info
        if hasattr(completion, 'usage') and completion.usage:
            span.update(
                output=completion.choices[0].message.content[:500],  # Truncate for trace
                usage={
                    "input": completion.usage.prompt_tokens,
                    "output": completion.usage.completion_tokens,
                    "total": completion.usage.total_tokens
                }
            )
        
        # Parse the response
        response_content = completion.choices[0].message.content
        parsed_json = json.loads(response_content)
        
        # Ensure page_number is set correctly
        parsed_json["page_number"] = page_number
        
        # Validate with Pydantic model
        validated_page = ParsedContractPage.model_validate(parsed_json)
        print(f"✅ Page {page_number} parsed and validated successfully")
        print(f"   Found {len(validated_page.sections)} sections")
        
        # Update span with success
        span.update(
            output={"success": True, "sections_found": len(validated_page.sections)},
            status_message="success"
        )
        
        return validated_page
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        span.update(output={"error": str(e)}, status_message="error")
        return None
    except ValidationError as e:
        print(f"❌ Validation error: {e}")
        span.update(output={"error": str(e)}, status_message="error")
        return None
    except Exception as e:
        print(f"❌ Error parsing image: {e}")
        span.update(output={"error": str(e)}, status_message="error")
        return None
    finally:
        span.end()


def parse_contract_folder(
    folder_path: str,
    session: TracingSession
) -> List[ParsedContractPage]:
    """
    Parse all contract images in a folder.
    
    Args:
        folder_path: Path to folder containing contract page images
        session: TracingSession for Langfuse monitoring (REQUIRED)
        
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
    
    # Create span for folder parsing (REQUIRED)
    span = session.create_span(
        name=f"parse_folder_{Path(folder_path).name}",
        input_data={"folder_path": folder_path, "image_count": len(image_files)},
        metadata={
            "session_id": getattr(session, 'session_id', None),
            "contract_paid_id": getattr(session, 'contract_id', None),
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
        }
    )
    
    parsed_pages = []
    for i, image_file in enumerate(image_files, start=1):
        page = parse_contract_image(str(image_file), page_number=i, session=session)
        if page:
            parsed_pages.append(page)
    
    # Update span with results
    span.update(
        output={"pages_parsed": len(parsed_pages), "pages_failed": len(image_files) - len(parsed_pages)},
        status_message="success" if len(parsed_pages) == len(image_files) else "partial"
    )
    span.end()
    
    return parsed_pages


# CLI entry point for testing
if __name__ == "__main__":
    from tracing import create_session, flush_traces
    
    input_path = "data/test_contracts/pair_1/original/page_01.png"
    
    # Create a tracing session
    session = create_session(contract_id="cli_test")
    
    if Path(input_path).is_dir():
        # Parse all images in folder
        pages = parse_contract_folder(input_path, session=session)
        print(f"\n{'='*60}")
        print(f"Parsed {len(pages)} pages")
        print(f"{'='*60}")
        for page in pages:
            print(f"\nPage {page.page_number}: {len(page.sections)} sections")
            for section in page.sections:
                print(f"  {section.id}. {section.title} ({len(section.clauses)} clauses)")
    else:
        # Parse single image
        page = parse_contract_image(input_path, session=session)
        if page:
            print(f"\n{'='*60}")
            print(f"Page {page.page_number}")
            print(f"Sections found: {len(page.sections)}")
            print(f"{'='*60}")
            for section in page.sections:
                print(f"\n{section.id}. {section.title}")
                print(f"   Clauses: {len(section.clauses)}")
    
    # End session and flush traces
    session.end(status="success")
    flush_traces()
