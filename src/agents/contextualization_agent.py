"""
Contextualization Agent (Agent 1)

Single LLM call that:
1. Receives parsed pages from BOTH original and amended contracts
2. Merges page-spanning sections
3. Aligns corresponding sections between documents
4. Identifies which sections have meaningful changes

Output is passed to Agent 2 (Extraction Agent) for detailed change extraction.
Includes Langfuse tracing for monitoring.
"""

import os
import json
from typing import List, Optional
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import ValidationError

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    ParsedContractPage,
    ContextualizationOutput
)
from tracing import TracingSession

# Load environment variables from project root .env file
ENV_FILE = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_FILE, override=True)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI_MODEL = os.getenv("OPENAI_MODEL")
USE_OPEN_ROUTER = os.getenv("USE_OPEN_ROUTER", "false").lower() == "true"

# Keep original model name for Langfuse cost tracking (e.g., "gpt-4o")
LANGFUSE_MODEL_NAME = AI_MODEL

# Initialize async client
if USE_OPEN_ROUTER:
    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    AI_MODEL = f"openai/{AI_MODEL}"
else:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# =============================================================================
# SINGLE-CALL PROMPT: Assembly + Alignment in One Pass
# =============================================================================

CONTEXTUALIZATION_PROMPT = """You are a legal document comparison expert. You will receive parsed pages from TWO contracts:
1. ORIGINAL contract
2. AMENDED contract

Your task (in ONE pass):
1. MERGE pages: Detect section continuations across pages, combine split content
2. ALIGN sections: Match corresponding sections between original and amended
3. DETECT changes: Mark sections with meaningful value changes (dates, numbers, terms)

ORIGINAL CONTRACT PAGES:
{original_pages}

AMENDED CONTRACT PAGES:
{amended_pages}

Return JSON:
{{
    "original_title": "<document title>",
    "amended_title": "<document title>",
    "aligned_sections": [
        {{
            "section_id": "<section ID>",
            "section_title": "<section title>",
            "original_content": "<full merged content from original>",
            "amended_content": "<full merged content from amended>",
            "has_changes": true/false,
            "is_removed": false,
            "is_added": false
        }}
    ]
}}

RULES:
- Merge same section across pages (detect mid-sentence cuts)
- Include ALL clauses within each section (e.g., 2.1, 2.2, 2.3 all go in Section II)
- has_changes: true ONLY for substantive changes (values, dates, terms)
- has_changes: false for identical content or OCR artifacts ("|" characters)
- is_removed: true if section only in original (empty amended_content)
- is_added: true if section only in amended (empty original_content)
- Preserve exact wording - no paraphrasing
- Ignore formatting differences (quotes, whitespace)
"""


# =============================================================================
# CONTEXTUALIZATION AGENT CLASS
# =============================================================================

class ContextualizationAgent:
    """
    Agent 1: Single LLM call for assembly + alignment.
    
    Takes parsed pages from both contracts and produces aligned sections
    with change detection for Agent 2.
    """
    
    def __init__(self, session: Optional[TracingSession] = None):
        """Initialize with optional tracing session."""
        self.session = session
    
    async def agent_contextualize(
        self,
        original_pages: List[ParsedContractPage],
        amended_pages: List[ParsedContractPage],
        original_name: str = "Original Contract",
        amended_name: str = "Amended Contract"
    ) -> Optional[ContextualizationOutput]:
        """
        Single LLM call: Assemble + Align both contracts.
        
        Args:
            original_pages: Parsed pages from original contract
            amended_pages: Parsed pages from amended contract
            original_name: Name hint for original
            amended_name: Name hint for amended
            
        Returns:
            ContextualizationOutput with aligned sections
        """
        if not original_pages or not amended_pages:
            print("❌ Missing pages for one or both contracts")
            return None
        
        print(f"🔄 Contextualizing: {len(original_pages)} original + {len(amended_pages)} amended pages")
        
        # Prepare compact JSON for both contracts
        original_data = self._pages_to_compact_json(original_pages)
        amended_data = self._pages_to_compact_json(amended_pages)
        
        # Create tracing span
        span = None
        if self.session:
            span = self.session.create_generation(
                name="agent_contextualize",
                model=LANGFUSE_MODEL_NAME,
                input_data={
                    "original_pages": len(original_pages),
                    "amended_pages": len(amended_pages)
                },
                metadata={
                    "agent": "contextualization_agent",
                    "mode": "single_call"
                }
            )
        
        try:
            # Format prompt
            prompt = CONTEXTUALIZATION_PROMPT.format(
                original_pages=original_data,
                amended_pages=amended_data
            )
            
            # Single async LLM call
            completion = await client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            # Update tracing with usage
            if span and hasattr(completion, 'usage') and completion.usage:
                span.update(
                    usage={
                        "input": completion.usage.prompt_tokens,
                        "output": completion.usage.completion_tokens,
                        "total": completion.usage.total_tokens
                    }
                )
            
            response_content = completion.choices[0].message.content
            parsed_json = json.loads(response_content)
            
            # Validate with Pydantic
            result = ContextualizationOutput.model_validate(parsed_json)
            
            changed_count = sum(1 for s in result.aligned_sections if s.has_changes)
            print(f"✅ Contextualization complete: {len(result.aligned_sections)} sections, {changed_count} with changes")
            
            if span:
                span.update(
                    output={
                        "success": True,
                        "sections": len(result.aligned_sections),
                        "changed": changed_count
                    },
                    status_message="success"
                )
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            if span:
                span.update(output={"error": str(e)}, status_message="error")
            return None
        except ValidationError as e:
            print(f"❌ Validation error: {e}")
            if span:
                span.update(output={"error": str(e)}, status_message="error")
            return None
        except Exception as e:
            print(f"❌ Error in contextualization: {e}")
            if span:
                span.update(output={"error": str(e)}, status_message="error")
            return None
        finally:
            if span:
                span.end()
    
    def _pages_to_compact_json(self, pages: List[ParsedContractPage]) -> str:
        """Convert pages to compact JSON for minimal tokens."""
        pages_data = []
        for page in pages:
            page_dict = {
                "page": page.page_number,
                "sections": [
                    {
                        "id": s.id,
                        "title": s.title,
                        "content": s.content,
                        "clauses": [
                            {"id": c.id, "title": c.title, "content": c.content}
                            for c in s.clauses
                        ]
                    }
                    for s in page.sections
                ]
            }
            pages_data.append(page_dict)
        
        # Use compact separators to minimize tokens
        return json.dumps(pages_data, separators=(',', ':'))
