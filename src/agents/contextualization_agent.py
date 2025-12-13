"""
Contextualization Agent (Agent 1)

In a multi-agent system where there are two distinct agents with separate prompts 
and responsibilities, this is Agent 1:
- Step 1: Intelligent document assembly (merge pages, detect section continuations)
- Step 2: Cross-document alignment (match corresponding sections between original and amended)

Output is passed to Agent 2 (Extraction Agent) for change identification.
Includes Langfuse tracing for monitoring all LLM calls.
Uses AsyncOpenAI for concurrent LLM calls.
"""

import asyncio
import datetime
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
    ParsedContractDocument,
    Section,
    ContextualizationOutput,
    SectionAlignment
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
# STEP 1: Intelligent Document Assembly
# =============================================================================

ASSEMBLY_PROMPT = """Merge parsed contract pages into ONE coherent document.

TASK: Detect page-spanning sections, merge split content, extract metadata.

OUTPUT (JSON):
{"title":"...","effective_date":"...or null","parties":["..."],"sections":[{"id":"I","title":"...","content":"preamble","clauses":[{"id":"1.1","title":"...or null","content":"..."}],"subsections":[]}]}

RULES:
- Merge same section across pages (detect mid-sentence cuts, continuing clause numbers)
- Preserve EXACT wording - no paraphrasing
- Keep "|" characters as-is (OCR artifacts)
- Extract title from first page, parties if mentioned
"""


async def assemble_document(
    pages: List[ParsedContractPage],
    document_name: str,
    session: TracingSession
) -> Optional[ParsedContractDocument]:
    """
    Step 1: Intelligently merge parsed pages into a coherent document.
    
    Uses LLM to detect section continuations and merge split content.
    
    Args:
        pages: List of ParsedContractPage from image parser
        document_name: Name hint for the document
        session: TracingSession for Langfuse monitoring (REQUIRED)
        
    Returns:
        ParsedContractDocument with intelligently merged sections
    """
    if not pages:
        print("❌ No pages to assemble")
        return None
    
    # Prepare pages data for LLM
    pages_data = []
    for page in pages:
        page_dict = {
            "page_number": page.page_number,
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
            ],
            "raw_content": page.raw_content
        }
        pages_data.append(page_dict)
    
    print(f"📄 Assembling {document_name} {len(pages)} pages into document...")
    
    # Create generation for LLM call tracing (tracks model, tokens, cost)
    span = session.create_generation(
        name=f"assemble_document_{document_name}",
        model=LANGFUSE_MODEL_NAME,
        input_data={"document_name": document_name, "page_count": len(pages)},
        metadata={
            "session_id": getattr(session, 'session_id', None),
            "contract_paid_id": getattr(session, 'contract_id', None),
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
        }
    )
    
    try:
        # Async LLM call
        completion = await client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": ASSEMBLY_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Merge {len(pages)} pages into '{document_name}':\n{json.dumps(pages_data, separators=(',',':'))}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        if hasattr(completion, 'usage') and completion.usage:
            span.update(
                output=completion.choices[0].message.content[:500],
                usage={
                    "input": completion.usage.prompt_tokens,
                    "output": completion.usage.completion_tokens,
                    "total": completion.usage.total_tokens
                }
            )
        
        response_content = completion.choices[0].message.content
        parsed_json = json.loads(response_content)
        
        # Validate with Pydantic model
        document = ParsedContractDocument.model_validate(parsed_json)
        print(f"✅ Document assembled: {document.title} ({len(document.sections)} sections)")
        
        span.update(
            output={"success": True, "title": document.title, "sections_count": len(document.sections)},
            status_message="success"
        )
        
        return document
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error in assembly: {e}")
        span.update(output={"error": str(e)}, status_message="error")
        return None
    except ValidationError as e:
        print(f"❌ Validation error in assembly: {e}")
        span.update(output={"error": str(e)}, status_message="error")
        return None
    except Exception as e:
        print(f"❌ Error assembling document: {e}")
        span.update(output={"error": str(e)}, status_message="error")
        return None
    finally:
        span.end()


# =============================================================================
# STEP 2: Cross-Document Alignment
# =============================================================================

ALIGNMENT_PROMPT = """You are a legal document comparison expert. Your task is to align corresponding sections between an ORIGINAL contract and its AMENDMENT.

You will receive two parsed documents. Your job is to:

1. IDENTIFY CORRESPONDING SECTIONS:
   - Match sections by ID (e.g., "II" in original matches "II" in amended)
   - Match sections by title if IDs differ slightly
   - Note sections that only appear in one document

2. COMPARE CONTENT:
   - For each aligned section, extract the relevant content from both documents
   - Mark whether the section has changes (has_changes: true/false)
   - Include enough context to understand the change

3. FLAG STRUCTURAL CHANGES:
   - Sections added in amendment
   - Sections removed from original
   - Sections that were renumbered

Return a JSON object with this structure:
{
    "original_title": "<original document title>",
    "amended_title": "<amended document title>",
    "aligned_sections": [
        {
            "section_id": "<section identifier>",
            "section_title": "<section title>",
            "original_content": "<relevant content from original>",
            "amended_content": "<relevant content from amended>",
            "has_changes": true/false
        }
    ],
    "sections_only_in_original": ["<section ids>"],
    "sections_only_in_amended": ["<section ids>"]
}

Rules:
- Include ALL sections from both documents
- Extract complete clause content where changes occur
- For unchanged sections, still include them with has_changes: false
- Be precise about what content changed
- The pipe character "|" represents unrecognized characters from OCR
- When comparing, treat "|" as a wildcard - e.g., "$15|000" and "$15,000" may be the same value
- Do NOT flag differences that are only due to "|" OCR artifacts
"""


async def align_documents(
    original: ParsedContractDocument,
    amended: ParsedContractDocument,
    session: TracingSession
) -> Optional[ContextualizationOutput]:
    """
    Step 2: Align sections between original and amended documents.
    
    Uses LLM to match corresponding sections and identify where changes occurred.
    
    Args:
        original: The original ParsedContractDocument
        amended: The amended ParsedContractDocument
        session: TracingSession for Langfuse monitoring (REQUIRED)
        
    Returns:
        ContextualizationOutput with aligned sections for Agent 2
    """
    print(f"🔄 Aligning documents: '{original.title}' ↔ '{amended.title}'")
    
    # Prepare document data for LLM
    original_data = {
        "title": original.title,
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
            for s in original.sections
        ]
    }
    
    amended_data = {
        "title": amended.title,
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
            for s in amended.sections
        ]
    }
    
    # Create generation for LLM call tracing (tracks model, tokens, cost)
    span = session.create_generation(
        name="align_documents",
        model=LANGFUSE_MODEL_NAME,
        input_data={
            "original_title": original.title,
            "amended_title": amended.title,
            "original_sections": len(original.sections),
            "amended_sections": len(amended.sections)
        },
        metadata={"agent": "contextualization_agent", "operation": "align_documents"}
    )
    
    try:
        # Async LLM call
        completion = await client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": ALIGNMENT_PROMPT
                },
                {
                    "role": "user",
                    "content": f"ORIGINAL:\n{json.dumps(original_data, separators=(',',':'))}\n\nAMENDED:\n{json.dumps(amended_data, separators=(',',':'))}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        if hasattr(completion, 'usage') and completion.usage:
            span.update(
                output=completion.choices[0].message.content[:500],
                usage={
                    "input": completion.usage.prompt_tokens,
                    "output": completion.usage.completion_tokens,
                    "total": completion.usage.total_tokens
                }
            )
        
        response_content = completion.choices[0].message.content
        parsed_json = json.loads(response_content)
        
        # Validate with Pydantic model
        alignment = ContextualizationOutput.model_validate(parsed_json)
        
        # Count changes
        changed_count = sum(1 for s in alignment.aligned_sections if s.has_changes)
        print(f"✅ Alignment complete: {len(alignment.aligned_sections)} sections aligned, {changed_count} with changes")
        
        span.update(
            output={
                "success": True,
                "aligned_sections": len(alignment.aligned_sections),
                "changed_sections": changed_count
            },
            status_message="success"
        )
        
        return alignment
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error in alignment: {e}")
        span.update(output={"error": str(e)}, status_message="error")
        return None
    except ValidationError as e:
        print(f"❌ Validation error in alignment: {e}")
        span.update(output={"error": str(e)}, status_message="error")
        return None
    except Exception as e:
        print(f"❌ Error aligning documents: {e}")
        span.update(output={"error": str(e)}, status_message="error")
        return None
    finally:
        span.end()


# =============================================================================
# MAIN AGENT CLASS
# =============================================================================

class ContextualizationAgent:
    """
    Agent 1: Contextualizes and aligns contract documents.
    
    Two-step process:
    1. Assemble pages into coherent documents (intelligent merging)
    2. Align sections between original and amended documents
    
    Langfuse tracing is REQUIRED for all operations.
    """
    
    def __init__(self, session: TracingSession):
        self.session = session
        self.original_document: Optional[ParsedContractDocument] = None
        self.amended_document: Optional[ParsedContractDocument] = None
        self.alignment: Optional[ContextualizationOutput] = None
    
    async def agent_contextualize(
        self,
        original_pages: List[ParsedContractPage],
        amended_pages: List[ParsedContractPage],
        original_name: str = "Original Contract",
        amended_name: str = "Amended Contract"
    ) -> Optional[ContextualizationOutput]:
        """
        Main entry point: Process both contracts and produce aligned output.
        
        Args:
            original_pages: Parsed pages from original contract images
            amended_pages: Parsed pages from amended contract images
            original_name: Name for the original document
            amended_name: Name for the amended document
            
        Returns:
            ContextualizationOutput ready for Agent 2, or None on error
        """
        print("\n" + "="*60)
        print("CONTEXTUALIZATION AGENT - Starting")
        print("="*60)
        
        # Create agent-level span (REQUIRED)
        agent_span = self.session.create_span(
            name="agent_contextualize",
            input_data={
                "original_pages": len(original_pages),
                "amended_pages": len(amended_pages),
                "original_name": original_name,
                "amended_name": amended_name
            },
            metadata={
                "session_id": getattr(self.session, 'session_id', None),
                "contract_paid_id": getattr(self.session, 'contract_id', None),
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
            }
        )
        
        try:
            # Step 1: Assemble documents (PARALLEL using asyncio.gather)
            print("\n📋 STEP 2.1: Document Assembly (parallel async)")
            print("-"*40)
            
            # Run both assembly tasks concurrently
            results = await asyncio.gather(
                assemble_document(original_pages, original_name, self.session),
                assemble_document(amended_pages, amended_name, self.session),
                return_exceptions=True
            )
            
            # Handle results
            if isinstance(results[0], Exception):
                print(f"❌ Failed to assemble original document: {results[0]}")
                return None
            if isinstance(results[1], Exception):
                print(f"❌ Failed to assemble amended document: {results[1]}")
                return None
                
            self.original_document = results[0]
            self.amended_document = results[1]
            
            if not self.original_document:
                print("❌ Failed to assemble original document")
                return None
            
            if not self.amended_document:
                print("❌ Failed to assemble amended document")
                return None
            
            # Step 2: Align documents
            print("\n🔗 STEP 2.2: Document Alignment")
            print("-"*40)
            
            self.alignment = await align_documents(self.original_document, self.amended_document, self.session)
            if not self.alignment:
                print("❌ Failed to align documents")
                return None
            
            # Validate output
            if not self.validate_output():
                print("❌ Output validation failed")
                return None
            
            print("\n" + "="*60)
            print("CONTEXTUALIZATION AGENT - Complete")
            print("="*60)
            
            agent_span.update(
                output={
                    "success": True,
                    "original_sections": len(self.original_document.sections),
                    "amended_sections": len(self.amended_document.sections),
                    "aligned_sections": len(self.alignment.aligned_sections),
                    "changed_sections": len(self.get_changed_sections())
                },
                status_message="success"
            )
            
            return self.alignment
            
        except Exception as e:
            print(f"❌ Agent error: {e}")
            agent_span.update(output={"error": str(e)}, status_message="error")
            return None
        finally:
            agent_span.end()
    
    def get_changed_sections(self) -> List[SectionAlignment]:
        """Get only sections that have changes."""
        if not self.alignment:
            return []
        return [s for s in self.alignment.aligned_sections if s.has_changes]
    
    def validate_output(self) -> bool:
        """Validate that the output is ready for Agent 2."""
        if not self.alignment:
            return False
        
        # Create validation span (REQUIRED)
        validation_span = self.session.create_span(
            name="validate_contextualization_output",
            input_data={"alignment_sections": len(self.alignment.aligned_sections)},
            metadata={"agent": "contextualization_agent", "operation": "validation"}
        )
        
        try:
            # Re-validate the alignment
            ContextualizationOutput.model_validate(self.alignment.model_dump())
            
            validation_span.update(output={"valid": True}, status_message="valid")
            
            return True
        except ValidationError as e:
            print(f"❌ Output validation failed: {e}")
            validation_span.update(output={"valid": False, "error": str(e)}, status_message="invalid")
            return False
        finally:
            validation_span.end()


# =============================================================================
# CLI Entry Point for Testing
# =============================================================================

# if __name__ == "__main__":
#     from image_parser import parse_contract_folder
    
#     # Get project root (two levels up from this file)
#     import os
#     project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
#     # Test with pair_1
#     original_path = os.path.join(project_root, "data/test_contracts/pair_2/original")
#     amended_path = os.path.join(project_root, "data/test_contracts/pair_2/amendment")
    
#     # Create tracing session
#     session = create_session(contract_id="pair_2_test")
    
#     print("Parsing original contract images...")
#     original_pages = parse_contract_folder(original_path, session=session)
    
#     print("\nParsing amended contract images...")
#     amended_pages = parse_contract_folder(amended_path, session=session)
    
#     if original_pages and amended_pages:
#         agent = ContextualizationAgent(session=session)
#         result = agent.agent_contextualize(
#             original_pages, 
#             amended_pages,
#             "Original",
#             "Amended"
#         )
        
#         if result:
#             # Save the result for later use by the extraction_agent
#             output_path = os.path.join(project_root, "data/test_contracts/pair_2/contextualization_output.json")
#             with open(output_path, "w") as f:
#                 json.dump(result.model_dump(), f, indent=2)
            
#             print("\n" + "="*60)
#             print("RESULTS")
#             print("="*60)
#             print(f"\nSections with changes:")
#             for section in agent.get_changed_sections():
#                 print(f"  - {section.section_id}: {section.section_title}")
    
#     # End session and flush traces
#     session.end(output={"status": "completed"}, status="success")
#     flush_traces()
