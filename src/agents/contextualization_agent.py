"""
Contextualization Agent (Agent 1)

In a multi-agent system where there are two distinct agents with separate prompts 
and responsibilities, this is Agent 1:
- Step 1: Build section index from parsed pages (fast, programmatic)
- Step 2: LLM-powered alignment to detect meaningful changes

OPTIMIZED: Skips expensive LLM document assembly but uses LLM for intelligent
alignment that can distinguish real changes from OCR/formatting artifacts.

Output is passed to Agent 2 (Extraction Agent) for detailed change extraction.
Includes Langfuse tracing for monitoring all LLM calls.
"""

import datetime
import os
import json
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import ValidationError

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    ParsedContractPage,
    Section,
    ContextualizationOutput,
    SectionAlignment
)
from tracing import TracingSession

# Load environment variables
ENV_FILE = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_FILE, override=True)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI_MODEL = os.getenv("OPENAI_MODEL")
USE_OPEN_ROUTER = os.getenv("USE_OPEN_ROUTER", "false").lower() == "true"
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
# STEP 1: Build Section Index (Fast, Programmatic)
# =============================================================================

def is_roman_numeral(s: str) -> bool:
    """Check if a string is a valid Roman numeral."""
    if not s:
        return False
    return all(c in 'IVXLCDM' for c in s.upper()) and s == s.upper()


def arabic_to_roman(n: int) -> str:
    """Convert Arabic number to Roman numeral."""
    roman_map = [(10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
    result = ''
    for value, numeral in roman_map:
        while n >= value:
            result += numeral
            n -= value
    return result


def get_parent_section_id(clause_id: str) -> Optional[str]:
    """
    Get the Roman numeral parent section ID for a clause.
    E.g., "2.1" -> "II", "3.3" -> "III", "6.2" -> "VI"
    """
    if '.' in clause_id:
        prefix = clause_id.split('.')[0]
    else:
        prefix = clause_id
    
    try:
        num = int(prefix)
        return arabic_to_roman(num)
    except ValueError:
        return None


def build_section_index(pages: List[ParsedContractPage]) -> Tuple[Dict[str, Section], str]:
    """
    Build a section index from parsed pages by combining sections with same ID.
    
    This is a fast programmatic operation (no LLM needed).
    Sections spanning multiple pages are merged by appending content.
    Non-Roman sections (clauses like 2.2, 3.3) are merged into their parent Roman section.
    
    Args:
        pages: List of ParsedContractPage from image parser
        
    Returns:
        Tuple of (section_map, document_title)
    """
    section_map: Dict[str, Section] = {}
    orphan_clauses: List[Tuple[str, Section]] = []  # (parent_id, section with clauses)
    document_title = "Contract Document"
    
    # First pass: collect Roman numeral sections and track orphan clauses
    for page in pages:
        for section in page.sections:
            section_id = section.id
            
            if is_roman_numeral(section_id):
                # Roman numeral section - add/merge normally
                if section_id in section_map:
                    existing = section_map[section_id]
                    if section.content:
                        existing.content = (existing.content or "") + " " + section.content
                    if section.title and not existing.title:
                        existing.title = section.title
                    # Append clauses
                    existing_clause_ids = {c.id for c in existing.clauses}
                    for clause in section.clauses:
                        if clause.id not in existing_clause_ids:
                            existing.clauses.append(clause.model_copy(deep=True))
                        else:
                            for ec in existing.clauses:
                                if ec.id == clause.id and clause.content:
                                    ec.content = ec.content + " " + clause.content
                                    break
                else:
                    section_map[section_id] = section.model_copy(deep=True)
                
                # Extract document title
                if section.title and document_title == "Contract Document":
                    if any(kw in section.title.upper() for kw in 
                           ["AGREEMENT", "CONTRACT", "AMENDMENT", "MASTER", "SERVICE"]):
                        document_title = section.title
            else:
                # Non-Roman section - find parent and save for merging
                parent_id = get_parent_section_id(section_id)
                if parent_id:
                    orphan_clauses.append((parent_id, section))
    
    # Second pass: merge orphan clauses into their parent sections
    for parent_id, orphan_section in orphan_clauses:
        if parent_id in section_map:
            parent = section_map[parent_id]
            existing_clause_ids = {c.id for c in parent.clauses}
            for clause in orphan_section.clauses:
                if clause.id not in existing_clause_ids:
                    parent.clauses.append(clause.model_copy(deep=True))
    
    return section_map, document_title


def get_section_content(section: Section) -> str:
    """Get full content of a section including all clauses."""
    parts = []
    
    if section.content:
        parts.append(section.content)
    
    for clause in section.clauses:
        clause_text = f"{clause.id}. "
        if clause.title:
            clause_text += f"{clause.title}: "
        clause_text += clause.content
        parts.append(clause_text)
    
    return "\n".join(parts) if parts else ""


# =============================================================================
# STEP 2: LLM-Powered Alignment
# =============================================================================

ALIGNMENT_PROMPT = """You are a legal document comparison expert. Compare the ORIGINAL and AMENDED contract sections to identify meaningful changes.

For each section pair, determine if there are REAL changes in values (not just OCR artifacts or formatting).

ORIGINAL DOCUMENT: {original_title}
AMENDED DOCUMENT: {amended_title}

SECTIONS TO COMPARE:
{sections_data}

Return a JSON object with this structure:
{{
    "aligned_sections": [
        {{
            "section_id": "<section ID>",
            "section_title": "<section title>",
            "original_content": "<original content>",
            "amended_content": "<amended content>",
            "has_changes": true/false,
            "is_removed": false,
            "is_added": false
        }}
    ]
}}

RULES:
- Sections are Roman numeral top-level ONLY (I, II, III, IV, V, VI, VII)
- Do NOT create separate entries for clauses (2.1, 2.2, etc.) - include all clauses within their parent section
- has_changes: true ONLY for changes in values (values, dates, terms, conditions) and false for same content in meaning or OCR artifacts ("|" characters)
- is_removed: true if section exists in original but NOT in amended (use empty string for amended_content)
- is_added: true if section exists in amended but NOT in original (use empty string for original_content)
- Ignore minor formatting differences (quotes, whitespace)
- Focus on numbers, dates, durations, and legal terms to detect changes in values
"""

# Keep roman numerals for section IDs (e.g., "II.1" instead of "2.1")

async def align_sections_with_llm(
    original_index: Dict[str, Section],
    amended_index: Dict[str, Section],
    original_title: str,
    amended_title: str,
    session: TracingSession
) -> Optional[ContextualizationOutput]:
    """
    Step 2: Use LLM to intelligently align sections and detect changes.
    
    The LLM can distinguish real changes from OCR artifacts and formatting.
    
    Args:
        original_index: Section map from original contract
        amended_index: Section map from amended contract
        original_title: Title of original document
        amended_title: Title of amended document
        session: TracingSession for Langfuse monitoring
        
    Returns:
        ContextualizationOutput with aligned sections
    """
    print(f"🔄 Aligning sections with LLM: {len(original_index)} original ↔ {len(amended_index)} amended")
    
    # Identify section sets
    original_ids = set(original_index.keys())
    amended_ids = set(amended_index.keys())
    only_in_original = list(original_ids - amended_ids)
    only_in_amended = list(amended_ids - original_ids)
    common_ids = original_ids & amended_ids
    
    # Prepare section comparison data for LLM
    sections_data = []
    for section_id in sorted(common_ids):
        orig_section = original_index[section_id]
        amend_section = amended_index[section_id]
        
        orig_content = get_section_content(orig_section)
        amend_content = get_section_content(amend_section)
        
        title = orig_section.title or amend_section.title or f"Section {section_id}"
        
        sections_data.append({
            "section_id": section_id,
            "section_title": title,
            "original_content": orig_content,
            "amended_content": amend_content
        })
    
    # Create generation span for tracing
    span = session.create_generation(
        name="align_sections_llm",
        model=LANGFUSE_MODEL_NAME,
        input_data={
            "original_title": original_title,
            "amended_title": amended_title,
            "sections_count": len(sections_data)
        },
        metadata={"agent": "contextualization_agent", "operation": "alignment"}
    )
    
    try:
        # Format prompt
        prompt = ALIGNMENT_PROMPT.format(
            original_title=original_title,
            amended_title=amended_title,
            sections_data=json.dumps(sections_data, indent=2)
        )
        
        # LLM call
        completion = await client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        # Track usage
        if hasattr(completion, 'usage') and completion.usage:
            span.update(
                usage={
                    "input": completion.usage.prompt_tokens,
                    "output": completion.usage.completion_tokens,
                    "total": completion.usage.total_tokens
                }
            )
        
        response_content = completion.choices[0].message.content
        parsed = json.loads(response_content)
        
        # Build aligned sections with full content
        aligned_sections = []
        llm_results = {item["section_id"]: item for item in parsed.get("aligned_sections", [])}
        
        for section_id in sorted(common_ids):
            orig_section = original_index[section_id]
            amend_section = amended_index[section_id]
            
            orig_content = get_section_content(orig_section)
            amend_content = get_section_content(amend_section)
            
            title = orig_section.title or amend_section.title or f"Section {section_id}"
            
            # Get LLM's determination
            llm_result = llm_results.get(section_id, {})
            has_changes = llm_result.get("has_changes", False)
            
            aligned_sections.append(SectionAlignment(
                section_id=section_id,
                section_title=title,
                original_content=orig_content,
                amended_content=amend_content,
                has_changes=has_changes
            ))
        
        # Add sections only in original (removed)
        for section_id in sorted(only_in_original):
            section = original_index[section_id]
            aligned_sections.append(SectionAlignment(
                section_id=section_id,
                section_title=section.title or f"Section {section_id}",
                original_content=get_section_content(section),
                amended_content="[SECTION REMOVED]",
                has_changes=True
            ))
        
        # Add sections only in amended (added)
        for section_id in sorted(only_in_amended):
            section = amended_index[section_id]
            aligned_sections.append(SectionAlignment(
                section_id=section_id,
                section_title=section.title or f"Section {section_id}",
                original_content="[SECTION ADDED]",
                amended_content=get_section_content(section),
                has_changes=True
            ))
        
        # Sort by section ID
        aligned_sections.sort(key=lambda x: x.section_id)
        
        changed_count = sum(1 for s in aligned_sections if s.has_changes)
        print(f"✅ LLM alignment complete: {len(aligned_sections)} sections, {changed_count} with changes")
        
        span.update(
            output={
                "success": True,
                "aligned_count": len(aligned_sections),
                "changed_count": changed_count
            },
            status_message="success"
        )
        
        return ContextualizationOutput(
            original_title=original_title,
            amended_title=amended_title,
            aligned_sections=aligned_sections,
            sections_only_in_original=only_in_original,
            sections_only_in_amended=only_in_amended
        )
        
    except Exception as e:
        print(f"❌ LLM alignment error: {e}")
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
    1. Build section indexes from pages (programmatic, fast)
    2. LLM-powered alignment to detect meaningful changes
    
    Langfuse tracing is REQUIRED for all operations.
    """
    
    def __init__(self, session: TracingSession):
        self.session = session
        self.original_index: Dict[str, Section] = {}
        self.amended_index: Dict[str, Section] = {}
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
        
        Flow:
        1. Build section indexes from pages (programmatic)
        2. LLM alignment to detect meaningful changes
        
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
            # Step 1: Build section indexes (programmatic)
            print("\n📋 STEP 1: Building Section Indexes")
            print("-"*40)
            
            self.original_index, original_title = build_section_index(original_pages)
            self.amended_index, amended_title = build_section_index(amended_pages)
            
            print(f"✅ Original: {len(self.original_index)} sections")
            print(f"✅ Amended: {len(self.amended_index)} sections")
            
            if not self.original_index:
                print("❌ No sections found in original document")
                return None
            
            if not self.amended_index:
                print("❌ No sections found in amended document")
                return None
            
            # Step 2: LLM-powered alignment
            print("\n🤖 STEP 2: LLM Alignment")
            print("-"*40)
            
            self.alignment = await align_sections_with_llm(
                self.original_index,
                self.amended_index,
                original_title,
                amended_title,
                self.session
            )
            
            if not self.alignment:
                print("❌ Failed to align sections")
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
                    "original_sections": len(self.original_index),
                    "amended_sections": len(self.amended_index),
                    "aligned_sections": len(self.alignment.aligned_sections),
                    "changed_sections": len(self.get_changed_sections())
                },
                status_message="success"
            )
            
            return self.alignment
            
        except Exception as e:
            print(f"❌ Agent error: {e}")
            import traceback
            traceback.print_exc()
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
        
        validation_span = self.session.create_span(
            name="validate_contextualization_output",
            input_data={"alignment_sections": len(self.alignment.aligned_sections)},
            metadata={"agent": "contextualization_agent", "operation": "validation"}
        )
        
        try:
            ContextualizationOutput.model_validate(self.alignment.model_dump())
            validation_span.update(output={"valid": True}, status_message="valid")
            return True
        except ValidationError as e:
            print(f"❌ Output validation failed: {e}")
            validation_span.update(output={"valid": False, "error": str(e)}, status_message="invalid")
            return False
        finally:
            validation_span.end()
