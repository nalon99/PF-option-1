"""
Pydantic models for contract analysis with hierarchical document structure.

Designed for:
- Preserving document hierarchy from multimodal image parsing
- Supporting agent contextualization and extraction tasks
- Validating final analysis output
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# HIERARCHICAL DOCUMENT STRUCTURE (for Image Parser output)
# =============================================================================

class Clause(BaseModel):
    """Lowest level of document hierarchy - individual clause or paragraph."""
    
    id: str = Field(
        ...,
        min_length=1,
        description="Clause identifier (e.g., '1.1', '2.3.1', 'a', 'b')"
    )
    title: Optional[str] = Field(
        default=None,
        description="Clause title if present (e.g., 'Scope of Work')"
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Full text content of the clause"
    )


class Section(BaseModel):
    """Mid-level hierarchy - a section containing clauses or subsections."""
    
    id: str = Field(
        ...,
        min_length=1,
        description="Section identifier (e.g., 'I', 'II', '1', '2')"
    )
    title: str = Field(
        default="",
        description="Section title (e.g., 'TERM AND TERMINATION'). May be empty for continued sections."
    )
    content: Optional[str] = Field(
        default=None,
        description="Section preamble text if any (before clauses)"
    )
    clauses: List[Clause] = Field(
        default_factory=list,
        description="List of clauses within this section"
    )
    subsections: List["Section"] = Field(
        default_factory=list,
        description="Nested subsections if any"
    )


class ParsedContractDocument(BaseModel):
    """
    Complete hierarchical representation of a parsed contract document.
    
    This is the output from the multimodal image parser, preserving
    the full document structure for downstream agent processing.
    """
    
    title: str = Field(
        ...,
        min_length=1,
        description="Document title (e.g., 'MASTER SERVICE AGREEMENT')"
    )
    effective_date: Optional[str] = Field(
        default=None,
        description="Contract effective date if found"
    )
    parties: List[str] = Field(
        default_factory=list,
        description="List of parties involved (e.g., ['Alpha Corp (Client)', 'Beta Solutions LLC (Provider)'])"
    )
    preamble: Optional[str] = Field(
        default=None,
        description="Introductory text before sections"
    )
    sections: List[Section] = Field(
        ...,
        min_length=1,
        description="List of top-level sections in the document"
    )
    
    @field_validator('sections')
    @classmethod
    def validate_sections_not_empty(cls, v: List[Section]) -> List[Section]:
        if not v:
            raise ValueError("Document must have at least one section")
        return v
    
    def get_section_by_id(self, section_id: str) -> Optional[Section]:
        """Helper to find a section by its ID."""
        for section in self.sections:
            if section.id == section_id:
                return section
        return None
    
    def get_all_section_ids(self) -> List[str]:
        """Get all section IDs for comparison."""
        return [s.id for s in self.sections]


# =============================================================================
# MULTI-PAGE DOCUMENT (combines multiple parsed pages)
# =============================================================================

class ParsedContractPage(BaseModel):
    """Single page extraction before merging into full document."""
    
    page_number: int = Field(..., gt=0, description="Page number (1-indexed)")
    sections: List[Section] = Field(
        default_factory=list,
        description="Sections found on this page (may be partial)"
    )
    raw_content: Optional[str] = Field(
        default=None,
        description="Raw text fallback if structure parsing fails"
    )


# =============================================================================
# AGENT OUTPUT MODELS
# =============================================================================

class SectionAlignment(BaseModel):
    """Alignment between corresponding sections in original and amended documents."""
    
    section_id: str = Field(
        ...,
        description="Section identifier (e.g., 'II', '2.1')"
    )
    section_title: str = Field(
        ...,
        description="Section title for readability"
    )
    original_content: str = Field(
        ...,
        description="Content from original document"
    )
    amended_content: str = Field(
        ...,
        description="Content from amended document"
    )
    has_changes: bool = Field(
        ...,
        description="Whether this section has differences"
    )


class ContextualizationOutput(BaseModel):
    """
    Output from Agent 1 (Contextualization Agent).
    
    Provides aligned structure between original and amended documents
    for Agent 2 to process.
    """
    
    original_title: str = Field(..., description="Original document title")
    amended_title: str = Field(..., description="Amended document title")
    aligned_sections: List[SectionAlignment] = Field(
        ...,
        min_length=1,
        description="List of aligned sections between documents"
    )
    sections_only_in_original: List[str] = Field(
        default_factory=list,
        description="Section IDs that exist only in original"
    )
    sections_only_in_amended: List[str] = Field(
        default_factory=list,
        description="Section IDs that exist only in amended"
    )


class ContractAnalysisResult(BaseModel):
    """
    Final output from Agent 2 (Extraction Agent).
    
    This is the validated structured output for the contract comparison,
    matching the expected format defined in the project requirements.
    """
    
    sections_changed: List[str] = Field(
        ...,
        min_length=1,
        description="List of section identifiers that were modified (e.g., 'II. TERM, TERMINATION')"
    )
    topics_touched: List[str] = Field(
        ...,
        min_length=1,
        description="List of business/legal topic categories affected (e.g., 'Payment Terms', 'Liability')"
    )
    summary_of_the_change: List[str] = Field(
        ...,
        min_length=1,
        description="List of detailed change descriptions with section references"
    )
    
    @field_validator('sections_changed', 'topics_touched')
    @classmethod
    def validate_non_empty_strings(cls, v: List[str]) -> List[str]:
        """Ensure list items are non-empty strings."""
        for i, item in enumerate(v):
            if not item or not item.strip():
                raise ValueError(f"Item at index {i} cannot be empty or whitespace")
        return v
    
    @field_validator('summary_of_the_change')
    @classmethod
    def validate_summary_items(cls, v: List[str]) -> List[str]:
        """Ensure summary items are meaningful descriptions."""
        for i, item in enumerate(v):
            if not item or not item.strip():
                raise ValueError(f"Summary item at index {i} cannot be empty")
            if len(item.strip()) < 20:
                raise ValueError(f"Summary item at index {i} is too short (min 20 chars)")
        return v

    def model_post_init(self, __context) -> None:
        """Post-initialization validation for cross-field consistency."""
        if len(self.topics_touched) != len(self.summary_of_the_change):
            raise ValueError(
                f"Number of topics_touched ({len(self.topics_touched)}) must match "
                f"number of summary_of_the_change items ({len(self.summary_of_the_change)})"
            )
