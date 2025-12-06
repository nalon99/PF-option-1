"""
Pydantic models for contract analysis output validation.

Defines structured output models with:
- Field constraints (min_length for lists/strings)
- Type hints for all fields
- Custom validators for business logic
- Graceful error handling
"""

from typing import List
from pydantic import BaseModel, Field, field_validator


class ParsedContractPage(BaseModel):
    """Model for a single parsed contract page from image."""
    
    page_number: int = Field(..., gt=0, description="Page number (1-indexed)")
    sections: List[str] = Field(
        default_factory=list,
        description="List of section titles found on this page"
    )
    content: str = Field(
        ..., 
        min_length=1,
        description="Full extracted text content from the page"
    )
    
    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace only")
        return v


class ParsedContract(BaseModel):
    """Model for a complete parsed contract (all images pages combined)."""
    
    title: str = Field(
        ...,
        min_length=1,
        description="Contract title or type"
    )
    pages: List[ParsedContractPage] = Field(
        ...,
        min_length=1,
        description="List of parsed pages"
    )
    full_text: str = Field(
        ...,
        min_length=1,
        description="Combined text from all pages"
    )


class ContractAnalysisResult(BaseModel):
    """
    Model for the final contract comparison analysis output.
    
    This is the structured output returned by the agent system after
    comparing an original contract with its amendment.
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
    
    @field_validator('topics_touched')
    @classmethod
    def validate_topics_match_sections(cls, v: List[str], info) -> List[str]:
        """Validate that number of topics aligns with number of summary items."""
        # This validator runs after validate_non_empty_strings
        return v

    def model_post_init(self, __context) -> None:
        """Post-initialization validation for cross-field consistency."""
        if len(self.topics_touched) != len(self.summary_of_the_change):
            raise ValueError(
                f"Number of topics_touched ({len(self.topics_touched)}) must match "
                f"number of summary_of_the_change items ({len(self.summary_of_the_change)})"
            )
