"""
Pydantic Validation Tests

Tests for the ContractAnalysisResult model:
- Valid outputs pass validation
- Invalid outputs raise appropriate errors
- Field validators work correctly
- Cross-field validation (topics_touched count == summary_of_the_change count)
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pydantic import ValidationError
from models import ContractAnalysisResult, Section, Clause, ParsedContractPage


class TestContractAnalysisResultValidation:
    """Tests for ContractAnalysisResult Pydantic model validation."""

    def test_valid_output_passes_validation(self):
        """Test that a valid output passes Pydantic validation."""
        valid_data = {
            "sections_changed": [
                "II. TERM, TERMINATION, AND SUSPENSION",
                "III. COMPENSATION, BILLING, AND EXPENSES"
            ],
            "topics_touched": [
                "Contract Duration and Termination",
                "Payment Terms"
            ],
            "summary_of_the_change": [
                "Contract term extended from 24 to 36 months, cure period extended from 15 to 30 days (Sec. 2.1, 2.2).",
                "Monthly fee increased from $15,000 to $18,000 and late payment interest increased from 1.5% to 2.0% (Sec. 3.1, 3.2)."
            ]
        }
        
        # Should not raise any exception
        result = ContractAnalysisResult(**valid_data)
        
        assert len(result.sections_changed) == 2
        assert len(result.topics_touched) == 2
        assert len(result.summary_of_the_change) == 2

    def test_empty_sections_changed_fails(self):
        """Test that empty sections_changed list fails validation."""
        invalid_data = {
            "sections_changed": [],  # Empty - should fail
            "topics_touched": ["Payment Terms"],
            "summary_of_the_change": ["Fee increased from $15,000 to $18,000 per month."]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ContractAnalysisResult(**invalid_data)
        
        assert "sections_changed" in str(exc_info.value)

    def test_empty_topics_touched_fails(self):
        """Test that empty topics_touched list fails validation."""
        invalid_data = {
            "sections_changed": ["II. TERM"],
            "topics_touched": [],  # Empty - should fail
            "summary_of_the_change": ["Term extended from 24 to 36 months."]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ContractAnalysisResult(**invalid_data)
        
        assert "topics_touched" in str(exc_info.value)

    def test_empty_summary_fails(self):
        """Test that empty summary_of_the_change list fails validation."""
        invalid_data = {
            "sections_changed": ["II. TERM"],
            "topics_touched": ["Contract Duration"],
            "summary_of_the_change": []  # Empty - should fail
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ContractAnalysisResult(**invalid_data)
        
        assert "summary_of_the_change" in str(exc_info.value)

    def test_mismatched_counts_fails(self):
        """Test that mismatched topics_touched and summary_of_the_change counts fail validation."""
        invalid_data = {
            "sections_changed": ["II. TERM", "III. COMPENSATION"],
            "topics_touched": ["Contract Duration", "Payment Terms", "Extra Topic"],  # 3 items
            "summary_of_the_change": [
                "Term extended from 24 to 36 months (Sec. 2.1).",
                "Fee increased from $15,000 to $18,000 (Sec. 3.1)."
            ]  # Only 2 items - mismatch!
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ContractAnalysisResult(**invalid_data)
        
        assert "must match" in str(exc_info.value).lower() or "topics_touched" in str(exc_info.value)

    def test_whitespace_only_section_fails(self):
        """Test that whitespace-only strings in sections_changed fail validation."""
        invalid_data = {
            "sections_changed": ["II. TERM", "   "],  # Whitespace-only string
            "topics_touched": ["Contract Duration"],
            "summary_of_the_change": ["Term extended from 24 to 36 months (Sec. 2.1)."]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ContractAnalysisResult(**invalid_data)
        
        assert "empty" in str(exc_info.value).lower() or "whitespace" in str(exc_info.value).lower()

    def test_short_summary_fails(self):
        """Test that summary items under 20 characters fail validation."""
        invalid_data = {
            "sections_changed": ["II. TERM"],
            "topics_touched": ["Duration"],
            "summary_of_the_change": ["Too short"]  # Less than 20 chars
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ContractAnalysisResult(**invalid_data)
        
        assert "short" in str(exc_info.value).lower() or "20" in str(exc_info.value)

    def test_missing_required_field_fails(self):
        """Test that missing required fields fail validation."""
        invalid_data = {
            "sections_changed": ["II. TERM"],
            "topics_touched": ["Duration"]
            # Missing summary_of_the_change
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ContractAnalysisResult(**invalid_data)
        
        assert "summary_of_the_change" in str(exc_info.value)


class TestParsedContractPageValidation:
    """Tests for ParsedContractPage model validation."""

    def test_valid_page_passes(self):
        """Test that a valid parsed page passes validation."""
        valid_page = {
            "page_number": 1,
            "sections": [
                {
                    "id": "I",
                    "title": "AGREEMENT DETAILS",
                    "content": "This agreement is made between...",
                    "clauses": [
                        {
                            "id": "1.1",
                            "title": "Scope of Work",
                            "content": "The Provider agrees to perform..."
                        }
                    ],
                    "subsections": []
                }
            ],
            "raw_content": "Full page text here..."
        }
        
        page = ParsedContractPage(**valid_page)
        assert page.page_number == 1
        assert len(page.sections) == 1
        assert page.sections[0].id == "I"

    def test_invalid_page_number_fails(self):
        """Test that page_number <= 0 fails validation."""
        invalid_page = {
            "page_number": 0,  # Must be > 0
            "sections": [],
            "raw_content": "Some text"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ParsedContractPage(**invalid_page)
        
        assert "page_number" in str(exc_info.value)


class TestClauseValidation:
    """Tests for Clause model validation."""

    def test_valid_clause_passes(self):
        """Test that a valid clause passes validation."""
        clause = Clause(id="1.1", title="Scope", content="The provider shall...")
        assert clause.id == "1.1"
        assert clause.title == "Scope"

    def test_empty_clause_id_fails(self):
        """Test that empty clause ID fails validation."""
        with pytest.raises(ValidationError):
            Clause(id="", title="Scope", content="The provider shall...")

    def test_empty_clause_content_fails(self):
        """Test that empty clause content fails validation."""
        with pytest.raises(ValidationError):
            Clause(id="1.1", title="Scope", content="")
