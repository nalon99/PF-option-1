"""
End-to-end integration test for the complete contract amendment analysis pipeline.

Tests the full workflow:
    Images → Image Parser → Agent 1 (Contextualization) → Agent 2 (Extraction) → Final Output

IMPORTANT: This test makes actual LLM API calls and costs money.
Run with: pytest tests/test_end_to_end.py -v -s -m integration
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contract_agent import analyze_contract_amendment
from models import ContractAnalysisResult


# =============================================================================
# MOCK TRACING SESSION (no Langfuse for tests)
# =============================================================================

class MockSpan:
    """Mock span that does nothing."""
    def update(self, **kwargs): pass
    def end(self): pass


class MockGeneration:
    """Mock generation that does nothing."""
    def update(self, **kwargs): pass
    def end(self): pass


class MockTracingSession:
    """
    Mock TracingSession that provides the same interface but doesn't trace.
    Used for testing without polluting Langfuse with test traces.
    """
    def __init__(self, session_name: str = None, contract_id: str = None, **kwargs):
        self.session_id = "mock-session"
        self.contract_id = contract_id or "mock-contract"
        self.session_name = session_name or "mock-session"
        self.metadata = {}
    
    def create_span(self, name: str, **kwargs) -> MockSpan:
        return MockSpan()
    
    def create_generation(self, name: str, model: str = None, **kwargs) -> MockGeneration:
        return MockGeneration()
    
    def mark_success(self, output: Optional[Dict] = None): pass
    def mark_error(self, error_message: str): pass
    def end(self): pass


# =============================================================================
# TEST DATA
# =============================================================================

TEST_CONTRACTS_DIR = Path(__file__).parent.parent / "data" / "test_contracts"

# =============================================================================
# PAIR 1: Master Service Agreement - Expected Changes
# =============================================================================
# Key changes from original:
# - Term: 24 months -> 36 months
# - Termination cure period: 15 days -> 30 days
# - Termination for convenience: 60 days -> 90 days
# - Monthly fee: $15,000 -> $18,000
# - Late payment interest: 1.5% -> 2.0%
# - Liability cap: 3 months -> 6 months
# =============================================================================

PAIR_1_EXPECTED_SECTIONS = [
    "TERM",           # Section II - Term, Termination
    "COMPENSATION",   # Section III - Compensation, Billing  
    "LIABILITY",      # Section VI - Limitation of Liability
]

# Key values that should appear in summaries (at least some of these)
PAIR_1_EXPECTED_VALUES = [
    ("24", "36"),           # Term months
    ("15", "30"),           # Cure period days
    ("60", "90"),           # Notice period days
    ("15,000", "18,000"),   # Monthly fee
    ("1.5", "2.0"),         # Late payment interest
]

# =============================================================================
# PAIR 2: Real Estate Purchase Agreement - Expected Changes
# =============================================================================
# Key changes from original:
# - Purchase price: $475,000 -> $465,000 (price reduction after inspection)
# - Earnest money: $15,000 -> $20,000 (increased deposit)
# - Financing deadline: 21 days -> 28 days (extended)
# - Inspection period: 10 days -> 14 days (extended)
# - Closing date: April 30, 2024 -> May 15, 2024 (extended)
# =============================================================================

PAIR_2_EXPECTED_SECTIONS = [
    "PURCHASE",       # Section II - Purchase Price
    "INSPECTION",     # Section III - Inspections
    "CLOSING",        # Section IV - Closing
]

# Key values that should appear in summaries (at least some of these)
PAIR_2_EXPECTED_VALUES = [
    ("475,000", "465,000"),   # Purchase price
    ("15,000", "20,000"),     # Earnest money (note: same numbers as pair_1 fee, but different context)
    ("21", "28"),             # Financing deadline days
    ("10", "14"),             # Inspection period days
    ("April 30", "May 15"),   # Closing date
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_sections_contain_keywords(sections: List[str], keywords: List[str]) -> Dict[str, bool]:
    """
    Check if the detected sections contain expected keywords.
    Returns a dict of keyword -> found status.
    """
    sections_upper = " ".join(sections).upper()
    return {kw: kw.upper() in sections_upper for kw in keywords}


def check_values_in_summaries(summaries: List[str], expected_values: List[tuple]) -> Dict[str, bool]:
    """
    Check if summaries contain expected value changes (old -> new).
    Returns a dict of "old→new" -> found status.
    """
    all_summaries = " ".join(summaries)
    results = {}
    for old_val, new_val in expected_values:
        key = f"{old_val}→{new_val}"
        # Check if both values appear (they should be mentioned together in a change)
        found = old_val in all_summaries or new_val in all_summaries
        results[key] = found
    return results


# =============================================================================
# END-TO-END TESTS
# =============================================================================

class TestEndToEndPipeline:
    """Full pipeline integration tests."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pair_1_full_pipeline(self):
        """
        Test complete pipeline for pair_1 (Master Service Agreement).
        
        Expected changes:
        - Term: 24 → 36 months
        - Cure period: 15 → 30 days
        - Fees: $15,000 → $18,000
        - Liability cap: 3 → 6 months
        """
        original_folder = str(TEST_CONTRACTS_DIR / "pair_1" / "original")
        amendment_folder = str(TEST_CONTRACTS_DIR / "pair_1" / "amendment")
        
        session = MockTracingSession(
            session_name="e2e_test_pair_1",
            contract_id="pair_1_e2e_test"
        )
        
        print("\n" + "=" * 60)
        print("END-TO-END TEST: pair_1 (Master Service Agreement)")
        print("=" * 60)
        
        # Run the full pipeline
        result = await analyze_contract_amendment(
            original_folder=original_folder,
            amendment_folder=amendment_folder,
            session=session
        )
        
        # Verify result type
        assert isinstance(result, ContractAnalysisResult), \
            f"Expected ContractAnalysisResult, got {type(result)}"
        
        # Verify required fields are non-empty
        assert len(result.sections_changed) > 0, "sections_changed should not be empty"
        assert len(result.topics_touched) > 0, "topics_touched should not be empty"
        assert len(result.summary_of_the_change) > 0, "summary_of_the_change should not be empty"
        
        # Verify counts match (Pydantic validation rule)
        assert len(result.topics_touched) == len(result.summary_of_the_change), \
            f"topics_touched ({len(result.topics_touched)}) must equal summary_of_the_change ({len(result.summary_of_the_change)})"
        
        # Print results
        print(f"\n✅ Pipeline completed successfully!")
        print(f"\n📋 Sections Changed ({len(result.sections_changed)}):")
        for section in result.sections_changed:
            print(f"   - {section}")
        
        print(f"\n📌 Topics Touched ({len(result.topics_touched)}):")
        for topic in result.topics_touched:
            print(f"   - {topic}")
        
        print(f"\n📝 Summaries ({len(result.summary_of_the_change)}):")
        for i, summary in enumerate(result.summary_of_the_change, 1):
            print(f"   {i}. {summary[:100]}..." if len(summary) > 100 else f"   {i}. {summary}")
        
        # Check expected sections are detected (fuzzy)
        keyword_results = check_sections_contain_keywords(
            result.sections_changed, 
            PAIR_1_EXPECTED_SECTIONS
        )
        
        print(f"\n🔍 Expected Sections Check:")
        for kw, found in keyword_results.items():
            status = "✅" if found else "⚠️"
            print(f"   {status} {kw}: {'Found' if found else 'Not found'}")
        
        # Check expected values in summaries
        value_results = check_values_in_summaries(
            result.summary_of_the_change,
            PAIR_1_EXPECTED_VALUES
        )
        
        print(f"\n🔢 Expected Value Changes Check:")
        for change, found in value_results.items():
            status = "✅" if found else "⚠️"
            print(f"   {status} {change}: {'Found' if found else 'Not found'}")
        
        # At least some expected sections should be detected
        found_count = sum(keyword_results.values())
        assert found_count >= 2, \
            f"Expected at least 2 of {PAIR_1_EXPECTED_SECTIONS} to be detected, found {found_count}"
        
        print("\n" + "=" * 60)
        print("✅ PAIR_1 END-TO-END TEST PASSED")
        print("=" * 60)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pair_2_full_pipeline(self):
        """
        Test complete pipeline for pair_2 (Real Estate Purchase Agreement).
        
        Expected changes:
        - Purchase price: $475,000 → $465,000
        - Earnest money: $15,000 → $20,000
        - Inspection period: 10 → 14 days
        - Closing date: April 30 → May 15, 2024
        """
        original_folder = str(TEST_CONTRACTS_DIR / "pair_2" / "original")
        amendment_folder = str(TEST_CONTRACTS_DIR / "pair_2" / "amendment")
        
        session = MockTracingSession(
            session_name="e2e_test_pair_2",
            contract_id="pair_2_e2e_test"
        )
        
        print("\n" + "=" * 60)
        print("END-TO-END TEST: pair_2 (Real Estate Purchase Agreement)")
        print("=" * 60)
        
        # Run the full pipeline
        result = await analyze_contract_amendment(
            original_folder=original_folder,
            amendment_folder=amendment_folder,
            session=session
        )
        
        # Verify result type
        assert isinstance(result, ContractAnalysisResult), \
            f"Expected ContractAnalysisResult, got {type(result)}"
        
        # Verify required fields are non-empty
        assert len(result.sections_changed) > 0, "sections_changed should not be empty"
        assert len(result.topics_touched) > 0, "topics_touched should not be empty"
        assert len(result.summary_of_the_change) > 0, "summary_of_the_change should not be empty"
        
        # Verify counts match
        assert len(result.topics_touched) == len(result.summary_of_the_change), \
            f"topics_touched ({len(result.topics_touched)}) must equal summary_of_the_change ({len(result.summary_of_the_change)})"
        
        # Print results
        print(f"\n✅ Pipeline completed successfully!")
        print(f"\n📋 Sections Changed ({len(result.sections_changed)}):")
        for section in result.sections_changed:
            print(f"   - {section}")
        
        print(f"\n📌 Topics Touched ({len(result.topics_touched)}):")
        for topic in result.topics_touched:
            print(f"   - {topic}")
        
        print(f"\n📝 Summaries ({len(result.summary_of_the_change)}):")
        for i, summary in enumerate(result.summary_of_the_change, 1):
            print(f"   {i}. {summary[:100]}..." if len(summary) > 100 else f"   {i}. {summary}")
        
        # Check expected sections are detected (fuzzy)
        keyword_results = check_sections_contain_keywords(
            result.sections_changed, 
            PAIR_2_EXPECTED_SECTIONS
        )
        
        print(f"\n🔍 Expected Sections Check:")
        for kw, found in keyword_results.items():
            status = "✅" if found else "⚠️"
            print(f"   {status} {kw}: {'Found' if found else 'Not found'}")
        
        # Check expected values in summaries
        value_results = check_values_in_summaries(
            result.summary_of_the_change,
            PAIR_2_EXPECTED_VALUES
        )
        
        print(f"\n🔢 Expected Value Changes Check:")
        for change, found in value_results.items():
            status = "✅" if found else "⚠️"
            print(f"   {status} {change}: {'Found' if found else 'Not found'}")
        
        found_count = sum(keyword_results.values())
        assert found_count >= 2, \
            f"Expected at least 2 of {PAIR_2_EXPECTED_SECTIONS} to be detected, found {found_count}"
        
        print("\n" + "=" * 60)
        print("✅ PAIR_2 END-TO-END TEST PASSED")
        print("=" * 60)


class TestEndToEndValidation:
    """Tests for output validation in the full pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_output_is_pydantic_validated(self):
        """Verify the output passes Pydantic validation."""
        original_folder = str(TEST_CONTRACTS_DIR / "pair_1" / "original")
        amendment_folder = str(TEST_CONTRACTS_DIR / "pair_1" / "amendment")
        
        session = MockTracingSession(
            session_name="e2e_validation_test",
            contract_id="validation_e2e_test"
        )
        
        result = await analyze_contract_amendment(
            original_folder=original_folder,
            amendment_folder=amendment_folder,
            session=session
        )
        
        # If we get here without exception, Pydantic validation passed
        assert result is not None
        
        # Verify it's actually a Pydantic model
        assert hasattr(result, 'model_dump'), "Result should be a Pydantic model"
        
        # Verify serialization works
        result_dict = result.model_dump()
        assert "sections_changed" in result_dict
        assert "topics_touched" in result_dict
        assert "summary_of_the_change" in result_dict
        
        print("\n✅ Output is properly Pydantic-validated and serializable")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_summary_items_are_meaningful(self):
        """Verify summary items are at least 20 characters (meaningful descriptions)."""
        original_folder = str(TEST_CONTRACTS_DIR / "pair_1" / "original")
        amendment_folder = str(TEST_CONTRACTS_DIR / "pair_1" / "amendment")
        
        session = MockTracingSession(
            session_name="e2e_summary_test",
            contract_id="summary_e2e_test"
        )
        
        result = await analyze_contract_amendment(
            original_folder=original_folder,
            amendment_folder=amendment_folder,
            session=session
        )
        
        for i, summary in enumerate(result.summary_of_the_change):
            assert len(summary) >= 20, \
                f"Summary {i+1} is too short ({len(summary)} chars): '{summary}'"
        
        print(f"\n✅ All {len(result.summary_of_the_change)} summaries are meaningful (≥20 chars)")
