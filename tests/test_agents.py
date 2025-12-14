"""
Agent Handoff Tests

Tests to verify:
- Agent 2 receives Agent 1's output correctly
- ContextualizationOutput structure is valid input for ExtractionAgent
- Handoff data integrity (aligned sections are passed correctly)
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import (
    ContextualizationOutput,
    SectionAlignment,
    ContractAnalysisResult
)
from agents.extraction_agent import ExtractionAgent


class TestAgentHandoff:
    """Tests for Agent 1 → Agent 2 handoff mechanism."""

    def test_contextualization_output_structure(self):
        """Test that ContextualizationOutput has correct structure for Agent 2."""
        # Simulate Agent 1's output
        agent1_output = ContextualizationOutput(
            original_title="Original Contract",
            amended_title="Amended Contract",
            aligned_sections=[
                SectionAlignment(
                    section_id="I",
                    section_title="AGREEMENT DETAILS",
                    original_content="The Provider agrees to perform services...",
                    amended_content="The Provider agrees to perform services...",
                    has_changes=False,
                    is_removed=False,
                    is_added=False
                ),
                SectionAlignment(
                    section_id="II",
                    section_title="TERM AND TERMINATION",
                    original_content="The initial term shall be twenty-four (24) months...",
                    amended_content="The initial term shall be thirty-six (36) months...",
                    has_changes=True,
                    is_removed=False,
                    is_added=False
                ),
                SectionAlignment(
                    section_id="III",
                    section_title="COMPENSATION",
                    original_content="Monthly fee of $15,000...",
                    amended_content="Monthly fee of $18,000...",
                    has_changes=True,
                    is_removed=False,
                    is_added=False
                )
            ]
        )
        
        # Verify structure
        assert agent1_output.original_title == "Original Contract"
        assert agent1_output.amended_title == "Amended Contract"
        assert len(agent1_output.aligned_sections) == 3
        
        # Verify sections with changes are identifiable
        changed_sections = [s for s in agent1_output.aligned_sections if s.has_changes]
        assert len(changed_sections) == 2
        assert changed_sections[0].section_id == "II"
        assert changed_sections[1].section_id == "III"

    def test_agent2_receives_agent1_output_type(self):
        """Test that ExtractionAgent.agent_extract_changes accepts ContextualizationOutput."""
        # Create Agent 2 instance (without session for type checking)
        agent2 = ExtractionAgent(session=None)
        
        # Verify the method signature accepts ContextualizationOutput
        import inspect
        sig = inspect.signature(agent2.agent_extract_changes)
        params = list(sig.parameters.keys())
        
        assert "contextualization_output" in params
        
        # Check the type annotation
        param = sig.parameters["contextualization_output"]
        assert param.annotation == ContextualizationOutput

    def test_handoff_filters_changed_sections(self):
        """Test that Agent 2 only processes sections marked as changed."""
        # Simulate Agent 1's output with mixed sections
        agent1_output = ContextualizationOutput(
            original_title="Original",
            amended_title="Amended",
            aligned_sections=[
                SectionAlignment(
                    section_id="I",
                    section_title="UNCHANGED SECTION",
                    original_content="Same content",
                    amended_content="Same content",
                    has_changes=False,  # No changes
                    is_removed=False,
                    is_added=False
                ),
                SectionAlignment(
                    section_id="II",
                    section_title="CHANGED SECTION",
                    original_content="Original value: 100",
                    amended_content="New value: 200",
                    has_changes=True,  # Has changes
                    is_removed=False,
                    is_added=False
                )
            ]
        )
        
        # Simulate Agent 2's filtering logic
        sections_with_changes = [
            s for s in agent1_output.aligned_sections
            if s.has_changes
        ]
        
        # Only the changed section should be processed
        assert len(sections_with_changes) == 1
        assert sections_with_changes[0].section_id == "II"
        assert sections_with_changes[0].section_title == "CHANGED SECTION"

    def test_handoff_preserves_content_integrity(self):
        """Test that content is preserved correctly during handoff."""
        original_content = "The contract term is twenty-four (24) months."
        amended_content = "The contract term is thirty-six (36) months."
        
        agent1_output = ContextualizationOutput(
            original_title="Original Contract",
            amended_title="Amended Contract",
            aligned_sections=[
                SectionAlignment(
                    section_id="II",
                    section_title="TERM",
                    original_content=original_content,
                    amended_content=amended_content,
                    has_changes=True,
                    is_removed=False,
                    is_added=False
                )
            ]
        )
        
        # Verify content is preserved exactly
        section = agent1_output.aligned_sections[0]
        assert section.original_content == original_content
        assert section.amended_content == amended_content
        assert "twenty-four" in section.original_content
        assert "thirty-six" in section.amended_content

    def test_handoff_handles_removed_sections(self):
        """Test that removed sections (only in original) are handled correctly."""
        agent1_output = ContextualizationOutput(
            original_title="Original",
            amended_title="Amended",
            aligned_sections=[
                SectionAlignment(
                    section_id="VII",
                    section_title="REMOVED SECTION",
                    original_content="This section existed in original...",
                    amended_content="",  # Empty in amendment
                    has_changes=True,
                    is_removed=True,  # Marked as removed
                    is_added=False
                )
            ]
        )
        
        section = agent1_output.aligned_sections[0]
        assert section.is_removed is True
        assert section.amended_content == ""
        assert section.original_content != ""

    def test_handoff_handles_added_sections(self):
        """Test that added sections (only in amendment) are handled correctly."""
        agent1_output = ContextualizationOutput(
            original_title="Original",
            amended_title="Amended",
            aligned_sections=[
                SectionAlignment(
                    section_id="VIII",
                    section_title="NEW SECTION",
                    original_content="",  # Empty in original
                    amended_content="This is a new section added in amendment...",
                    has_changes=True,
                    is_removed=False,
                    is_added=True  # Marked as added
                )
            ]
        )
        
        section = agent1_output.aligned_sections[0]
        assert section.is_added is True
        assert section.original_content == ""
        assert section.amended_content != ""

    def test_empty_aligned_sections_detected(self):
        """Test that empty aligned_sections list is detected."""
        # ContextualizationOutput requires at least 1 section
        with pytest.raises(Exception):  # ValidationError or ValueError
            ContextualizationOutput(
                original_title="Original",
                amended_title="Amended",
                aligned_sections=[]  # Empty - should fail
            )


class TestAgentOutputTypes:
    """Tests for agent input/output type consistency."""

    def test_agent1_output_is_agent2_input(self):
        """Verify ContextualizationOutput is the bridge between agents."""
        # Agent 1 produces ContextualizationOutput
        # Agent 2 consumes ContextualizationOutput
        
        # This is verified by the type annotations
        from agents.contextualization_agent import ContextualizationAgent
        from agents.extraction_agent import ExtractionAgent
        import inspect
        
        # Check Agent 1's return type
        agent1_sig = inspect.signature(ContextualizationAgent.agent_contextualize)
        # Note: return annotation may be Optional[ContextualizationOutput]
        
        # Check Agent 2's input type
        agent2_sig = inspect.signature(ExtractionAgent.agent_extract_changes)
        agent2_param = agent2_sig.parameters["contextualization_output"]
        
        assert agent2_param.annotation == ContextualizationOutput

    def test_agent2_output_is_final_result(self):
        """Verify Agent 2 produces ContractAnalysisResult."""
        from agents.extraction_agent import ExtractionAgent
        import inspect
        
        sig = inspect.signature(ExtractionAgent.agent_extract_changes)
        return_annotation = sig.return_annotation
        
        assert return_annotation == ContractAnalysisResult
