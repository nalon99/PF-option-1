"""
Integration test for image parser accuracy.

This test parses contract images and compares the output against ground truth
to verify ≥95% text accuracy.

IMPORTANT: This test makes actual LLM API calls and costs money.
Run with: pytest tests/test_accuracy.py -v -s
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image_parser import parse_contract_folder


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
        """Create a no-op span."""
        return MockSpan()
    
    def create_generation(self, name: str, model: str = None, **kwargs) -> MockGeneration:
        """Create a no-op generation."""
        return MockGeneration()
    
    def mark_success(self, output: Optional[Dict] = None):
        """No-op success marker."""
        pass
    
    def mark_error(self, error_message: str):
        """No-op error marker."""
        pass
    
    def end(self):
        """No-op end."""
        pass


# Test data paths
TEST_CONTRACTS_DIR = Path(__file__).parent.parent / "data" / "test_contracts"


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.
    
    Edit distance = minimum number of single-character edits
    (insertions, deletions, substitutions) required to transform s1 into s2.
    
    Uses dynamic programming with O(min(m,n)) space optimization.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 if substitution needed
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def word_levenshtein_distance(words1: list, words2: list) -> int:
    """
    Calculate Levenshtein edit distance at the word level.
    
    Edit distance = minimum number of word-level edits
    (insertions, deletions, substitutions) required to transform words1 into words2.
    """
    if len(words1) < len(words2):
        return word_levenshtein_distance(words2, words1)
    
    if len(words2) == 0:
        return len(words1)
    
    previous_row = range(len(words2) + 1)
    
    for i, w1 in enumerate(words1):
        current_row = [i + 1]
        for j, w2 in enumerate(words2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (w1.lower() != w2.lower())
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_accuracy(parsed_text: str, ground_truth: str) -> dict:
    """
    Calculate text accuracy between parsed output and ground truth.
    
    Uses formal CER (Character Error Rate) and WER (Word Error Rate):
    - CER = (substitutions + insertions + deletions) / total_reference_characters
    - WER = (substitutions + insertions + deletions) / total_reference_words
    - Accuracy = 1 - Error Rate
    
    Args:
        parsed_text: Text extracted by the image parser
        ground_truth: Expected text from ground_truth.txt
        
    Returns:
        Dictionary with accuracy metrics
    """
    # Normalize texts for comparison
    def normalize(text: str) -> str:
        # Remove ** markdown bold markers
        text = text.replace("**", "")
        # Normalize whitespace
        text = " ".join(text.split())
        # Normalize quotes
        text = text.replace(""", '"').replace(""", '"')
        text = text.replace("'", "'").replace("'", "'")
        return text.strip()
    
    parsed_normalized = normalize(parsed_text)
    truth_normalized = normalize(ground_truth)
    
    # Character Error Rate (CER)
    # CER = edit_distance / total_reference_characters
    char_edit_distance = levenshtein_distance(parsed_normalized, truth_normalized)
    total_reference_chars = len(truth_normalized)
    
    if total_reference_chars > 0:
        cer = char_edit_distance / total_reference_chars
        char_accuracy = max(0, 1 - cer)  # Clamp to 0 if CER > 1
    else:
        cer = 0
        char_accuracy = 1.0
    
    # Word Error Rate (WER)
    # WER = word_edit_distance / total_reference_words
    parsed_words = parsed_normalized.split()
    truth_words = truth_normalized.split()
    
    word_edit_distance = word_levenshtein_distance(parsed_words, truth_words)
    total_reference_words = len(truth_words)
    
    if total_reference_words > 0:
        wer = word_edit_distance / total_reference_words
        word_accuracy = max(0, 1 - wer)  # Clamp to 0 if WER > 1
    else:
        wer = 0
        word_accuracy = 1.0
    
    return {
        "cer": cer,
        "wer": wer,
        "character_accuracy": char_accuracy,
        "word_accuracy": word_accuracy,
        "char_edit_distance": char_edit_distance,
        "word_edit_distance": word_edit_distance,
        "parsed_char_count": len(parsed_normalized),
        "truth_char_count": total_reference_chars,
        "parsed_word_count": len(parsed_words),
        "truth_word_count": total_reference_words,
    }


def load_ground_truth(pair_name: str, contract_type: str) -> str:
    """Load ground truth text for a contract."""
    path = TEST_CONTRACTS_DIR / pair_name / contract_type / "ground_truth.txt"
    if not path.exists():
        raise FileNotFoundError(f"Ground truth not found: {path}")
    return path.read_text(encoding="utf-8")


def pages_to_text(pages: list) -> str:
    """
    Convert list of ParsedContractPage objects to plain text.
    
    Extracts text from the hierarchical structure (sections -> clauses -> subsections).
    """
    all_text = []
    
    for page in pages:
        # Handle both dict and Pydantic model
        if hasattr(page, "model_dump"):
            page_dict = page.model_dump()
        elif hasattr(page, "dict"):
            page_dict = page.dict()
        else:
            page_dict = page
        
        # Use raw_content fallback if available
        if page_dict.get("raw_content"):
            all_text.append(page_dict["raw_content"])
            continue
        
        # Extract sections
        for section in page_dict.get("sections", []):
            # Section header: "I. TERM AND TERMINATION"
            section_header = f"{section.get('id', '')}. {section.get('title', '')}".strip()
            if section_header and section_header != ".":
                all_text.append(section_header)
            
            # Section preamble content if any
            if section.get("content"):
                all_text.append(section["content"])
            
            # Clauses
            for clause in section.get("clauses", []):
                clause_id = clause.get("id", "")
                clause_title = clause.get("title", "") or ""
                clause_content = clause.get("content", "") or ""
                
                if clause_title:
                    clause_text = f"{clause_id} {clause_title} {clause_content}".strip()
                else:
                    clause_text = f"{clause_id} {clause_content}".strip()
                
                if clause_text:
                    all_text.append(clause_text)
            
            # Handle subsections recursively
            for subsection in section.get("subsections", []):
                subsec_header = f"{subsection.get('id', '')}. {subsection.get('title', '')}".strip()
                if subsec_header and subsec_header != ".":
                    all_text.append(subsec_header)
                if subsection.get("content"):
                    all_text.append(subsection["content"])
                for clause in subsection.get("clauses", []):
                    clause_text = f"{clause.get('id', '')} {clause.get('content', '')}".strip()
                    if clause_text:
                        all_text.append(clause_text)
    
    return "\n\n".join(all_text)


class TestImageParserAccuracy:
    """Integration tests for image parser accuracy against ground truth."""
    
    @pytest.fixture(scope="class")
    def event_loop(self):
        """Create event loop for async tests."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pair_1_original_accuracy(self):
        """Test parsing accuracy for pair_1 original contract."""
        folder_path = TEST_CONTRACTS_DIR / "pair_1" / "original"
        
        # Create mock session (no Langfuse tracing for tests)
        session = MockTracingSession(
            session_name="accuracy_test_pair_1_original",
            contract_id="pair_1_original_test"
        )
        
        # Parse images
        print(f"\n📂 Parsing: {folder_path}")
        pages = await parse_contract_folder(str(folder_path), session)
        
        assert pages, "No pages parsed"
        print(f"✅ Parsed {len(pages)} pages")
        
        # Convert to text
        parsed_text = pages_to_text(pages)
        print(f"📝 Parsed text length: {len(parsed_text)} characters")
        
        # Load ground truth
        ground_truth = load_ground_truth("pair_1", "original")
        print(f"📝 Ground truth length: {len(ground_truth)} characters")
        
        # Calculate accuracy
        accuracy = calculate_accuracy(parsed_text, ground_truth)
        
        print("\n📊 Accuracy Metrics (using Levenshtein distance):")
        print(f"   Character Accuracy: {accuracy['character_accuracy']:.1%} (CER: {accuracy['cer']:.2%})")
        print(f"   Word Accuracy:      {accuracy['word_accuracy']:.1%} (WER: {accuracy['wer']:.2%})")
        print(f"   Edit distances:     {accuracy['char_edit_distance']} chars, {accuracy['word_edit_distance']} words")
        
        # Assert ≥95% accuracy (using character accuracy = 1 - CER)
        assert accuracy["character_accuracy"] >= 0.95, \
            f"Character accuracy {accuracy['character_accuracy']:.1%} is below 95% threshold (CER: {accuracy['cer']:.2%})"
        
        print("\n✅ PASSED: pair_1/original achieves ≥95% accuracy")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pair_1_amendment_accuracy(self):
        """Test parsing accuracy for pair_1 amendment contract."""
        folder_path = TEST_CONTRACTS_DIR / "pair_1" / "amendment"
        
        session = MockTracingSession(
            session_name="accuracy_test_pair_1_amendment",
            contract_id="pair_1_amendment_test"
        )
        
        print(f"\n📂 Parsing: {folder_path}")
        pages = await parse_contract_folder(str(folder_path), session)
        
        assert pages, "No pages parsed"
        print(f"✅ Parsed {len(pages)} pages")
        
        parsed_text = pages_to_text(pages)
        ground_truth = load_ground_truth("pair_1", "amendment")
        
        accuracy = calculate_accuracy(parsed_text, ground_truth)
        
        print("\n📊 Accuracy Metrics (using Levenshtein distance):")
        print(f"   Character Accuracy: {accuracy['character_accuracy']:.1%} (CER: {accuracy['cer']:.2%})")
        print(f"   Word Accuracy:      {accuracy['word_accuracy']:.1%} (WER: {accuracy['wer']:.2%})")
        print(f"   Edit distances:     {accuracy['char_edit_distance']} chars, {accuracy['word_edit_distance']} words")
        
        assert accuracy["character_accuracy"] >= 0.95, \
            f"Character accuracy {accuracy['character_accuracy']:.1%} is below 95% threshold (CER: {accuracy['cer']:.2%})"
        
        print("\n✅ PASSED: pair_1/amendment achieves ≥95% accuracy")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pair_2_original_accuracy(self):
        """Test parsing accuracy for pair_2 original contract."""
        folder_path = TEST_CONTRACTS_DIR / "pair_2" / "original"
        
        session = MockTracingSession(
            session_name="accuracy_test_pair_2_original",
            contract_id="pair_2_original_test"
        )
        
        print(f"\n📂 Parsing: {folder_path}")
        pages = await parse_contract_folder(str(folder_path), session)
        
        assert pages, "No pages parsed"
        print(f"✅ Parsed {len(pages)} pages")
        
        parsed_text = pages_to_text(pages)
        ground_truth = load_ground_truth("pair_2", "original")
        
        accuracy = calculate_accuracy(parsed_text, ground_truth)
        
        print("\n📊 Accuracy Metrics (using Levenshtein distance):")
        print(f"   Character Accuracy: {accuracy['character_accuracy']:.1%} (CER: {accuracy['cer']:.2%})")
        print(f"   Word Accuracy:      {accuracy['word_accuracy']:.1%} (WER: {accuracy['wer']:.2%})")
        print(f"   Edit distances:     {accuracy['char_edit_distance']} chars, {accuracy['word_edit_distance']} words")
        
        assert accuracy["character_accuracy"] >= 0.95, \
            f"Character accuracy {accuracy['character_accuracy']:.1%} is below 95% threshold (CER: {accuracy['cer']:.2%})"
        
        print("\n✅ PASSED: pair_2/original achieves ≥95% accuracy")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pair_2_amendment_accuracy(self):
        """Test parsing accuracy for pair_2 amendment contract."""
        folder_path = TEST_CONTRACTS_DIR / "pair_2" / "amendment"
        
        session = MockTracingSession(
            session_name="accuracy_test_pair_2_amendment",
            contract_id="pair_2_amendment_test"
        )
        
        print(f"\n📂 Parsing: {folder_path}")
        pages = await parse_contract_folder(str(folder_path), session)
        
        assert pages, "No pages parsed"
        print(f"✅ Parsed {len(pages)} pages")
        
        parsed_text = pages_to_text(pages)
        ground_truth = load_ground_truth("pair_2", "amendment")
        
        accuracy = calculate_accuracy(parsed_text, ground_truth)
        
        print("\n📊 Accuracy Metrics (using Levenshtein distance):")
        print(f"   Character Accuracy: {accuracy['character_accuracy']:.1%} (CER: {accuracy['cer']:.2%})")
        print(f"   Word Accuracy:      {accuracy['word_accuracy']:.1%} (WER: {accuracy['wer']:.2%})")
        print(f"   Edit distances:     {accuracy['char_edit_distance']} chars, {accuracy['word_edit_distance']} words")
        
        assert accuracy["character_accuracy"] >= 0.95, \
            f"Character accuracy {accuracy['character_accuracy']:.1%} is below 95% threshold (CER: {accuracy['cer']:.2%})"
        
        print("\n✅ PASSED: pair_2/amendment achieves ≥95% accuracy")


class TestAccuracyAllPairs:
    """Run accuracy test on all contract pairs at once."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_all_contracts_accuracy(self):
        """Test accuracy for all 4 contracts (both pairs, original + amendment)."""
        results = []
        all_passed = True
        
        test_cases = [
            ("pair_1", "original"),
            ("pair_1", "amendment"),
            ("pair_2", "original"),
            ("pair_2", "amendment"),
        ]
        
        print("\n" + "=" * 60)
        print("IMAGE PARSER ACCURACY TEST")
        print("Using Levenshtein distance for CER/WER calculation")
        print("=" * 60)
        
        for pair_name, contract_type in test_cases:
            folder_path = TEST_CONTRACTS_DIR / pair_name / contract_type
            
            print(f"\n📂 Testing: {pair_name}/{contract_type}")
            
            try:
                # Create mock session (no Langfuse tracing for tests)
                session = MockTracingSession(
                    session_name=f"accuracy_test_{pair_name}_{contract_type}",
                    contract_id=f"{pair_name}_{contract_type}_test"
                )
                
                # Parse images
                pages = await parse_contract_folder(str(folder_path), session)
                
                if not pages:
                    print(f"   ❌ No pages parsed")
                    results.append((pair_name, contract_type, 0, "No pages"))
                    all_passed = False
                    continue
                
                # Convert and compare
                parsed_text = pages_to_text(pages)
                ground_truth = load_ground_truth(pair_name, contract_type)
                accuracy = calculate_accuracy(parsed_text, ground_truth)
                
                char_acc = accuracy["character_accuracy"]
                status = "✅" if char_acc >= 0.95 else "❌"
                
                print(f"   {status} Character Accuracy: {char_acc:.1%} (CER: {accuracy['cer']:.2%})")
                print(f"      Word Accuracy: {accuracy['word_accuracy']:.1%} (WER: {accuracy['wer']:.2%})")
                
                results.append((pair_name, contract_type, char_acc, "OK"))
                
                # Mark session as successful
                session.mark_success({"accuracy": char_acc})
                
                if char_acc < 0.95:
                    all_passed = False
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append((pair_name, contract_type, 0, str(e)))
                all_passed = False
                # Mark session as error if it was created
                if 'session' in locals():
                    session.mark_error(str(e))
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Contract':<25} {'Accuracy':>10} {'CER':>10} {'Status':>10}")
        print("-" * 55)
        
        for pair_name, contract_type, acc, status in results:
            name = f"{pair_name}/{contract_type}"
            acc_str = f"{acc:.1%}" if acc > 0 else "N/A"
            cer_str = f"{1-acc:.2%}" if acc > 0 else "N/A"
            status_str = "PASS" if acc >= 0.95 else "FAIL"
            print(f"{name:<25} {acc_str:>10} {cer_str:>10} {status_str:>10}")
        
        avg_accuracy = sum(r[2] for r in results) / len(results) if results else 0
        avg_cer = 1 - avg_accuracy
        print("-" * 55)
        print(f"{'Average':<25} {avg_accuracy:.1%} {avg_cer:.2%}")
        print("=" * 60)
        
        assert all_passed, "One or more contracts failed the 95% accuracy threshold"
        print("\n✅ ALL CONTRACTS PASSED ≥95% ACCURACY TEST")
