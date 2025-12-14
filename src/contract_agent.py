"""
Contract Agent - Main Entry Point

Entry point that:
(1) accepts two folder paths as arguments (original and amendment contract images),
(2) calls multimodal LLM to parse images from both folders,
(3) executes Agent 1 (contextualization),
(4) executes Agent 2 (change extraction),
(5) validates output with Pydantic,
(6) returns structured JSON.

Must be runnable from command line.

Workflow:
    Image Parsing → Agent 1 (Contextualization) → Agent 2 (Extraction) → Output

Usage:
    # Direct mode: provide two folder paths
    python contract_agent.py /path/to/original /path/to/amendment
    python contract_agent.py /path/to/original /path/to/amendment --output results.json

    # Batch mode: process from data/test_contracts/
    python contract_agent.py --batch                    # Process all pairs
    python contract_agent.py --batch pair_1             # Process specific pair
    python contract_agent.py --batch pair_1 pair_2      # Process multiple pairs

Contract folder structure (for batch mode):
    data/test_contracts/
    ├── pair_1/
    │   ├── original/     # Original contract images
    │   └── amendment/    # Amendment contract images
    └── pair_2/
        ├── original/
        └── amendment/
"""

import argparse
import asyncio
import os
import sys
import json
from pathlib import Path
from typing import Optional, Tuple, List

import langfuse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import (
    ParsedContractPage,
    ContextualizationOutput,
    ContractAnalysisResult
)
from image_parser import parse_contract_folder
from agents.contextualization_agent import ContextualizationAgent
from agents.extraction_agent import ExtractionAgent
from tracing import TracingSession


# =============================================================================
# CONSTANTS
# =============================================================================

# Base directory for test contracts (relative to src/)
TEST_CONTRACTS_DIR = Path(__file__).parent.parent / "data" / "test_contracts"


# =============================================================================
# CONTRACT PAIR DISCOVERY
# =============================================================================

def discover_contract_pairs() -> List[Tuple[str, Path, Path]]:
    """
    Discover all contract pairs in the test_contracts directory.
    
    Returns:
        List of tuples: (pair_name, original_folder, amendment_folder)
    """
    pairs = []
    
    if not TEST_CONTRACTS_DIR.exists():
        print(f"❌ Test contracts directory not found: {TEST_CONTRACTS_DIR}")
        return pairs
    
    for item in sorted(TEST_CONTRACTS_DIR.iterdir()):
        if item.is_dir():
            original_folder = item / "original"
            amendment_folder = item / "amendment"
            
            if original_folder.exists() and amendment_folder.exists():
                pairs.append((item.name, original_folder, amendment_folder))
            else:
                print(f"⚠️  Skipping {item.name}: missing original/ or amendment/ subfolder")
    
    return pairs


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_folder(folder_path: str) -> Tuple[bool, str]:
    """
    Validate that folder exists and contains image files.
    
    Args:
        folder_path: Path to folder to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(folder_path)
    
    if not path.exists():
        return False, f"Folder not found: {folder_path}"
    
    if not path.is_dir():
        return False, f"Path is not a directory: {folder_path}"
    
    # Check for image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
    image_files = [f for f in path.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        return False, f"No image files found in: {folder_path}"
    
    return True, f"Found {len(image_files)} image files"


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

async def analyze_contract_amendment(
    original_folder: str,
    amendment_folder: str,
    session: TracingSession
) -> ContractAnalysisResult:
    """
    Main workflow: Parse images, run Agent 1, run Agent 2, return results.
    
    Args:
        original_folder: Path to folder with original contract images
        amendment_folder: Path to folder with amendment contract images
        session: Langfuse tracing session
        
    Returns:
        ContractAnalysisResult: Validated structured output
        
    Raises:
        ValueError: If any step fails
    """    
    try:
        # =================================================================
        # STEP 1: Parse Images (parallel)
        # =================================================================
        print("\n" + "="*70)
        print("STEP 1: IMAGE PARSING (parallel)")
        print("="*70)
        
        print(f"\n📂 Parsing both contracts in parallel...")
        original_pages, amendment_pages = await asyncio.gather(
            parse_contract_folder(original_folder, session),
            parse_contract_folder(amendment_folder, session)
        )
        
        if not original_pages:
            raise ValueError("Failed to parse original contract images")
        print(f"✅ Parsed {len(original_pages)} pages from original contract")
        
        if not amendment_pages:
            raise ValueError("Failed to parse amendment contract images")
        print(f"✅ Parsed {len(amendment_pages)} pages from amendment contract")
        
        # =================================================================
        # STEP 2: Agent 1 - Contextualization
        # =================================================================
        print("\n" + "="*70)
        print("STEP 2: AGENT 1 - CONTEXTUALIZATION")
        print("="*70)
        
        agent1 = ContextualizationAgent(session)
        contextualization_output = await agent1.agent_contextualize(
            original_pages=original_pages,
            amended_pages=amendment_pages,
            original_name="Original Contract",
            amended_name="Amended Contract"
        )
        
        if not contextualization_output:
            raise ValueError("Agent 1 (Contextualization) failed to produce output")
        
        # Save contextualization output for debugging/inspection
        ctx_output_file = Path(original_folder).parent / "contextualization_output.json"
        with open(ctx_output_file, 'w') as f:
            json.dump(contextualization_output.model_dump(), f, indent=2)
        
        print(f"\n✅ Agent 1 complete:")
        print(f"   - Aligned sections: {len(contextualization_output.aligned_sections)}")
        print(f"   - Sections with changes: {sum(1 for s in contextualization_output.aligned_sections if s.has_changes)}")
        print(f"   - Saved to: {ctx_output_file}")
        
        # =================================================================
        # STEP 3: Agent 2 - Extraction (HANDOFF)
        # =================================================================
        print("\n" + "="*70)
        print("STEP 3: AGENT 2 - EXTRACTION (HANDOFF FROM AGENT 1)")
        print("="*70)
        
        # Create handoff span to trace the agent-to-agent communication
        handoff_span = session.create_span(
            name="agent_handoff_1_to_2",
            input_data={
                "from_agent": "contextualization_agent",
                "to_agent": "extraction_agent",
                "data_type": "ContextualizationOutput",
                "aligned_sections": len(contextualization_output.aligned_sections)
            },
            metadata={"operation": "handoff"}
        )
        handoff_span.update(output={"handoff": "complete"})
        handoff_span.end()
        
        agent2 = ExtractionAgent(session=session)
        extraction_result = await agent2.agent_extract_changes(contextualization_output)
        
        print(f"\n✅ Agent 2 complete:")
        print(f"   - Sections changed: {len(extraction_result.sections_changed)}")
        print(f"   - Topics touched: {len(extraction_result.topics_touched)}")
        print(f"   - Summary items: {len(extraction_result.summary_of_the_change)}")
        
        # =================================================================
        # STEP 4: Final Validation (already done by Pydantic in Agent 2)
        # =================================================================
        print("\n" + "="*70)
        print("STEP 4: OUTPUT VALIDATION")
        print("="*70)
        
        # Pydantic validation already done, but we can add additional checks
        validation_span = session.create_span(
            name="final_validation",
            input_data={"result_type": "ContractAnalysisResult"},
            metadata={"operation": "validation"}
        )
        
        # Cross-check validations
        validations = {
            "has_sections_changed": len(extraction_result.sections_changed) > 0,
            "has_topics_touched": len(extraction_result.topics_touched) > 0,
            "has_summary": len(extraction_result.summary_of_the_change) > 0,
            "topics_match_summary": len(extraction_result.topics_touched) == len(extraction_result.summary_of_the_change)
        }
        
        all_valid = all(validations.values())
        if all_valid:
            validation_span.update(output={"validations": validations, "all_valid": all_valid})
        else:
            validation_span.error(f"Validation failed: {validations}")
        validation_span.end()
        
        if not all_valid:
            raise ValueError(f"Final validation failed: {validations}")
        
        print("✅ All validations passed")
        
        # Mark trace as SUCCESS
        session.mark_success({
            "success": True,
            "sections_changed": len(extraction_result.sections_changed),
            "topics_touched": len(extraction_result.topics_touched)
        })
        session.end()
        
        return extraction_result
        
    except Exception as e:
        # Mark trace as ERROR (clearly visible in Langfuse UI)
        session.mark_error(str(e))
        session.end()
        raise


# =============================================================================
# PROCESS SINGLE CONTRACT PAIR
# =============================================================================

async def process_contract_pair(
    pair_name: str,
    original_folder: Path,
    amendment_folder: Path
) -> Optional[ContractAnalysisResult]:
    """
    Process a single contract pair and save results.
    
    Args:
        pair_name: Name of the contract pair (e.g., "pair_1")
        original_folder: Path to original contract images
        amendment_folder: Path to amendment contract images
        
    Returns:
        ContractAnalysisResult or None on error
    """
    print("\n" + "="*70)
    print(f"CONTRACT AMENDMENT ANALYSIS: {pair_name}")
    print("="*70)
    
    session = TracingSession(
        contract_id=pair_name,
        session_name=f"Contract Analysis - {pair_name}"
    )
   
    try:
        # Run analysis (async)
        result = await analyze_contract_amendment(
            original_folder=str(original_folder),
            amendment_folder=str(amendment_folder),
            session=session
        )
        
        # Output results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        
        result_json = result.model_dump()
        
        # Save to file in the pair folder
        output_file = original_folder.parent / "extraction_output.json"
        with open(output_file, 'w') as f:
            json.dump(result_json, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Print summary
        print("\n" + "-"*70)
        print("📊 SUMMARY")
        print("-"*70)
        print(f"\nSections Changed ({len(result.sections_changed)}):")
        for sec in result.sections_changed:
            print(f"  • {sec}")
        
        print(f"\nTopics Touched ({len(result.topics_touched)}):")
        for topic in result.topics_touched:
            print(f"  • {topic}")
        
        print(f"\nChange Details ({len(result.summary_of_the_change)}):")
        for i, summary in enumerate(result.summary_of_the_change, 1):
            print(f"  {i}. {summary}")
        
        print("\n" + "="*70)
        print(f"✅ {pair_name} ANALYSIS COMPLETE")
        print("="*70 + "\n")
        
        # End session successfully
        session.root_span.update(
            output={
                "status": "success",
                "sections_changed": len(result.sections_changed),
                "topics_touched": len(result.topics_touched)
            },
            status_message="success"
        )
        
        return result
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # End session with error
        session.root_span.update(
            output={"status": "error", "error": str(e)},
            status_message="error"
        )
        return None
    
    finally:
        session.end()

async def process_all_pairs(pairs_to_process: List[Tuple[str, Path, Path]]):
    """Process all contract pairs concurrently using native async."""
    async def process_pair_async(pair_name, original_folder, amendment_folder):
        """Async wrapper that returns pair_name with result."""
        result = await process_contract_pair(pair_name, original_folder, amendment_folder)
        return pair_name, result
    
    # Create tasks for all pairs
    tasks = [
        process_pair_async(pair_name, original_folder, amendment_folder)
        for pair_name, original_folder, amendment_folder in pairs_to_process
    ]
    
    # Run all tasks concurrently and gather results
    return await asyncio.gather(*tasks, return_exceptions=True)
    

# =============================================================================
# DIRECT MODE: Process two folder paths
# =============================================================================

async def process_direct_mode(
    original_folder: str,
    amendment_folder: str,
    output_file: Optional[str] = None
) -> Optional[ContractAnalysisResult]:
    """
    Process a single contract pair from direct folder paths.
    
    Args:
        original_folder: Path to folder with original contract images
        amendment_folder: Path to folder with amendment contract images
        output_file: Optional output file path (default: stdout + JSON in parent dir)
        
    Returns:
        ContractAnalysisResult or None on error
    """
    # Generate a pair name from folder paths
    orig_path = Path(original_folder).resolve()
    amend_path = Path(amendment_folder).resolve()
    pair_name = f"{orig_path.parent.name}" if orig_path.parent == amend_path.parent else "direct_analysis"
    
    print("\n" + "="*70)
    print(f"CONTRACT AMENDMENT ANALYSIS")
    print("="*70)
    print(f"\n📂 Original:  {original_folder}")
    print(f"📂 Amendment: {amendment_folder}")
    
    # Validate folders
    print("\n🔍 Validating input folders...")
    valid, msg = validate_folder(original_folder)
    if not valid:
        print(f"❌ Original folder: {msg}")
        sys.exit(1)
    print(f"✅ Original: {msg}")
    
    valid, msg = validate_folder(amendment_folder)
    if not valid:
        print(f"❌ Amendment folder: {msg}")
        sys.exit(1)
    print(f"✅ Amendment: {msg}")
    
    # Create tracing session
    session = TracingSession(
        contract_id=pair_name,
        session_name=f"Contract Analysis - {pair_name}"
    )
    
    try:
        # Run analysis
        result = await analyze_contract_amendment(
            original_folder=original_folder,
            amendment_folder=amendment_folder,
            session=session
        )
        
        # Output results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        
        result_json = result.model_dump()
        
        # Determine output file path
        if output_file:
            output_path = Path(output_file)
        else:
            # Default: save in parent of original folder
            output_path = orig_path.parent / "extraction_output.json"
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(result_json, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Print summary
        print("\n" + "-"*70)
        print("📊 SUMMARY")
        print("-"*70)
        print(f"\nSections Changed ({len(result.sections_changed)}):")
        for sec in result.sections_changed:
            print(f"  • {sec}")
        
        print(f"\nTopics Touched ({len(result.topics_touched)}):")
        for topic in result.topics_touched:
            print(f"  • {topic}")
        
        print(f"\nChange Details ({len(result.summary_of_the_change)}):")
        for i, summary in enumerate(result.summary_of_the_change, 1):
            print(f"  {i}. {summary}")
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70 + "\n")
        
        # Print JSON to stdout as well
        print("\n📄 JSON Output:")
        print(json.dumps(result_json, indent=2))
        
        return result
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        session.end()


# =============================================================================
# BATCH MODE: Process from test_contracts directory
# =============================================================================

def run_batch_mode(pair_names: Optional[List[str]] = None):
    """
    Process contract pairs from data/test_contracts/ directory.
    
    Args:
        pair_names: Optional list of specific pair names to process
    """
    print("\n" + "="*70)
    print("CONTRACT AMENDMENT ANALYZER (BATCH MODE)")
    print("="*70)
    print(f"\n📂 Scanning for contracts in: {TEST_CONTRACTS_DIR}")
    
    # Discover all contract pairs
    all_pairs = discover_contract_pairs()
    
    if not all_pairs:
        print("❌ No contract pairs found!")
        sys.exit(1)
    
    print(f"\n📋 Found {len(all_pairs)} contract pair(s):")
    for name, orig, amend in all_pairs:
        print(f"   • {name}")
    
    # Determine which pairs to process
    if pair_names:
        pairs_to_process = [
            (name, orig, amend) 
            for name, orig, amend in all_pairs 
            if name in pair_names
        ]
        
        not_found = set(pair_names) - {name for name, _, _ in pairs_to_process}
        if not_found:
            print(f"\n⚠️  Pairs not found: {', '.join(not_found)}")
    else:
        pairs_to_process = all_pairs
    
    if not pairs_to_process:
        print("❌ No valid pairs to process!")
        sys.exit(1)
    
    # Validate ALL folders upfront (fail-fast approach)
    print("\n🔍 Validating all input folders...")
    all_valid = True
    for pair_name, original_folder, amendment_folder in pairs_to_process:
        valid, msg = validate_folder(str(original_folder))
        if not valid:
            print(f"❌ {pair_name}/original: {msg}")
            all_valid = False
        else:
            print(f"✅ {pair_name}/original: {msg}")
        
        valid, msg = validate_folder(str(amendment_folder))
        if not valid:
            print(f"❌ {pair_name}/amendment: {msg}")
            all_valid = False
        else:
            print(f"✅ {pair_name}/amendment: {msg}")
    
    if not all_valid:
        print("\n❌ Validation failed. Fix issues before processing.")
        sys.exit(1)
    
    # Process all pairs concurrently using asyncio
    print(f"\n🔄 Processing {len(pairs_to_process)} pair(s) concurrently...")
    pair_results = asyncio.run(process_all_pairs(pairs_to_process))
    
    # Collect successful results
    results = {}
    for item in pair_results:
        if isinstance(item, Exception):
            print(f"❌ Task failed with exception: {item}")
        elif item is not None:
            pair_name, result = item
            if result:
                results[pair_name] = result
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\n✅ Successfully processed: {len(results)}/{len(pairs_to_process)} pairs")
    
    for pair_name, result in results.items():
        print(f"\n📄 {pair_name}:")
        print(f"   Sections changed: {len(result.sections_changed)}")
        print(f"   Topics touched: {len(result.topics_touched)}")
    
    if len(results) < len(pairs_to_process):
        failed = len(pairs_to_process) - len(results)
        print(f"\n❌ Failed: {failed} pair(s)")
        sys.exit(1)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for command line usage.
    
    Supports two modes:
    1. Direct mode: python contract_agent.py <original_folder> <amendment_folder>
    2. Batch mode:  python contract_agent.py --batch [pair_names...]
    """
    parser = argparse.ArgumentParser(
        description="Analyze contract amendments using multimodal LLM and two-agent system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct mode - process two folders
  python contract_agent.py ./original ./amendment
  python contract_agent.py /path/to/original /path/to/amendment --output results.json

  # Batch mode - process from data/test_contracts/
  python contract_agent.py --batch
  python contract_agent.py --batch pair_1
  python contract_agent.py --batch pair_1 pair_2
        """
    )
    
    # Direct mode arguments (positional)
    parser.add_argument(
        "original_folder",
        nargs="?",
        help="Path to folder containing original contract images"
    )
    parser.add_argument(
        "amendment_folder",
        nargs="?",
        help="Path to folder containing amendment contract images"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path for JSON results (default: extraction_output.json in parent folder)"
    )
    
    # Batch mode arguments
    parser.add_argument(
        "--batch", "-b",
        nargs="*",
        metavar="PAIR",
        help="Run in batch mode. Optionally specify pair names (e.g., pair_1 pair_2)"
    )
    
    args = parser.parse_args()
    
    # Determine which mode to run
    if args.batch is not None:
        # Batch mode
        pair_names = args.batch if args.batch else None
        run_batch_mode(pair_names)
    elif args.original_folder and args.amendment_folder:
        # Direct mode with two folder paths
        asyncio.run(process_direct_mode(
            original_folder=args.original_folder,
            amendment_folder=args.amendment_folder,
            output_file=args.output
        ))
    else:
        # No arguments - show help
        parser.print_help()
        print("\n" + "="*70)
        print("ERROR: Please provide either:")
        print("  1. Two folder paths: python contract_agent.py <original> <amendment>")
        print("  2. Batch mode flag:  python contract_agent.py --batch [pair_names...]")
        print("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()