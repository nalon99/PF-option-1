"""
Extraction Agent (Agent 2)

In a multi-agent system where there are two distinct agents with separate prompts 
and responsibilities, this is Agent 2:
- Receives Agent 1's output (ContextualizationOutput)
- Extracts specific changes between original and amended documents
- Returns structured ContractAnalysisResult with sections_changed, topics_touched, summary_of_the_change

Shows clear handoff mechanism: Agent 1 output → Agent 2 input
Includes Langfuse tracing for monitoring all LLM calls.
Uses AsyncOpenAI for async LLM calls.
"""

import asyncio
import os
import json
from typing import Optional
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import ValidationError

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    ContextualizationOutput,
    ContractAnalysisResult
)
from tracing import SpanWrapper, TracingSession

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
# EXTRACTION PROMPT
# =============================================================================

EXTRACTION_PROMPT = """You are a legal document analyst specializing in contract amendments. Your task is to analyze the aligned sections between an original contract and its amendment, then extract and summarize all changes.

You will receive ONLY the sections that contain changes (pre-filtered). Analyze the differences between original_content and amended_content for EACH section provided.

Your analysis should produce:

1. **sections_changed**: List the section identifiers and titles that were modified.
   Format: "SECTION_ID. SECTION_TITLE"
   Example: "II. TERM, TERMINATION, AND SUSPENSION"

2. **topics_touched**: Identify the business/legal topics affected by the changes.
   These should be high-level categories, not section names.
   Examples: "Contract Duration", "Payment Terms", "Liability Cap", "Termination Conditions"
   
3. **summary_of_the_change**: Provide detailed descriptions of each specific change.
   Each item should:
   - Clearly state what changed from original to new value
   - Use **bold** markdown for new values
   - Include section reference in parentheses
   - Be a complete, standalone sentence
   Example: "The initial contract term was extended from twenty-four (24) months to **thirty-six (36) months** (Sec. 2.1)."

IMPORTANT RULES:
- The number of items in "topics_touched" MUST equal the number of items in "summary_of_the_change"
- Each summary item describes one specific change that corresponds to one topic
- Only report actual substantive changes, not formatting or minor wording differences
- Ignore changes where the pipe "|" character appears - this represents OCR artifacts, not real changes
- Summary items must be at least 20 characters long
- Remove any markdown formatting from the summary items

Return a JSON object with this exact structure:
{
    "sections_changed": ["II. TERM, TERMINATION, AND SUSPENSION", "III. COMPENSATION"],
    "topics_touched": ["Contract Duration", "Payment Terms"],
    "summary_of_the_change": [
        "The initial contract term was extended from twenty-four (24) months to **thirty-six (36) months** (Sec. 2.1).",
        "The fixed monthly compensation rate was increased from $15,000 USD to **$18,000 USD** (Sec. 3.1)."
    ]
}
"""


# =============================================================================
# EXTRACTION AGENT CLASS
# =============================================================================

class ExtractionAgent:
    """
    Agent 2: Receives contextualization output and extracts specific changes.
    
    Responsibilities:
    - Analyze aligned sections from Agent 1
    - Identify which sections have changes
    - Extract topics touched by changes
    - Generate detailed summaries of each change
    """
    
    def __init__(self, session: Optional[TracingSession] = None):
        """Initialize the extraction agent with optional tracing session."""
        self.session = session
    
    async def agent_extract_changes(
        self,
        contextualization_output: ContextualizationOutput
    ) -> ContractAnalysisResult:
        """
        Extract specific changes from aligned sections.
        
        Args:
            contextualization_output: Output from Agent 1 with aligned sections
            
        Returns:
            ContractAnalysisResult: Validated structured output with changes
            
        Raises:
            ValueError: If extraction or validation fails
        """
        if not self.session:
            raise ValueError("Tracing session is required for ExtractionAgent")
        
        # Create generation for LLM call tracing (tracks model, tokens, cost)
        span = self.session.create_generation(
            name="agent_extract_changes",
            model=LANGFUSE_MODEL_NAME,
            input_data={
                "original_title": contextualization_output.original_title,
                "amended_title": contextualization_output.amended_title,
                "sections_count": len(contextualization_output.aligned_sections),
                "sections_with_changes": sum(1 for s in contextualization_output.aligned_sections if s.has_changes)
            },
            metadata={"agent": "extraction_agent"}
        )
        
        try:
            # Filter sections that have changes for focused analysis
            sections_with_changes = [
                s for s in contextualization_output.aligned_sections 
                if s.has_changes
            ]
            
            if not sections_with_changes:
                # No changes detected - return minimal result
                span.update(output={"status": "no_changes_detected"})
                span.end()
                raise ValueError("No changes detected between documents")
            
            # Prepare input for LLM - ONLY send changed sections (optimization)
            changed_sections = [
                {
                    "section_id": s.section_id,
                    "section_title": s.section_title,
                    "original_content": s.original_content,
                    "amended_content": s.amended_content
                }
                for s in contextualization_output.aligned_sections
                if s.has_changes  # Filter to only changed sections
            ]
            
            if not changed_sections:
                raise ValueError("No changes detected between documents")
            
            input_data = {
                "original_document": contextualization_output.original_title,
                "amended_document": contextualization_output.amended_title,
                "changed_sections_count": len(changed_sections),
                "aligned_sections": changed_sections
            }
            
            # Call LLM for extraction (async)
            result = await self._call_extraction_llm(input_data, span)
            
            span.update(output={
                "sections_changed": result.sections_changed,
                "topics_count": len(result.topics_touched),
                "summary_count": len(result.summary_of_the_change)
            })
            span.end()
            
            return result
            
        except Exception as e:
            span.error(str(e))  # Mark as ERROR in Langfuse
            span.end()
            raise
    
    async def _call_extraction_llm(self, input_data: dict, span: SpanWrapper) -> ContractAnalysisResult:
        """
        Call LLM to extract changes from aligned sections.
        
        Args:
            input_data: Dictionary with aligned sections
            
        Returns:
            ContractAnalysisResult: Validated extraction result
        """
        input_json = json.dumps(input_data, indent=2)
        
        try:
            # Async LLM call
            completion = await client.chat.completions.create(
                model=AI_MODEL,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": f"Analyze these aligned contract sections and extract all changes:\n\n{input_json}"}
                ],
                temperature=0.0,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            response_content = completion.choices[0].message.content
            
            # Update trace with usage
            span.update(
                output=response_content,
                usage={
                    "input": completion.usage.prompt_tokens,
                    "output": completion.usage.completion_tokens,
                    "total": completion.usage.total_tokens
                }
            )
            
        except Exception as e:
            span.error(f"LLM call failed: {str(e)}")
            raise ValueError(f"LLM call failed: {str(e)}")
        
        # Parse and validate response
        return self._parse_and_validate(response_content)
    
    def _parse_and_validate(self, response_content: str) -> ContractAnalysisResult:
        """
        Parse LLM response and validate with Pydantic.
        
        Args:
            response_content: Raw JSON string from LLM
            
        Returns:
            ContractAnalysisResult: Validated result
            
        Raises:
            ValueError: If parsing or validation fails
        """
        # Create validation span
        span = self.session.create_span(
            name="validate_extraction_output",
            input_data={"response_length": len(response_content)},
            metadata={"operation": "validation"}
        )
        
        try:
            # Parse JSON
            try:
                parsed = json.loads(response_content)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")
            
            # Validate with Pydantic
            try:
                result = ContractAnalysisResult(**parsed)
            except ValidationError as e:
                raise ValueError(f"Pydantic validation failed: {str(e)}")
            
            span.update(
                output={
                    "valid": True,
                    "sections_changed": len(result.sections_changed),
                    "topics_touched": len(result.topics_touched),
                    "summary_items": len(result.summary_of_the_change)
                },
                status_message="valid"
            )
            span.end()
            
            return result
            
        except Exception as e:
            span.error(str(e), output={"error": str(e), "valid": False})
            span.end()
            raise


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def extract_contract_changes(
    contextualization_output: ContextualizationOutput,
    session: TracingSession
) -> ContractAnalysisResult:
    """
    Convenience function to extract changes from contextualization output.
    
    Args:
        contextualization_output: Output from Agent 1
        session: Langfuse tracing session
        
    Returns:
        ContractAnalysisResult: Structured extraction result
    """
    agent = ExtractionAgent(session=session)
    return await agent.agent_extract_changes(contextualization_output)


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from tracing import create_session
    
    async def main():
        parser = argparse.ArgumentParser(description="Test Extraction Agent with contextualization output")
        parser.add_argument(
            "input_file",
            type=str,
            help="Path to contextualization_output.json file"
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output file path (default: same directory as input)"
        )
        
        args = parser.parse_args()
        
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"❌ Input file not found: {input_path}")
            sys.exit(1)
        
        # Load contextualization output
        print(f"\n📄 Loading contextualization output from: {input_path}")
        with open(input_path, 'r') as f:
            ctx_data = json.load(f)
        
        ctx_output = ContextualizationOutput(**ctx_data)
        print(f"✅ Loaded {len(ctx_output.aligned_sections)} aligned sections")
        print(f"   Sections with changes: {sum(1 for s in ctx_output.aligned_sections if s.has_changes)}")
        
        # Create tracing session
        session = create_session(contract_id="extraction_test")
        
        # Run extraction
        print("\n🔍 Running Extraction Agent...")
        try:
            result = await extract_contract_changes(ctx_output, session)
            
            print("\n✅ Extraction complete!")
            print(f"\n📊 Results:")
            print(f"   Sections changed: {len(result.sections_changed)}")
            for sec in result.sections_changed:
                print(f"      - {sec}")
            
            print(f"\n   Topics touched: {len(result.topics_touched)}")
            for topic in result.topics_touched:
                print(f"      - {topic}")
            
            print(f"\n   Summary of changes:")
            for i, summary in enumerate(result.summary_of_the_change, 1):
                print(f"      {i}. {summary}")
            
            # Save output
            output_path = args.output or input_path.parent / "extraction_output.json"
            with open(output_path, 'w') as f:
                json.dump(result.model_dump(), f, indent=2)
            print(f"\n💾 Output saved to: {output_path}")
            
        except Exception as e:
            print(f"\n❌ Extraction failed: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            session.end()
    
    asyncio.run(main())
