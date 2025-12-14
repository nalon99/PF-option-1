"""
Langfuse Tracing Module (Simplified)

Provides tracing capabilities for the contract analysis workflow.
Each trace captures: input, output, latency, tokens, cost
Custom metadata: session_id, contract_id, agent names
"""

import os
import uuid
from datetime import datetime, UTC
from typing import Optional, Any, Dict
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root .env file
ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_FILE, override=True)

# Langfuse imports
from langfuse import Langfuse


# =============================================================================
# LANGFUSE CLIENT INITIALIZATION
# =============================================================================

def get_langfuse_client() -> Langfuse:
    """Initialize and return Langfuse client. Raises error if not configured."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    if not public_key or not secret_key:
        raise ValueError(
            "Langfuse configuration REQUIRED. "
            "Please set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in your .env file."
        )
    
    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host
    )
    print(f"✅ Langfuse tracing initialized (host: {host})")
    return client


# Global Langfuse client (initialized on module load)
langfuse_client = get_langfuse_client()


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

class TracingSession:
    """
    Manages a tracing session for a complete contract analysis workflow.
    Uses Langfuse's start_as_current_span for the root span.
    """
    
    def __init__(
        self,
        contract_id: Optional[str] = None,
        session_name: Optional[str] = None,
        input_data: Optional[Dict] = None
    ):
        self.session_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        self.contract_id = contract_id or f"contract_{now}"
        self.session_name = session_name or f"Contract Analysis - {self.contract_id}"
        self.metadata = {
            "session_id": self.session_id,
            "contract_id": self.contract_id,
            "started_at": now
        }
        
        # Create root span - store context manager to keep it open
        self._root_context = langfuse_client.start_as_current_span(
            name="session_root",
            metadata=self.metadata
        )
        # Enter the context to get the span object
        self.root_span = self._root_context.__enter__()
        
        # Update trace with name and metadata
        self.root_span.update_trace(
            name=self.session_name,
            input={"contract_id": self.contract_id},
            metadata=self.metadata
        )
        print(f"📊 Tracing session started: {self.session_id}")
    
    def create_span(
        self,
        name: str,
        input_data: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        """Create a new span within this trace (as child of current context)."""
        span_metadata = {**self.metadata, **(metadata or {})}
        span = langfuse_client.start_span(
            name=name,
            input=input_data,
            metadata=span_metadata
        )
        return SpanWrapper(span)
    
    def create_generation(
        self,
        name: str,
        model: str,
        input_data: Optional[Any] = None,
        metadata: Optional[Dict] = None
    ):
        """Create a new generation (LLM call) - tracks model, tokens, cost."""
        gen_metadata = {**self.metadata, **(metadata or {})}
        generation = langfuse_client.start_generation(
            name=name,
            model=model,
            input=input_data,
            metadata=gen_metadata
        )
        return GenerationWrapper(generation)
    
    def mark_error(self, error_message: str):
        """Mark the trace as ERROR for clear visibility in Langfuse UI."""
        # Update trace output/metadata
        self.root_span.update_trace(
            output={"error": error_message, "success": False},
            metadata={**self.metadata, "status": "error", "ended_at": datetime.now(UTC).isoformat()}
        )
        # Set level on the span (level not supported on update_trace)
        self.root_span.update(level="ERROR")
        print(f"❌ Trace marked as ERROR: {error_message[:50]}...")
    
    def mark_success(self, output: Optional[Dict] = None):
        """Mark the trace as successful."""
        self.root_span.update_trace(
            output=output or {"success": True},
            metadata={**self.metadata, "status": "success", "ended_at": datetime.now(UTC).isoformat()}
        )
    
    def end(self):
        """End the tracing session."""
        # Exit the context manager properly
        if hasattr(self, '_root_context') and self._root_context:
            self._root_context.__exit__(None, None, None)
        langfuse_client.flush()
        print(f"📊 Tracing session ended: {self.session_id}")


class SpanWrapper:
    """Wrapper for Langfuse span with simplified interface."""
    
    def __init__(self, span):
        self.span = span
    
    def update(self, output=None, status_message=None, level=None, **kwargs):
        """Update span with output and metadata.
        
        Args:
            output: Output data
            status_message: Status message
            level: "DEBUG", "DEFAULT", "WARNING", or "ERROR" - visible in Langfuse UI
        """
        update_kwargs = {}
        if output is not None:
            update_kwargs['output'] = output
        if status_message is not None:
            update_kwargs['status_message'] = status_message
        if level is not None:
            update_kwargs['level'] = level
        update_kwargs.update(kwargs)
        self.span.update(**update_kwargs)
    
    def error(self, error_message: str, output=None):
        """Mark span as ERROR for clear visibility in Langfuse."""
        self.update(
            output=output or {"error": error_message},
            status_message=error_message,
            level="ERROR"
        )
    
    def end(self):
        """End the span."""
        self.span.end()


class GenerationWrapper:
    """Wrapper for Langfuse generation (LLM call) with token/cost tracking."""
    
    def __init__(self, generation):
        self.generation = generation
    
    def update(self, output=None, usage=None, level=None, **kwargs):
        """Update generation with output and usage (tokens).
        
        Args:
            output: Output data
            usage: Token usage dict {"input": X, "output": Y, "total": Z}
            level: "DEBUG", "DEFAULT", "WARNING", or "ERROR" - visible in Langfuse UI
        """
        update_kwargs = {}
        if output is not None:
            update_kwargs['output'] = output
        if usage is not None:
            # Langfuse expects usage_details for token tracking
            update_kwargs['usage_details'] = usage
            # usage keys must be as the following example shown below
            # update_kwargs['usage_details'] = {
            #     "input": usage.get("input", usage.get("prompt_tokens", 0)),
            #     "output": usage.get("output", usage.get("completion_tokens", 0)),
            #     "total": usage.get("total", usage.get("total_tokens", 0)),
            #     "unit": "TOKENS"
            # }
        if level is not None:
            update_kwargs['level'] = level
        update_kwargs.update(kwargs)
        self.generation.update(**update_kwargs)
    
    def error(self, error_message: str, output=None):
        """Mark generation as ERROR for clear visibility in Langfuse."""
        self.update(
            output=output or {"error": error_message},
            status_message=error_message,
            level="ERROR"
        )
    
    def end(self):
        """End the generation."""
        self.generation.end()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_session(
    contract_id: Optional[str] = None,
    session_name: Optional[str] = None,
    input_data: Optional[Dict] = None
) -> TracingSession:
    """Create a new tracing session."""
    return TracingSession(contract_id=contract_id, session_name=session_name, input_data=input_data)


def flush_traces():
    """Flush all pending traces to Langfuse."""
    if langfuse_client:
        langfuse_client.flush()
        print("📊 Traces flushed to Langfuse")


if __name__ == "__main__":
    session = create_session(contract_id="test_contract_id", session_name="test_session_name", input_data={"test_input_data": "empty now"})
    
    generation = session.create_generation(name="test_generation", model="test_model", input_data={"test_input_generation": "test_data_generation"})
    generation.update(output="test_output", usage={"test_usage": "test_usage_data"})
    generation.end()

    span = session.create_span(name="test_span", input_data={"test_input_span": "test_data_span"})
    span.update(output="test_output_span", status_message="test_status_message")
    span.end()

    session.end()
    flush_traces()
    
    print(f"Session: {session}")
    print(f"Generation: {generation}")
    print(f"Span: {span}")
    print(f"Langfuse client: {langfuse_client}")
    # print(f"Langfuse client configuration: {langfuse_client.config}")