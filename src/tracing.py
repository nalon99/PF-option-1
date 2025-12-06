"""
Langfuse Tracing Module

Provides tracing capabilities for the contract analysis workflow:
- Image parsing traces
- Agent 1 (Contextualization) execution traces
- Agent 2 (Extraction) execution traces
- Validation step traces

Each trace captures: input, output, latency, tokens, cost
Custom metadata: session_id, contract_id, agent names
"""

import os
import uuid
from datetime import datetime
from typing import Optional, Any, Dict
from functools import wraps
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

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
    
    Uses Langfuse's context-based approach with start_as_current_span.
    """
    
    def __init__(
        self,
        contract_id: Optional[str] = None,
        session_name: Optional[str] = None
    ):
        self.session_id = str(uuid.uuid4())
        self.contract_id = contract_id or f"contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_name = session_name or f"Contract Analysis - {self.contract_id}"
        self.metadata = {
            "session_id": self.session_id,
            "contract_id": self.contract_id,
            "started_at": datetime.now().isoformat()
        }
        
        # Create the root span for the session
        self.root_span = langfuse_client.start_as_current_span(
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
        """Create a new generation (LLM call) within this trace."""
        gen_metadata = {**self.metadata, **(metadata or {})}
        generation = langfuse_client.start_generation(
            name=name,
            model=model,
            input=input_data,
            metadata=gen_metadata
        )
        return GenerationWrapper(generation)
    
    def end(self, output: Optional[Any] = None, status: str = "success"):
        """End the tracing session."""
        self.root_span.update(
            output=output,
            metadata={
                **self.metadata,
                "ended_at": datetime.now().isoformat(),
                "status": status
            }
        )
        self.root_span.end()
        # Flush to ensure all traces are sent
        langfuse_client.flush()
        print(f"📊 Tracing session ended: {self.session_id}")


class SpanWrapper:
    """Wrapper for Langfuse span with simplified interface."""
    
    def __init__(self, span):
        self.span = span
    
    def update(self, output=None, status_message=None, **kwargs):
        """Update span with output and metadata."""
        update_kwargs = {}
        if output is not None:
            update_kwargs['output'] = output
        if status_message is not None:
            update_kwargs['status_message'] = status_message
        update_kwargs.update(kwargs)
        self.span.update(**update_kwargs)
    
    def end(self):
        """End the span."""
        self.span.end()


class GenerationWrapper:
    """Wrapper for Langfuse generation with simplified interface."""
    
    def __init__(self, generation):
        self.generation = generation
    
    def update(self, output=None, usage=None, **kwargs):
        """Update generation with output and usage info."""
        update_kwargs = {}
        if output is not None:
            update_kwargs['output'] = output
        if usage is not None:
            # Convert usage dict to usage_details format
            update_kwargs['usage_details'] = usage
        update_kwargs.update(kwargs)
        self.generation.update(**update_kwargs)
    
    def end(self):
        """End the generation."""
        self.generation.end()


# =============================================================================
# TRACING CONTEXT MANAGER
# =============================================================================

def trace_llm_call(
    session: TracingSession,
    name: str,
    model: str,
    agent_name: str = "unknown"
):
    """
    Context manager for tracing LLM calls with token/cost tracking.
    
    Usage:
        with trace_llm_call(session, "parse_page", "gpt-4o", "image_parser") as gen:
            response = client.chat.completions.create(...)
            gen.update(output=response.choices[0].message.content, 
                      usage={"input": response.usage.prompt_tokens, 
                             "output": response.usage.completion_tokens})
    """
    class LLMTraceContext:
        def __init__(self):
            self.generation = session.create_generation(
                name=name,
                model=model,
                metadata={"agent": agent_name}
            )
            self.start_time = datetime.now()
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            latency_ms = (datetime.now() - self.start_time).total_seconds() * 1000
            self.generation.update(
                metadata={"latency_ms": latency_ms}
            )
            self.generation.end()
        
        def update(self, output=None, usage=None, **kwargs):
            update_data = {}
            if output:
                update_data['output'] = output
            if usage:
                update_data['usage'] = usage
            update_data.update(kwargs)
            self.generation.update(**update_data)
    
    return LLMTraceContext()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_session(
    contract_id: Optional[str] = None,
    session_name: Optional[str] = None
) -> TracingSession:
    """Create a new tracing session."""
    return TracingSession(contract_id=contract_id, session_name=session_name)


def flush_traces():
    """Flush all pending traces to Langfuse."""
    langfuse_client.flush()
    print("📊 Traces flushed to Langfuse")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test tracing setup
    print("\n" + "="*60)
    print("Testing Langfuse Tracing")
    print("="*60)
    
    session = create_session(contract_id="test_001")
    
    # Simulate a span
    span = session.create_span(
        name="test_operation",
        input_data={"test": True},
        metadata={"purpose": "testing"}
    )
    span.update(output={"result": "success"})
    span.end()
    
    session.end(output={"test": "completed"}, status="success")
    flush_traces()
    
    print("\n✅ Tracing test complete")
