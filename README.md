# Legal contracts - Analysis Project

## 1. Project description

This project aims to analyze legal contracts imported from scanned images, to identify differences. The underlying AI system leverages multi-agents execution to accomplish this task effectively.  
> **Note:** All contract images used in tests for this project are synthetically generated and contain purely fictional ("fantasy") content. No real legal documents, personal data, or confidential materials are used at any stage of model development or evaluation. This ensures full compliance with privacy standards and allows safe public sharing of example data and outputs.


1. Implement multimodal document parsing using GPT-4o, Gemini Vision, or Claude Vision to convert scanned contract images into structured text preserving document hierarchy. This eliminates manual OCR preprocessing and handles real-world document quality variations. The system uses a Multimodal LLM to parse both images into text.

2. Build a two-agent collaborative system where:  
    - Agent 1 contextualizes both documents (identifying structure and corresponding sections) 
    - Agent 2 extracts specific changes using Agent 1's analysis to isolate all specific changes introduced by the amendment

    This mimics how legal analysts work: first understanding context, then identifying changes.  

3. Return Pydantic-validated structured output with "sections_changed", "topics_touched", and a precise "summary_of_the_change" fields. This creates a stable contract for downstream systems (legal databases, review queues, compliance dashboards) and prevents malformed data from breaking integrations.  

4. All agent calling actions must be traced using a tracing tool: Langfuse. Instrument complete workflow to capture every step:  
    - image parsing
    - agent execution
    - handoffs
    - validation  

    This enables debugging misclassifications, performance bottlenecks, and cost analysis for production deployment.  

5. Document technical decisions explaining why you chose specific multimodal models, agent collaboration patterns, and prompt engineering strategies. This demonstrates engineering judgment beyond "making it work" and prepares the system for team handoff.  

## 2. Agents workflow and collaboration pattern

```txt
┌─────────────────────┐     ┌─────────────────────┐
│   Original Images   │     │   Amendment Images  │
│   (5 pages .png)    │     │   (5 pages .png)    │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           ▼                           ▼
┌──────────────────────────────────────────────────┐
│              image_parser.py                     │
│         parse_contract_folder() [async]          │
│    (Parallel page parsing with asyncio.gather)   │
└──────────┬───────────────────────────┬───────────┘
           │                           │
           │ List[ParsedContractPage]  │ List[ParsedContractPage]
           ▼                           ▼
┌──────────────────────────────────────────────────┐
│       ContextualizationAgent (Agent 1)           │
├──────────────────────────────────────────────────┤
│  SINGLE LLM CALL: Assembly + Alignment           │
│  - Merges pages from BOTH contracts              │
│  - Aligns corresponding sections                 │
│  - Detects meaningful changes (values, dates)    │
└──────────────────────┬───────────────────────────┘
                       │ ContextualizationOutput
                       │ (aligned_sections with has_changes flags)
                       ▼
┌──────────────────────────────────────────────────┐
│          ExtractionAgent (Agent 2)               │
├──────────────────────────────────────────────────┤
│  LLM CALL with RETRY on validation failure       │
│  - Receives only sections with changes           │
│  - Groups related changes by topic               │
│  - Returns structured output with validation     │
└──────────────────────┬───────────────────────────┘
                       │ ContractAnalysisResult
                       ▼
┌──────────────────────────────────────────────────┐
│              extraction_output.json              │
│  {sections_changed, topics_touched,              │
│   summary_of_the_change}                         │
└──────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Single LLM Call for Agent 1**: Instead of separate assembly + alignment calls, Agent 1 uses ONE LLM call that receives all parsed pages from both contracts. The LLM handles page merging and section alignment in a single pass. This reduces latency (1 round-trip vs 3) and gives the LLM full context to make better alignment decisions.

2. **Grouped Output Format**: Agent 2 groups related changes by topic rather than listing each individual change separately. For example, all term/termination changes appear under one topic with a combined summary referencing multiple sections (Sec. 2.1, 2.2, 2.3). This produces cleaner, more readable output.

3. **Retry with Correction Prompt**: If Agent 2's LLM output fails Pydantic validation (e.g., mismatched counts), the agent automatically retries with a correction prompt explaining the error. This self-healing mechanism reduces manual intervention.

```
## 3. Setup instructions

## 4. Usage

with example commands

## 5. Expected output

Describe expected output with format sample

## 6. Technical decisions

why two agents? why this model? use at least 100 words

## 7. Langfuse tracing guide

how to view dashboard, at least 50 words

## 8. Performance Optimizations

Several optimizations were implemented to reduce latency and cost of the multi-agent pipeline.

### 8.1 AsyncOpenAI for Native Async

**Problem**: The synchronous OpenAI client blocks threads during I/O waits, limiting concurrency.

**Solution**: Use `AsyncOpenAI` client with native async/await throughout the codebase:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# All LLM calls are now non-blocking
completion = await client.chat.completions.create(...)
```

**Result**: Enables true async concurrency without thread overhead. Multiple LLM calls can run simultaneously in a single thread.

### 8.2 Single-Call Contextualization

**Problem**: The original `ContextualizationAgent` used 3 separate LLM calls: assemble original, assemble amended, then align. This meant 3 round-trips to the API.

**Solution**: Combine into a SINGLE LLM call that receives all parsed pages from both contracts:

```python
# Single LLM call handles assembly + alignment
prompt = CONTEXTUALIZATION_PROMPT.format(
    original_pages=self._pages_to_compact_json(original_pages),
    amended_pages=self._pages_to_compact_json(amended_pages)
)

completion = await client.chat.completions.create(
    model=AI_MODEL,
    messages=[{"role": "system", "content": prompt}],
    response_format={"type": "json_object"},
    temperature=0.0
)
```

**Result**: 
- Reduced from 3 LLM calls to 1 (66% fewer API calls)
- LLM has full context of both documents simultaneously
- Better alignment decisions since LLM sees everything at once
- Contextualization step reduced from ~24s to ~8-10s

### 8.3 Parallel Image Parsing

**Problem**: Contract pages were parsed sequentially (5 pages × ~3s = ~15s per folder).

**Solution**: Parse all pages in a folder concurrently using `asyncio.gather()`:

```python
# Parse all pages concurrently
tasks = [
    parse_contract_image(str(image_file), page_number=i, session=session)
    for i, image_file in enumerate(image_files, start=1)
]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

**Result**: Folder parsing reduced from ~15s to ~3s (time of slowest page).

### 8.4 Compact LLM Prompts

**Problem**: Verbose prompts with pretty-printed JSON examples consume more tokens and slow down LLM responses.

**Solution**: Condensed prompts from ~50 lines to ~10 lines using:
- One-line task descriptions
- Compact inline JSON examples (no indentation)
- Essential rules only

**Before (ASSEMBLY_PROMPT)**:
```
47 lines, ~1100 characters, ~280 tokens
```

**After**:
```python
ASSEMBLY_PROMPT = """Merge parsed contract pages into ONE coherent document.

TASK: Detect page-spanning sections, merge split content, extract metadata.

OUTPUT (JSON):
{"title":"...","sections":[{"id":"I","title":"...","content":"...","clauses":[...]}]}

RULES:
- Merge same section across pages
- Preserve EXACT wording - no paraphrasing
- Keep "|" characters as-is (OCR artifacts)
"""
```

**Result**: ~60% fewer prompt tokens, ~2-3s faster per LLM call.

### 8.5 Optimized Output Format

**Problem**: The alignment step asked the LLM to output full content for ALL sections, even unchanged ones.

**Solution**: Instruct LLM to output empty strings for unchanged sections:

```python
# UNCHANGED sections
{"section_id":"III","section_title":"...","original_content":"","amended_content":"","has_changes":false}

# CHANGED sections only include relevant excerpt
{"section_id":"II","section_title":"Term","original_content":"<excerpt>","amended_content":"<excerpt>","has_changes":true}
```

**Result**: ~70% fewer output tokens, alignment reduced from ~12s to ~4-6s.

### 8.6 Compact Input JSON

**Problem**: `json.dumps(..., indent=2)` adds whitespace, increasing input token count.

**Solution**: Use minimal JSON serialization:

```python
# Before
json.dumps(data, indent=2)  # Pretty-printed, ~2000 tokens

# After
json.dumps(data, separators=(',',':'))  # Compact, ~1200 tokens
```

**Result**: ~25% fewer input tokens per LLM call.

### Summary of Performance Gains

| Step | Before | After | Improvement |
|------|--------|-------|-------------|
| Image Parsing (5 pages) | ~15s (sequential) | ~3s (parallel async) | **80%** |
| Contextualization (Agent 1) | ~24s (3 LLM calls) | ~8-10s (1 LLM call) | **60%** |
| Extraction Input | All sections | Changed only | **~70% fewer tokens** |
| LLM Validation Failures | Manual re-run | Auto-retry with correction | **Self-healing** |
| **Total Pipeline (per pair)** | ~43s+ | ~15-20s | **~60%** |

### 8.7 Extraction Input Filtering

**Problem**: Agent 2 (Extraction) received ALL aligned sections, including unchanged ones with empty content.

**Solution**: Filter to only send sections with actual changes:

```python
# Before: Sent ALL sections (8 sections, many empty)
aligned_sections: [s for s in all_sections]

# After: Only send changed sections (3 sections with content)
changed_sections = [s for s in all_sections if s.has_changes]
```

**Result**: ~70% fewer input tokens for extraction, clearer context for LLM.

### 8.8 Error Level Tracking in Langfuse

**Problem**: Errors were not clearly visible in Langfuse UI - required reading output to identify failures.

**Solution**: Added `level="ERROR"` support for clear visibility:

```python
# TracingSession methods
session.mark_error("Error message")  # Sets level="ERROR" on trace
session.mark_success({...})          # Marks trace as successful

# SpanWrapper/GenerationWrapper methods
span.error("Error message")          # Sets level="ERROR" on span
```

**Result**: Failed traces are now highlighted in Langfuse with ERROR level, visible at a glance.

### Rationale

These optimizations follow key principles:

1. **Native Async**: Use `AsyncOpenAI` for true non-blocking I/O without thread overhead
2. **Parallelism**: Independent operations (page parsing, document assembly) run concurrently via `asyncio.gather()`
3. **Token efficiency**: Fewer tokens = faster responses = lower cost
4. **Output minimization**: Don't ask the LLM to generate redundant data
5. **Input filtering**: Only send relevant data to each agent (changed sections only)
6. **Observability**: Make errors clearly visible in tracing UI for faster debugging
7. **Fail-fast validation**: Catch setup errors before spending resources on LLM calls

### 8.9 Fail-Fast Validation

**Behavior**: Before processing any contract pairs, all input folders are validated upfront. If any folder is missing or contains no images, the entire batch exits with an error.

```
🔍 Validating all input folders...
✅ pair_1/original: Found 5 image files
✅ pair_1/amendment: Found 5 image files
❌ pair_2/original: No image files found
❌ Validation failed. Fix issues before processing.
```

**Rationale**:
- **Validation is cheap, processing is expensive** — Folder checks take milliseconds; LLM calls cost money and time
- **Missing files = setup error** — Should be fixed before spending resources, not silently skipped
- **Clear mental model** — Either all pairs succeed, or you get a clear error list upfront

This is the preferred default for small batches with explicit pair selection. As an improvment measure for large batch jobs where partial results are acceptable, a `--best-effort` flag could be added in the future.

### 8.10 LLM Retry with Correction Prompt

**Problem**: LLM outputs sometimes fail Pydantic validation (e.g., `topics_touched` count doesn't match `summary_of_the_change` count). Without retry, the entire pipeline fails.

**Solution**: Implement automatic retry with a correction prompt that explains the validation error:

```python
async def _call_extraction_llm(self, input_data: dict, span: SpanWrapper, max_retries: int = 2):
    for attempt in range(max_retries + 1):
        try:
            completion = await client.chat.completions.create(...)
            result = ContractAnalysisResult.model_validate(parsed_json)
            return result
        except ValidationError as e:
            if attempt < max_retries and "must match" in str(e):
                # Add correction message to conversation
                correction_msg = CORRECTION_PROMPT.format(
                    error_message=str(e),
                    topics_count=len(parsed.get("topics_touched", [])),
                    summary_count=len(parsed.get("summary_of_the_change", []))
                )
                messages.append({"role": "user", "content": correction_msg})
                continue  # Retry with correction
            raise  # No more retries
```

**Correction Prompt**:
```python
CORRECTION_PROMPT = """Your previous response had a validation error:
{error_message}

Please fix your response. The CRITICAL requirement is:
- topics_touched and summary_of_the_change MUST have EXACTLY THE SAME NUMBER OF ITEMS

Your previous response had {topics_count} topics but {summary_count} summaries.
Return a corrected JSON object with matching counts.
"""
```

**Result**: 
- Self-healing mechanism reduces manual intervention
- Most validation errors are fixed on first retry
- Failed attempts are logged in Langfuse for debugging
- `max_retries=2` provides good balance between reliability and cost

## Test Data Structure

The `data/test_contracts/` directory contains sample scanned contract documents organized as pairs (original + amendment).

```
data/test_contracts/
├── pair_1/
│   ├── original/          # Original contract (5-10 page scans)
│   │   ├── page_01.png
│   │   ├── page_02.png
│   │   └── ...
│   └── amendment/         # Amendment to pair_1 original
│       ├── page_01.png
│       ├── page_02.png
│       └── ...
└── pair_2/
    ├── original/
    │   └── ...
    └── amendment/
        └── ...
```

### Requirements

- **Minimum**: 2 contract pairs
- **Pages per document**: for this project contract of 5 pages long are used for testing
- **Format**: JPEG or PNG
- **Naming**: `page_01.png`, `page_02.png`, etc.

Each pair consists of an original contract and its corresponding amendment, allowing comparison and change detection between versions.

### Expected Differences (pair_1)

The following JSON represents the expected output format. Note that related changes are **grouped by topic** for cleaner output:

```json
{
  "sections_changed": [
    "II. TERM, TERMINATION, AND SUSPENSION",
    "III. COMPENSATION, BILLING, AND EXPENSES",
    "VI. LIMITATION OF LIABILITY"
  ],
  "topics_touched": [
    "Contract Duration and Notice Period",
    "Payment Terms",
    "Liability Cap"
  ],
  "summary_of_the_change": [
    "Contract term extended from 24 to 36 months, cure period for termination due to material breach extended from 15 to 30 days, and notice period for termination for convenience extended from 60 to 90 days (Sec. 2.1, 2.2, 2.3).",
    "Monthly fee increased from 15,000 USD to 18,000 USD and late payment interest rate increased from 1.5% to 2.0% per month (Sec. 3.1, 3.2).",
    "Liability cap increased from three months to six months of fees paid (Sec. 6.2)."
  ]
}
```

**Key changes from original to amended:**
- Term: 24 months → 36 months
- Termination cure period: 15 days → 30 days
- Termination for convenience: 60 days → 90 days
- Monthly fee: $15,000 → $18,000
- Late payment interest: 1.5% → 2.0%
- Liability cap: 3 months → 6 months

### Expected Differences (pair_2)

The following JSON represents the expected output for **Residential Real Estate Purchase Agreement** in `pair_2`. Related changes are grouped by topic:

```json
{
  "sections_changed": [
    "II. PURCHASE PRICE AND PAYMENT TERMS",
    "III. INSPECTIONS AND DUE DILIGENCE",
    "IV. TITLE AND CLOSING"
  ],
  "topics_touched": [
    "Purchase Price and Payment Terms",
    "Inspection Period",
    "Closing Date"
  ],
  "summary_of_the_change": [
    "Purchase price reduced from $475,000 to $465,000, earnest money deposit increased from $15,000 to $20,000, and financing approval deadline extended from 21 to 28 days (Sec. 2.1, 2.2, 2.3).",
    "Inspection period extended from 10 to 14 days (Sec. 3.1).",
    "Closing date extended from April 30, 2024 to May 15, 2024 (Sec. 4.3)."
  ]
}
```

**Key changes from original to amended:**
- Purchase price: $475,000 → $465,000
- Earnest money: $15,000 → $20,000
- Financing deadline: 21 days → 28 days
- Inspection period: 10 days → 14 days
- Closing date: April 30, 2024 → May 15, 2024

## 9. Known Limitations

### 9.1 Maximum Contract Size

The Contextualization Agent uses a **single LLM call** that receives all parsed pages from both contracts. This is limited by the model's context window.

| Model | Context Window | Max Pages (estimated) |
|-------|----------------|----------------------|
| GPT-4o | 128,000 tokens | ~50-60 pages per contract |
| GPT-4o-mini | 128,000 tokens | ~50-60 pages per contract |

**Calculation:**
- ~750 tokens per parsed page (JSON with sections/clauses)
- 128K context - 16K reserved for output = ~110K for input
- 110K / 750 = ~146 total pages (both contracts)
- Safe limit: **~50-60 pages per contract**

**Workaround for larger contracts:** Split into multiple processing batches or implement chunking strategy (not currently implemented).

### 9.2 LLM Output Variability

- **Grouped output format may vary**: The LLM groups related changes by topic, but grouping decisions may differ between runs. The content is accurate, but presentation may vary.
- **Retry mechanism**: Agent 2 has automatic retry with correction prompt (max 2 retries). Agent 1 does not have retry - if it fails, the pipeline fails.

### 9.3 OCR Artifacts

- The system is designed to handle "|" characters from OCR artifacts
- Other OCR errors (misread characters, merged words) may affect accuracy
- Best results with high-quality scanned images (300+ DPI)

### 9.4 Contract Structure Assumptions

- Contracts should have identifiable sections (e.g., "I. AGREEMENT", "II. TERMS")
- Works best with numbered/lettered section hierarchies
- Unstructured or heavily formatted contracts may have reduced accuracy

### 9.5 Language Support

- Currently optimized for **English** contracts only
- Other languages may work but are not tested

### 9.6 No Retry for Agent 1

- Contextualization Agent (Agent 1) does not have a retry mechanism
- If the LLM produces invalid JSON or fails validation, the pipeline fails
- This is acceptable because Agent 1's validation is minimal (just needs aligned sections list)
- Agent 2 has retry because it has stricter validation (count matching)

