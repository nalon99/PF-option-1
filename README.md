# Legal Contract Amendment Analyzer

## 1. Project Description

This project implements an AI-powered system for analyzing legal contract amendments from scanned document images. The system automatically identifies and extracts differences between an original contract and its amended version, producing structured output suitable for legal databases, compliance dashboards, and review workflows.

The core innovation is a **two-agent collaborative architecture** that mimics how legal analysts work: first understanding document context and structure, then identifying specific changes. The system uses **GPT-4o's multimodal capabilities** to parse scanned contract images directly, eliminating the need for separate OCR preprocessing and handling real-world document quality variations (different resolutions, scan artifacts, varied formatting).

Key capabilities include:
- **Multimodal document parsing**: Converts scanned images to structured text preserving document hierarchy (sections, clauses, subsections)
- **Two-agent collaboration**: Agent 1 contextualizes and aligns documents; Agent 2 extracts specific changes
- **Pydantic-validated output**: Ensures structured, type-safe results with `sections_changed`, `topics_touched`, and `summary_of_the_change` fields
- **Complete observability**: Langfuse tracing captures every step for debugging, cost analysis, and performance monitoring

> **Note:** All contract images used in this project are synthetically generated with fictional content. No real legal documents, personal data, or confidential materials are used.

---

## 2. Architecture and Agent Workflow

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

### Workflow Explanation

The pipeline processes contract pairs through four distinct stages. First, the **Image Parser** uses GPT-4o's vision capabilities to convert each scanned page into structured JSON, preserving the hierarchical structure (sections → clauses → subclauses). Pages are processed in parallel using `asyncio.gather()` for optimal performance.

Second, **Agent 1 (Contextualization)** receives all parsed pages from both contracts in a single LLM call. It merges page-spanning sections (detecting where content continues across pages), aligns corresponding sections between the original and amended documents, and flags sections with meaningful changes. This single-call approach gives the LLM full context for better alignment decisions.

Third, **Agent 2 (Extraction)** receives only the sections marked as changed. It analyzes the differences and produces structured output with sections changed, topics touched, and detailed summaries. If the output fails Pydantic validation, the agent automatically retries with a correction prompt.

Finally, the validated output is saved as JSON, with all steps traced in Langfuse for observability.

---

## 3. Setup Instructions

### Prerequisites

- Python 3.10+
- OpenAI API key (with GPT-4o access)
- Langfuse account (free tier available at [langfuse.com](https://langfuse.com))

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd PF-Option-1
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```

5. **Edit `.env` with your API keys:**
   ```env
   # OpenAI Configuration
   OPENAI_API_KEY=sk-your-openai-api-key-here
   OPENAI_MODEL=gpt-4o
   
   # Optional: Use OpenRouter instead of OpenAI directly
   USE_OPEN_ROUTER=false
   
   # Langfuse Configuration (get keys from https://cloud.langfuse.com)
   LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
   LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

### Test Images Setup

Place your contract images in the following structure:
```
data/test_contracts/
├── pair_1/
│   ├── original/          # Original contract pages
│   │   ├── page_01.png
│   │   ├── page_02.png
│   │   └── ...
│   └── amendment/         # Amended contract pages
│       ├── page_01.png
│       ├── page_02.png
│       └── ...
└── pair_2/
    ├── original/
    └── amendment/
```

**Image requirements:**
- Format: PNG or JPEG
- Naming: `page_01.png`, `page_02.png`, etc.
- Recommended: 300+ DPI for best OCR accuracy

---

## 4. Usage

### Direct Mode: Two Folder Paths

Process a single contract pair by providing paths to the original and amendment folders:

```bash
cd src

# Basic usage - provide two folder paths
python contract_agent.py /path/to/original /path/to/amendment

# With custom output file
python contract_agent.py /path/to/original /path/to/amendment --output results.json

# Example with test data
python contract_agent.py ../data/test_contracts/pair_1/original ../data/test_contracts/pair_1/amendment
```

### Batch Mode: Process from test_contracts/

Process multiple contract pairs from the `data/test_contracts/` directory:

```bash
# Process all pairs in data/test_contracts/
python contract_agent.py --batch

# Process specific pairs
python contract_agent.py --batch pair_1
python contract_agent.py --batch pair_1 pair_2
```

### Example Output (Direct Mode)

```
======================================================================
CONTRACT AMENDMENT ANALYSIS
======================================================================

📂 Original:  ../data/test_contracts/pair_1/original
📂 Amendment: ../data/test_contracts/pair_1/amendment

🔍 Validating input folders...
✅ Original: Found 5 image files
✅ Amendment: Found 5 image files

🔍 Validating all input folders...
✅ pair_1/original: Found 5 image files
✅ pair_1/amendment: Found 5 image files
✅ pair_2/original: Found 5 image files
✅ pair_2/amendment: Found 5 image files

🔄 Processing 2 pair(s) concurrently...

======================================================================
STEP 1: IMAGE PARSING (parallel)
======================================================================
✅ Parsed 5 pages from original contract
✅ Parsed 5 pages from amendment contract

======================================================================
STEP 2: AGENT 1 - CONTEXTUALIZATION
======================================================================
✅ Agent 1 complete:
   - Aligned sections: 7
   - Sections with changes: 3

======================================================================
STEP 3: AGENT 2 - EXTRACTION (HANDOFF FROM AGENT 1)
======================================================================
✅ Agent 2 complete:
   - Sections changed: 3
   - Topics touched: 3
   - Summary items: 3

💾 Results saved to: data/test_contracts/pair_1/extraction_output.json
```

---

## 5. Expected Output Format

The system produces a JSON file with three required fields, validated by Pydantic:

```json
{
  "sections_changed": [
    "II. TERM, TERMINATION, AND SUSPENSION",
    "III. COMPENSATION, BILLING, AND EXPENSES",
    "VI. LIMITATION OF LIABILITY"
  ],
  "topics_touched": [
    "Contract Duration and Termination",
    "Payment Terms",
    "Liability Cap"
  ],
  "summary_of_the_change": [
    "Contract term extended from 24 to 36 months, cure period extended from 15 to 30 days, and notice period extended from 60 to 90 days (Sec. 2.1, 2.2, 2.3).",
    "Monthly fee increased from $15,000 to $18,000 and late payment interest increased from 1.5% to 2.0% (Sec. 3.1, 3.2).",
    "Liability cap increased from three months to six months of fees paid (Sec. 6.2)."
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `sections_changed` | `List[str]` | Section identifiers that were modified (e.g., "II. TERM, TERMINATION") |
| `topics_touched` | `List[str]` | Business/legal topic categories affected (e.g., "Payment Terms") |
| `summary_of_the_change` | `List[str]` | Detailed descriptions of changes with section references |

**Validation rules:**
- All fields are required and must be non-empty lists
- `topics_touched` and `summary_of_the_change` must have equal counts (1:1 mapping)
- Summary items must be at least 20 characters (meaningful descriptions)

---

## 6. Technical Decisions

### Why Two Agents Instead of One?

The two-agent architecture separates **understanding** from **extraction**, mirroring how legal professionals analyze documents. Agent 1 (Contextualization) focuses on document structure: merging multi-page sections, aligning corresponding sections between documents, and identifying where changes exist. Agent 2 (Extraction) focuses on content analysis: understanding what specifically changed and grouping related changes by business topic.

This separation provides several benefits: (1) **cleaner prompts** - each agent has a focused task rather than a complex multi-step prompt; (2) **better accuracy** - Agent 2 only receives sections marked as changed, reducing noise and context length; (3) **debuggability** - intermediate output (`contextualization_output.json`) can be inspected to understand alignment decisions before extraction.

### Why GPT-4o?

GPT-4o was selected for its **native multimodal capabilities** and **large context window** (128K tokens). Unlike separate OCR + text LLM pipelines, GPT-4o processes images directly, understanding document layout, tables, and formatting in context. This eliminates OCR preprocessing errors and handles real-world document quality variations (scans, photos, varying resolutions).

The 128K context window supports contracts up to ~50-60 pages per document, sufficient for most legal contracts. GPT-4o also provides reliable JSON mode output (`response_format={"type": "json_object"}`), critical for producing valid structured output.

### Prompt Engineering Strategy

Prompts are designed for **token efficiency** and **output reliability**:
- Compact JSON examples (no pretty-printing) reduce input tokens by ~25%
- Single-line rule statements instead of verbose explanations
- Explicit output format specifications prevent ambiguity
- Agent 2 uses a **retry mechanism with correction prompts** - if Pydantic validation fails, the agent sends a follow-up message explaining the error, allowing the LLM to self-correct

---

## 7. Langfuse Tracing Guide

### Accessing the Dashboard

1. Log in to [Langfuse Cloud](https://cloud.langfuse.com) or your self-hosted instance
2. Select your project from the dashboard
3. Navigate to **Traces** in the left sidebar

### Understanding Traces

Each contract analysis creates a **trace** containing:
- **Session root span**: The overall workflow with success/error status
- **Generation spans**: Individual LLM calls with model, tokens, latency, and cost
- **Handoff spans**: Agent-to-agent data transfers
- **Validation spans**: Pydantic validation results

### Key Metrics to Monitor

| Metric | Where to Find | Why It Matters |
|--------|---------------|----------------|
| Total cost | Trace summary | Budget monitoring |
| Token usage | Generation details | Prompt optimization |
| Latency | Trace timeline | Performance bottlenecks |
| Error rate | Traces with ERROR level | Reliability tracking |

### Filtering Traces

Use filters to find specific traces:
- **By contract**: Search for `contract_pair_id` in metadata
- **By status**: Filter by `level=ERROR` to see failures
- **By time**: Use date range filters for production monitoring

Failed traces are marked with `level="ERROR"` for clear visibility - no need to inspect output to identify issues.

---

## 8. Performance Optimizations

Several optimizations reduce latency and cost:

| Optimization | Before | After | Improvement |
|--------------|--------|-------|-------------|
| Image parsing | Sequential (~15s) | Parallel async (~3s) | **80%** |
| Contextualization | 3 LLM calls (~24s) | 1 LLM call (~8-10s) | **60%** |
| Extraction input | All sections | Changed only | **~70% fewer tokens** |
| Validation failures | Manual re-run | Auto-retry | **Self-healing** |
| **Total pipeline** | ~43s+ | ~15-20s | **~60%** |

Key techniques:
- **AsyncOpenAI**: Native async for true non-blocking I/O
- **asyncio.gather()**: Parallel page parsing
- **Compact JSON**: Minimal serialization reduces token count
- **Input filtering**: Agent 2 only receives changed sections
- **Fail-fast validation**: Folder checks before expensive LLM calls

---

## 9. Known Limitations

### Maximum Contract Size
- ~50-60 pages per contract (128K context window limit)
- Workaround: Split larger contracts into batches

### LLM Output Variability
- Topic grouping may vary between runs (content accurate, presentation varies)
- Agent 2 retries on validation failure; Agent 1 does not

### OCR and Document Quality
- Handles "|" artifacts from OCR
- Best results with 300+ DPI scans
- English contracts only (other languages untested)

### Contract Structure
- Requires identifiable sections (e.g., "I. AGREEMENT", "II. TERMS")
- Works best with numbered/lettered hierarchies
