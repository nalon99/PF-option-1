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

## 3. Setup instructions

## 4. Usage

with example commands

## 5. Expected output

Describe expected output with format sample

## 6. Technical decisions

why two agents? why this model? use at least 100 words

## 7. Langfuse tracing guide

how to view dashboard, at least 50 words

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

The following JSON represents the ground truth differences between the original and amended contracts in `pair_1`:

```json
{
  "sections_changed": [
    "II. TERM, TERMINATION, AND SUSPENSION",
    "III. COMPENSATION, BILLING, AND EXPENSES",
    "VI. LIMITATION OF LIABILITY"
  ],
  "topics_touched": [
    "Contract Duration",
    "Termination Conditions",
    "Payment Terms",
    "Late Payment Penalties",
    "Liability Cap"
  ],
  "summary_of_the_change": [
    "The initial contract term was extended from twenty-four (24) months to **thirty-six (36) months** (Sec. 2.1).",
    "The termination conditions were modified: cure period for material breach extended from fifteen (15) to **thirty (30) days** (Sec. 2.2), and termination for convenience notice extended from sixty (60) to **ninety (90) days** (Sec. 2.3).",
    "The fixed monthly compensation rate was increased from $15,000 USD to **$18,000 USD** (Sec. 3.1).",
    "The late payment interest rate was increased from one and a half percent (1.5%) to **two percent (2.0%)** per month (Sec. 3.2).",
    "The Provider's aggregate liability cap was increased from the total fees paid during the three (3) months to the total fees paid during the **six (6) months** immediately preceding the claim (Sec. 6.2)."
  ]
}
```

### Expected Differences (pair_2)

The following JSON represents the ground truth differences between the original and amended **Residential Real Estate Purchase Agreement** in `pair_2`:

```json
{
  "sections_changed": [
    "II. PURCHASE PRICE AND PAYMENT TERMS",
    "III. INSPECTIONS AND DUE DILIGENCE",
    "IV. TITLE AND CLOSING"
  ],
  "topics_touched": [
    "Purchase Price",
    "Earnest Money Deposit",
    "Financing Timeline",
    "Inspection Period",
    "Closing Date"
  ],
  "summary_of_the_change": [
    "The purchase price was reduced from $475,000 USD to **$465,000 USD** following inspection negotiations (Sec. 2.1).",
    "The earnest money deposit was increased from $15,000 USD to **$20,000 USD** to demonstrate buyer commitment (Sec. 2.2).",
    "The financing approval deadline was extended from twenty-one (21) days to **twenty-eight (28) days** (Sec. 2.3).",
    "The inspection period was extended from ten (10) days to **fourteen (14) days** (Sec. 3.1).",
    "The closing date was extended from April 30, 2024 to **May 15, 2024** (Sec. 4.3)."
  ]
}
```

