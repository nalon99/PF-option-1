# Test Contract Data

This directory contains sample scanned contract documents for testing the contract amendment analyzer.

> **Note:** All contract images are synthetically generated with fictional content. No real legal documents, personal data, or confidential materials are used.

## Directory Structure

```
test_contracts/
├── pair_1/
│   ├── original/           # Original Master Service Agreement (5 pages)
│   │   ├── page_01.png
│   │   ├── page_02.png
│   │   ├── page_03.png
│   │   ├── page_04.png
│   │   └── page_05.png
│   └── amendment/          # Amended Master Service Agreement (5 pages)
│       ├── page_01.png
│       ├── page_02.png
│       ├── page_03.png
│       ├── page_04.png
│       └── page_05.png
└── pair_2/
    ├── original/           # Original Real Estate Purchase Agreement (5 pages)
    └── amendment/          # Amended Real Estate Purchase Agreement (5 pages)
```

## Image Specifications

- **Format**: PNG
- **Resolution**: 2480 × 3508 pixels (A4 at ~300 DPI)
- **File size**: ~500-750 KB per page
- **Pages per document**: 5

---

## Pair 1: Master Service Agreement

**Contract type**: B2B Master Service Agreement between Alpha Corp (Client) and Beta Solutions LLC (Provider)

### Expected Differences

| Section | Original Value | Amended Value |
|---------|----------------|---------------|
| II. Term (2.1) | 24 months | 36 months |
| II. Termination for Cause (2.2) | 15 days cure period | 30 days cure period |
| II. Termination for Convenience (2.3) | 60 days notice | 90 days notice |
| III. Fees (3.1) | $15,000/month | $18,000/month |
| III. Late Payment (3.2) | 1.5% per month | 2.0% per month |
| VI. Liability Cap (6.2) | 3 months of fees | 6 months of fees |

### Expected Output

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

---

## Pair 2: Residential Real Estate Purchase Agreement

**Contract type**: Residential real estate purchase agreement between buyer and seller

### Expected Differences

| Section | Original Value | Amended Value |
|---------|----------------|---------------|
| II. Purchase Price (2.1) | $475,000 | $465,000 |
| II. Earnest Money (2.2) | $15,000 | $20,000 |
| II. Financing Deadline (2.3) | 21 days | 28 days |
| III. Inspection Period (3.1) | 10 days | 14 days |
| IV. Closing Date (4.3) | April 30, 2024 | May 15, 2024 |

### Expected Output

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

---

## Usage

To process these test contracts:

```bash
cd src

# Process pair_1
python contract_agent.py ../data/test_contracts/pair_1/original ../data/test_contracts/pair_1/amendment

# Process pair_2
python contract_agent.py ../data/test_contracts/pair_2/original ../data/test_contracts/pair_2/amendment

# Process all pairs (batch mode)
python contract_agent.py --batch
```

Results are saved to `extraction_output.json` in each pair's folder.
