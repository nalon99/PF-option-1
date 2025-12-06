"""
Entry point that: 
(1) accepts two image file paths as arguments, 
(2) calls multimodal LLM to parse both images, 
(3) executes Agent 1 (contextualization), 
(4) executes Agent 2 (change extraction), 
(5) validates output with Pydantic, 
(6) returns structured JSON. 

Must be runnable from command line.

What is expected to accomplish:
 - Agent 1 (Contextualization): Reads both documents, understands context, identifies corresponding sections 
 - Agent 2 (Change Extraction): Receives Agent 1's analysis, extracts specific changes 
 - Agents have explicit handoff mechanism (Agent 1 output > Agent 2 input)
 - Uses LangChain agents OR custom agent implementation with tool calling 
 - Each agent has distinct system prompts and responsibilities (visible in code)
 - Trace shows sequential agent execution: Image Parsing > Agent 1 > Agent 2 > Output 
 
A __main__ function is expected to do the following clear separation of these steps:

- Parse the images using the image_parser.py
- Call Agent 1 (contextualization)
- Call Agent 2 (change extraction)
- Validate the output using the models.py
- Return the structured JSON

- No code duplication (DRY principle followed)
- try-except blocks for API calls and errors
- functions have type hints, follows style guide, no security issues
"""
