# A Human-in-the-Loop System for Research Paper Generation Using Local Large Language Models

## 1. Introduction
- Simple diagram of the entire workflow (the 6 phases), something like:
![Chevron Diagram](Phase_Diagram.svg)

## 2. Foundations
Important concepts: Language Models, Embedding Models, RAG, (Local) Inference, Prompt Engineering...

## 3. Related Work
CycleResearcher, Agent Laboratory, AI-Researcher, The AI-Scientist, Meow, DORA AI...

## 4. System Design

### 4.3 Human-in-the-Loop Strategy

### 4.4 Generation Process (Process Specification)

#### 4.4.1 Context Analysis
![Context Analysis Diagram](context_analysis.svg)

#### 4.4.2 Literature Search
![Lit Search](lit_search.svg)

#### 4.4.3 Hypothesis Generation
![Hyp Gen](hyp_gen.svg)

#### 4.4.4 Experimentation

#### 4.4.5 Paper Writing

#### 4.4.6 Document Compilation

## 5. Implementation

## 6. Evaluation

### 6.1 Methodology

### 6.2 Requirements Verification
| ID | Requirement | Method | Fit Criterion (Pass Condition) |
| :--- | :--- | :--- | :--- |
| FR1 | Context Analysis | Demonstration | Upon inputting user topic/requirements, the system generates a context.json (or paper_concept.md) file containing a structured title, abstract, and research questions. |
| FR2 | Literature Search | Inspection | The output directory contains a non-empty papers.json list and a subdirectory containing downloaded .pdf files corresponding to the search results. |
| FR3 | Hypothesis Generation | Inspection | The system produces a hypothesis.md file containing a distinct hypothesis statement, rationale, and success criteria derived from the context. |
| FR4 | Experimentation | Test | The system creates an experiment.py file, and execution logs confirm the subprocess ran to completion (Exit Code 0) and saved at least one artifact (plot or result JSON). |
| FR5 | Paper Writing | Inspection | Generated markdown files contain text with citation keys (e.g., [@Author2023]), and these keys exist in the generated bibliography.bib file. |
| FR6 | Document Compilation | Test | The compilation subprocess returns Exit Code 0, and a valid paper.pdf file with size > 0 KB exists in the output/latex/result/ directory. |
| FR7 | Human-in-the-Loop | Demonstration | After generating an artifact, the user manually edits the file and clicks "Continue"; the subsequent generation phase utilizes the modified text rather than the original AI generation. |
| FR8 | Model Selection | Log Analysis | Application logs record a successful API call to the inference server to unload the previous model and load the target model ID when the workflow transitions tasks. |
| NFR1 | Privacy | Analysis | Network traffic monitoring (or Config Inspection) confirms 0 outbound packets to commercial LLM API endpoints; all inference traffic targets 127.0.0.1 (localhost). |
| NFR2 | Free Execution | Inspection | Configuration inspection confirms no API keys for paid services are present, and the system executes without authentication errors related to billing. |
| C1 | Technology Stack | Inspection | Source code review confirms the presence of .py files importing tkinter and logic making HTTP requests to a local LM Studio server port. |

### 6.3 Proof of Concept (Case Study)
- Test entire workflow with a problem we know the answer to
- Did it generate a paper and does it come to the right conclusion?

## 7. Discussion

## 8. Conclusion + Future Work
