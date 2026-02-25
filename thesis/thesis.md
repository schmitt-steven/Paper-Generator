# A Human-in-the-Loop System for Research Paper Generation Using Local Large Language Models

## 1. Introduction
- Simple diagram of the entire workflow (the 6 phases), something like:
![Chevron Diagram](Phase_Diagram.svg)

## 2. Foundations
Important concepts: Language Models, Embedding Models, RAG, (Local) Inference, Prompt Engineering...

## 3. Related Work
CycleResearcher, Agent Laboratory, AI-Researcher, The AI-Scientist, Meow, DORA AI...

## 4. System Design

This chapter explains the design and architecture of the automated research paper generator.
Based on the foundations and related work discussed in the previous chapters, the system is built to solve specific problems
with current AI tools: hallucinations, data privacy concerns and lack of user control [CITE].

The core design philosophy centers around two principles: keeping the process local and keeping the user in charge.
Instead of relying on cloud APIs where data leaves the machine and usage fees might apply, the system is designed to run entirely on local hardware.
Furthermore, rather than trying to fully automate the writing process which lacks transparency, the system uses a Human-in-the-Loop (HITL) approach.
This means the AI acts as a drafter, while the user remains the editor and verifier at each phase of the process.

This chapter focuses on the concepts and logic of the system, while the specific code implementation is explained in Chapter 5.

The design is broken down into four parts:
1. **Requirements**: The specific features and constraints the system must satisfy.
2. **System Architecture**: The high-level structure of the system.
3. **Human-in-the-Loop Strategy**: The interaction pattern enabling the user to verify and correct the AI's output.
4. **Generation Process**: The step-by-step workflow, explaining the logic from initial context analysis to the final PDF compilation.

### 4.1 Requirements

Defining the system's requirements is essential for two reasons:
they establish the functional boundaries of the system and provide the specific criteria for
the evaluation in Chapter 6. While specialized metrics for automated text quality assessment exist [CITE],
their implementation is outside the scope of this work. Instead, the requirements defined below serve as the primary
baseline for verifying the system's core functions.

#### The Approach
For this work, the requirements are based on standard frameworks like the Volere Template [1] and IEEE 830 [2].
However, strictly following these standards would create excessive documentation for a single-developer project with a limited timeline.
Therefore, only the elements necessary for verification were selected: unique identifiers, categorization, titles, and deterministic "shall" statements.
This ensures the requirements are precise and testable, yet avoid unnecessary overhead.

#### Scope: System vs. Model
A fundamental challenge in developing AI tools is distinguishing the contribution of the software from the capabilities of
the used AI models [CITE]. This distinction is important for this work: the semantic quality of a generated research paper depends largely on the
intelligence of the specific LLMs used, rather than just the system architecture.
Therefore, the requirements are intentionally scoped to deterministic system behaviors.
For example, FR4 (Experimentation) requires the system to "generate code, execute it, and save artifacts."
It does not require the system to "produce a scientifically valid experiment."
This distinction allows the system's engineering quality to be verified objectively through programmatic
tests, rather than relying on subjective and time-consuming qualitative user studies, which are out of scope for this work.

#### The Specification

\begin{table}[htbp]
    \centering
    \caption{System Requirements}
    \label{tab:requirements}
    \begin{tabularx}{\textwidth}{l p{2cm} p{3cm} X}
        \toprule
        \textbf{ID} & \textbf{Category} & \textbf{Title} & \textbf{Description} \\
        \midrule
        FR1 & Functional & Context Analysis & The system shall process user data as input to produce a structured research topic definition. \\
        FR2 & Functional & Literature Search & The system shall query external databases to retrieve and store metadata and full-text documents of research papers. \\
        FR3 & Functional & Hypothesis \newline Generation & The system shall derive a formal hypothesis from the provided context. \\
        FR4 & Functional & Experimentation & The system shall generate code, execute it, and save the execution artifacts (logs, plots, data). \\
        FR5 & Functional & Paper Writing & The system shall generate text sections that include citations referenced from the retrieved literature. \\
        FR6 & Functional & Document \newline Compilation & The system shall compile the generated content into a PDF document. \\
        FR7 & Functional & Human-in-the-Loop & The system shall allow the user to view generated artifacts and optionally edit them. \\
        FR8 & Functional & Model Selection & The system shall allow the assignment of LLMs to specific tasks (e.g., Coding vs. Writing). \\
        NFR1 & Non-\newline Functional & Privacy & The system shall process all inference data locally. \\
        NFR2 & Non-\newline Functional & Free Execution & The system shall perform all functions free of charge. \\
        C1 & Constraint & Technology Stack & The system shall be implemented using Python (language), Tkinter (GUI), and LM Studio (inference engine). \\
        \bottomrule
    \end{tabularx}
\end{table}

The requirements, derived from the research objectives in Chapter 1 and the challenges indentified in Chapter 2 and 3,
are listed in Table \ref{tab:requirements}. They are categorized into three groups: functional requirements, non-functional requirements and constraints.

Functional Requirements (FR) define what the system must actually do [CITE].
The requirements FR1-FR6 mirror the standard scientfic method [CITE]:
\begin{description}
    \item \textbf{Observation \& Question:} The process begins with FR1 (Context Analysis) to define the research problem.
    \item \textbf{Background Research:} The system then performs FR2 (Literature Search) to gather existing knowledge.
    \item \textbf{Hypothesis:} Based on this data, the system performs FR3 (Hypothesis Generation).
    \item \textbf{Test:} The core of the scientific work is handled by FR4 (Experimentation), where code is generated and executed.
    \item \textbf{Conclusion:} Finally, the results are translated into a document via FR5 (Paper Writing) and FR6 (Document Compilation).
\end{description}
To ensure the quality of this process, FR7 (Human-in-the-Loop) allows the user to verify the output at every single one of these stages,
while FR8 (Model Selection) allows switching between specialized LLMs for different tasks to improve the quality of the output.

\medskip
Non-Functional Requirements (NFR) define the quality attributes of the system, rather than specific behaviors [CITE].
In this work, the focus is on data privacy and cost efficiency. The goal is to demonstrate that effective research tools do not require reliance on
paid cloud services. Therefore, NFR1 states that all inference data must be processed locally. This ensures that sensitive user inputs and generated
drafts are never sent to third-party AI providers, even while the system queries external databases for literature. NFR2 guarantees the system
operates without usage fees by integrating open-weights models and free APIs only.

\medskip
Constraints (C) define the technical boundaries of the project [1].
C1 restricts the implementation to the specific technology stack chosen for this thesis: Python, Tkinter, and LM Studio.
This constraint ensures the implementation remains feasible within the project scope and guarantees compatibility with consumer-grade hardware.

### 4.2 System Architecture

The system architecture is designed to demonstrate that automated research is feasible using entirely local resources. Besides privacy, the goal is to prove that open-weights models are capable of handling complex scientific workflows if the system is designed correctly.

To achieve this, the architecture avoids a "black box" approach and instead uses a modular pipeline that breaks the research process into smaller, manageable tasks that local LLMs can handle.

As shown in Figure \ref{fig:system_architecture}, the system is divided into four components: the \textbf{Frontend}, the \textbf{Backend}, \textbf{Project Data}, and \textbf{External Services}.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{bilder/system_architecture}
\caption{High-Level System Architecture. The User interacts with the Frontend to trigger Backend modules. The Backend processes Project Data and uses External Services for inference and retrieval.}
\label{fig:system_architecture}
\end{figure}

\paragraph{1. Frontend (User Interface)}
The Frontend is the interface of the application and is built with Tkinter [1]. It is designed to guide the user through the research workflow. Instead of a single chat window, the interface is split into separate screens (e.g., \textit{Research Context}, \textit{Experiment Plan}).
This structured design is critical for working with local LLMs:
\begin{itemize}
\item \textbf{Verification:} Since local LLMs may have lower accuracy than state-of-the-art cloud models, the interface displays the output of every phase. This allows the user to verify the work the the AI before moving to the next step.
\item \textbf{Navigation:} The user can move back and forth between screens. If the model struggles with a specific task (e.g., writing the code for an experiment), the user can edit the LLM's output or return to previous steps and refine the context.
\end{itemize}

\paragraph{2. Backend}
The Backend contains the application logic. It is organized into separate Python modules for each of the six phases: \textit{Context Analysis}, \textit{Literature Search}, \textit{Hypothesis Generation}, \textit{Experimentation}, \textit{Paper Writing}, and \textit{Compilation}.
The Backend is stateless, meaning it simply reads the current files from \textbf{Project Data}, executes a specific task, and saves the result. This design is critical because the user's hardware and model choice are unknown. By breaking the workflow into small, isolated steps, the system reduces the cognitive load and context requirements for the AI. This increases the probability that even smaller models (e.g., 8B parameters) or machines with limited memory can successfully generate a full paper.

\paragraph{3. Project Data}
A key architectural decision was to use a File-Based State instead of a database. The "Project Data" is made up of two folders on the user's hard drive containing:
\begin{itemize}
\item \textbf{User Files:} Inputs provided by the user (e.g., specifications, seed code).
\item \textbf{Artifacts:} Outputs generated by the system (Markdown text, JSON data, Python scripts).
\end{itemize}
This ensures the user is not locked into the tool. Since the "state" is just a set of standard files, the user can open, edit, or fix any part of the project using their preferred text or code editor.

\paragraph{4. External Services}
The system connects to two types of external resources: the local AI model and public paper databases.

\noindent\textbf{Local Inference}
\begin{itemize}
\item \textbf{LM Studio:} The system uses LM Studio [2] for all AI tasks. It runs on a local server (localhost), proving that consumer hardware is sufficient to power the research workflow without sending data to third-party providers (NFR1).
\end{itemize}

\noindent\textbf{Literature Search}
\begin{itemize}
\item \textbf{Semantic Scholar API:} The primary literature search engine [3]. It provides metadata (titles, abstracts, etc.) to identify relevant papers.
\item \textbf{Unpaywall API:} Used to resolve DOIs to legal, open-access PDF links [5].
\item \textbf{arXiv API:} Acts as a fallback [4] to retrieve pre-prints if no published version is available.
\end{itemize}

This setup ensures that while the system relies on the web for knowledge retrieval, the actual reasoning and content generation happen entirely on the user's machine.

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
