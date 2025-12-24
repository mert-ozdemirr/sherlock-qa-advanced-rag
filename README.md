# Sherlock QA Advanced RAG

## 1. Overview
This project implements a Retrieval-Augmented Generation (RAG) based question answering system over the four main Sherlock Holmes novels by Sir Arthur Conan Doyle.
The system retrieves relevant passages from the original texts and conditions a large language model on this context to produce grounded, context-aware answers.

This repository represents the first phase of a multi-part project exploring different AI interaction paradigms over the Sherlock Holmes books.

## 2. Motivation
In addition to studying the effectiveness of RAG for literary question answering, this project is motivated by the goal of systematically exploring and gaining hands-on experience with modern libraries and technologies commonly used in RAG-based systems, including retrieval, evaluation, and experiment tracking frameworks.

## 3. System Architecture

User Query  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
Embedding Model (Embed the Query)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
Qdrant Vector DB (Dense Vector Similarity Search)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
Top-K Results (First Retrieval)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
Reranker (Cross Encoder)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
Final Retrieved Context (Top-K/5 Filtering After Reranking)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
Generator (Context + Query, Gemini-2.5-Flash-Lite)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓  
Final Answer  


## 4. Data Preparation
The textual corpus used in this project consists of the four main Sherlock Holmes novels authored by Sir Arthur Conan Doyle. 
These long-form literary texts require careful preprocessing to support effective retrieval and downstream generation.

The novels were obtained in plain-text format and treated as a unified corpus. 
Given the narrative structure of the books, preserving logical boundaries such as chapters was considered important during preprocessing.

Data preparation follows a two-stage hierarchical splitting strategy:

1. **Chapter-Level Parsing**  
   Each novel is first split into chapters to preserve high-level semantic and narrative structure.

2. **Recursive Character-Based Chunking**  
   Chapters are further divided into smaller text chunks using a recursive character-based splitting strategy, where chunks are created by cutting at sentence boundaries rather than using fixed-length windows.  
   This approach ensures that each chunk fits within the context constraints of embedding and generation models while maintaining semantic coherence, without introducing overlap between chunks.

The chunk size is treated as a tunable hyperparameter, allowing the impact of different granularity levels on retrieval and generation performance to be systematically studied.

## 5. Retrieval and Generation Pipeline

The system follows a modular Retrieval-Augmented Generation (RAG) pipeline designed to retrieve relevant textual evidence from the corpus and generate grounded answers to user queries. 

### Query-Time Workflow

At inference time, the pipeline proceeds as follows:

1. **Query Embedding**  
   The user’s natural language query is transformed into a dense vector representation using the same embedding model employed during corpus indexing (`sentence-transformers/all-MiniLM-L6-v2` via FastEmbed).  
   This ensures consistency between query and document representations in the embedding space.

2. **Dense Vector Retrieval**  
   The query embedding is used to perform a similarity search against the Qdrant vector database.  
   The top-*K* most similar text chunks are retrieved based on dense vector similarity.

3. **Cross-Encoder Reranking**  
   The initial top-*K* candidates are reranked using a cross-encoder model (`jinaai/jina-reranker-v2-base-multilingual`, via FastEmbed).  
   Unlike dense retrieval, the cross-encoder jointly encodes the query and each candidate passage, enabling more precise semantic relevance estimation.

5. **Context Filtering**  
   After reranking, only the top *K/5* passages are retained.  
   This step reduces noise and compresses the retrieved context, ensuring that only the most relevant information is passed to the generation stage.

6. **Answer Generation**  
   The filtered context, together with the original user query, is provided to a generative language model (`Gemini 2.5 Flash Lite`, via Agno Agent).  
   The model generates a final answer conditioned on the retrieved evidence.

7. **Final Output**  
   The generated answer is returned to the user as the system’s response.

## 7. Evaluation Methodology
To systematically assess the performance of the Retrieval-Augmented Generation pipeline, an automated evaluation framework was designed combining synthetic data generation, structured RAG-specific metrics, and LLM-based judging. 
The goal of this evaluation setup is to measure both retrieval quality and answer faithfulness in a controlled and reproducible manner.

### Synthetic Dataset Construction

As no labeled question–answer dataset exists for the selected Sherlock Holmes novels, a synthetic evaluation dataset was constructed directly from the corpus using the **RAGAS** test set generation framework. 
RAGAS enables structured, controllable, and corpus-grounded generation of evaluation data, making it well suited for systematic RAG assessment.

The resulting dataset consists of **420 question–answer pairs**, composed of:
- **320 single-hop questions**
- **100 multi-hop questions**

A set of predefined **personas** (e.g., literary analyst, detective psychologist, prose critic) is used to guide question formulation, encouraging diversity in question style and thematic focus while remaining grounded in the source text.

The following personas are used:

- **Literature Archivist**  
  Focuses on the structural organization of the novels, including parts, chapters, and publication details.  
  This persona encourages questions related to narrative layout and document-level structure.

- **Story Analyst**  
  Examines events, clues, timelines, and plot progressions throughout the novels.  
  Questions generated under this persona often emphasize causal relationships and narrative development.

- **Consulting Detective Psychologist**  
  Analyzes character traits, motivations, and behavioral patterns of major figures such as Holmes and Watson.  
  This persona supports questions requiring reasoning about character psychology and interpersonal dynamics.

- **Crime Pattern Statistician**  
  Identifies recurring crime types, investigative strategies, and clue structures across different cases.  
  Questions from this perspective often involve pattern recognition and comparative reasoning.

- **Prose Critic**  
  Studies Arthur Conan Doyle’s writing style, tone, and narrative choices.  
  This persona enables stylistic and meta-level questions grounded in textual evidence.

By incorporating multiple personas, the synthetic dataset avoids stylistic homogeneity and better reflects the diversity of real user queries, while remaining fully grounded in the source corpus via the knowledge graph.

Both question types are generated using a corpus-level **knowledge graph (KG)** representation derived from the processed text chunks.

#### Knowledge Graph Construction

A knowledge graph is constructed from the chunked corpus, where nodes represent individual text chunks and edges encode semantic relationships based on shared keyphrases. 
This graph structure provides an explicit representation of semantic connectivity within the corpus and serves as the foundation for both single-hop and multi-hop query synthesis.

#### Single-Hop Question Generation

Single-hop questions are generated by sampling individual nodes from the knowledge graph. 
Each question is designed to be answerable using evidence contained within a single text chunk, enabling focused evaluation of retrieval accuracy and answer grounding.

Example:
=== TEST ITEM 320 ===  
QUERY:
What is a blow-pipe in the story?

REFERENCE:
A blow-pipe is mentioned in the context where a character, Tonga, managed to shoot a dart at the protagonists while in a boat. It indicates that he had lost all his darts except for the one that was in his blow-pipe at the time.

PERSONA:
Crime Pattern Statistician

#### Multi-Hop Question Generation

Multi-hop questions are generated by sampling pairs of semantically connected nodes from the knowledge graph. 
Node pairs are selected based on shared keyphrases, ensuring that the resulting questions require reasoning across multiple passages.

For each selected node pair, overlapping concepts are identified and matched with appropriate personas to guide query construction. 
The resulting questions require the integration of information from both nodes, explicitly testing the system’s multi-hop reasoning capabilities.

Example:
=== TEST ITEM 25 ===  
QUERY:
How does the power of observation relate to the power of comparison in Sherlock Holmes's detective work?

REFERENCE:
The power of observation allows Sherlock Holmes to notice specific details, such as the fact that Watson visited the Wigmore Street Post-Office, while the power of comparison enables him to draw parallels between different cases, like those at Riga in 1857 and St. Louis in 1871. Together, these powers enhance his analytical reasoning, allowing him to unravel complex cases.

PERSONA:
Consulting Detective Psychologist

#### Language Model for Synthetic Data Generation

Synthetic questions and corresponding reference (“golden”) answers are generated using a lightweight OpenAI model (**GPT-4o-mini**) through the RAGAS framework. 
This model was selected to balance generation quality and computational efficiency during dataset construction.

#### Rationale for Synthetic Evaluation

Synthetic data generation enables:
- Controlled evaluation across different reasoning complexities
- Scalable test set construction without manual annotation
- Reproducible benchmarking for hyperparameter optimization

All generated questions and reference answers remain grounded in the original corpus via the knowledge graph, ensuring that evaluation faithfully reflects the underlying textual evidence.

### Evaluation Metrics

Evaluation is based on a combination of RAG-specific metrics and reference-based comparison:

#### RAG Triad Metrics

Evaluation of the RAG pipeline is performed using the **RAG Triad** metrics as implemented in the **DeepEval** framework. 
All three metrics are computed using an **LLM-as-a-judge** paradigm, where a separate evaluation model assesses different aspects of system behavior based on structured prompts.

<img width="1921" height="1099" alt="RAG_Triad" src="https://github.com/user-attachments/assets/e8f3d0e1-c47c-40c1-9c82-b1fde6a7c066" />

- **Answer Relevance**  
  Answer Relevance evaluates how well the generated answer addresses the user query.  
  In DeepEval, this metric decomposes the generated answer into individual statements and uses an evaluation LLM to determine whether each statement is relevant to the input question.  
  The final score reflects the proportion of answer content that directly contributes to answering the query, independent of the retrieved context. 

- **Groundedness (Faithfulness)**  
  Groundedness measures whether the claims made in the generated answer are supported by the retrieved context.  
  DeepEval computes this metric by first extracting factual claims from the generated answer and then verifying whether each claim is entailed by the provided retrieval context, explicitly penalizing contradictions.  
  This metric focuses on detecting ungrounded or hallucinated content introduced during generation. 

- **Contextual Relevance**  
  Contextual Relevance assesses the quality of the retrieved passages themselves.  
  The evaluation model analyzes the retrieval context by extracting statements from the retrieved chunks and determining whether they are relevant to the user query.  
  This metric directly reflects retriever performance by measuring how much of the retrieved context is useful for answering the question. 

Each metric produces both a numerical score and a natural-language explanation, enabling interpretable and fine-grained analysis of system behavior. 
Together, the RAG Triad metrics provide a balanced evaluation of retrieval effectiveness, answer relevance, and generation faithfulness.

#### Golden Answer Comparison

In addition to the RAG Triad metrics, generated answers are evaluated against the corresponding synthetic reference (“golden”) answers using the **Answer Correctness** metric provided by the **DeepEval** framework.

Answer Correctness measures whether the generated answer is factually consistent with and semantically equivalent to the reference answer. 
The metric is computed using an **LLM-as-a-judge** approach, where an evaluation model compares the generated response to the golden answer and assigns a score in the range \([0, 1]\), with higher values indicating greater correctness.

This metric provides a complementary signal to the RAG Triad by focusing on **answer-level correctness and completeness**, independent of how the answer was retrieved. 
While RAG Triad metrics assess retrieval quality and grounding, Answer Correctness directly evaluates whether the system ultimately produces the correct answer content.

### Score Aggregation

All evaluation components are combined using **equal-weight aggregation**:

- Each RAG Triad metric is weighted equally. (Each component contributes 1/6 of the final score.)
- The aggregated RAG Triad score and the golden-answer comparison score are then combined with equal importance. (Correctness contributes 0.5 to the final score.)

This aggregation strategy avoids overemphasizing any single metric and encourages balanced system behavior across retrieval and generation stages.

### LLM-as-a-Judge Framework

All evaluation metrics are computed using an LLM-as-a-judge approach implemented via the **DeepEval** framework. 
A lightweight but capable language model (**GPT-5 Nano**) is used as the evaluation judge to assess answer relevance, groundedness, and contextual alignment.

This automated evaluation approach enables scalable and consistent assessment across large hyperparameter search spaces while maintaining semantic sensitivity.


## 8. Results & Observations

This section summarizes the outcomes of the hyperparameter optimization experiments and highlights key qualitative observations derived from systematic evaluation. 
The focus is on understanding how different design choices affect RAG system behavior rather than claiming absolute performance.

### Hyperparameter Optimization Results

Hyperparameter optimization was conducted in two stages using **Bayesian search**, with the objective of identifying configurations that balance retrieval quality, answer faithfulness, and generation stability. 
All experiments were logged and visualized using **Weights & Biases (W&B)**, enabling systematic comparison across runs and improved interpretability of parameter interactions.

#### Stage 1: Broad Search

In the first stage, a broad hyperparameter space was explored to identify promising regions for further refinement. 
The search space included variations in:

- **Chunk size**: \[500, 1500, 3000\]
- **Top-K retrieval**: \[5, 10, 15\]
- **Generation temperature**: \[0.03, 0.15, 0.40\]
- **Top-p sampling**: \[0.45, 0.65, 0.90\]

To control computational cost and isolate the effects of these parameters, the embedding model, reranker, and generative model were held fixed throughout this stage.

Figure below presents a parallel-coordinates visualization of the first-stage search, generated using W&B. 
Each line corresponds to a single hyperparameter configuration, with color intensity indicating the aggregated evaluation score.

From this exploration, several high-performing trends emerged, most notably the consistent advantage of smaller chunk sizes and lower generation temperatures.

<img width="13056" height="6656" alt="first_hyperparam_opt_bests png" src="https://github.com/user-attachments/assets/64a8192f-0370-4fb2-910a-76f5fb1194a7" />

Here two best configurations share the same chunk size of 500 chars, and top-k value of 15.  

Also, the best score is 0.6986.

#### Stage 2: Focused Refinement

Based on the observations from Stage 1, a second, more focused Bayesian search was conducted over a refined parameter range. 
In this stage, chunk size was fixed to the most promising value, and retrieval depth and generation parameters were explored more granularly:

- **Chunk size**: 500  
- **Top-K retrieval**: \[10, 15, 20, 25\]
- **Generation temperature**: \[0.10, 0.15, 0.20\]
- **Top-p sampling**: \[0.87, 0.90, 0.93\]

Figure below shows the corresponding W&B visualization for the second-stage optimization. 
This focused search enabled clearer separation between configurations and more stable convergence toward an optimal parameter set.

<img width="13056" height="6656" alt="second_hyperparam_opt_bests png" src="https://github.com/user-attachments/assets/4343ea25-d455-486b-844e-3d392e38ec03" />

The highlighted configurations have more final score than the best final score of the first optimization phase (> 0.6986).

#### Best Configuration

The best-performing configuration identified across both stages is:

- **Chunk size**: 500  
- **Top-K retrieval**: 25  
- **Temperature**: 0.10  
- **Top-p**: 0.90       


- **Final Score**: 0.7050

This configuration achieved the highest aggregated evaluation score under the equal-weight combination of **RAG Triad metrics** and **Answer Correctness**, indicating a favorable balance between retrieval precision, contextual grounding, and answer correctness.

Note: Here, the top-k numbers refer to the first stage retrieval chunk count. After reranking, it actually selects top-k/5 chunks. Top-k value of 25 corresponds to 5 chunk supply to the generative llm.

### Observed Trends

Several consistent trends were observed across the hyperparameter search:

- **Smaller chunk sizes** generally led to improved retrieval precision and grounding, likely due to reduced topic dilution within individual chunks.
- **Higher retrieval depth (Top-K)** improved multi-hop performance by increasing the likelihood that all relevant evidence was available prior to reranking.

These trends align with expected behavior in RAG systems and reinforce the importance of jointly tuning retrieval and generation parameters.

## 9. Application & Interface

To demonstrate the practical usability of the RAG system beyond offline experiments, the pipeline is integrated into an interactive application using **Agno OS** and **Agno UI**.

Agno OS provides the orchestration layer for managing the RAG workflow as an application, while Agno UI serves as the user-facing interface for submitting queries and viewing generated responses.

### Interaction Flow

The interaction flow of the application proceeds as follows:

1. The user submits a natural language question through the Agno UI interface.
2. The query is forwarded to the backend RAG pipeline managed by Agno OS.
3. The retrieval and generation pipeline is executed, including embedding, retrieval, reranking, and answer generation.
4. The generated answer is returned and displayed in the UI, optionally alongside supporting contextual information.

This setup enables real-time interaction with the system and allows qualitative inspection of system behavior under different queries and configurations.

### User Interface

Figure below shows a screenshot of the Agno UI used to interact with the system. 
The interface provides a lightweight and intuitive environment for querying the Sherlock Holmes corpus, facilitating rapid testing and demonstration of the RAG pipeline.

<img width="2986" height="1688" alt="image" src="https://github.com/user-attachments/assets/5e1ad2c8-90ae-4ca4-874c-eb216df232f3" />

By integrating the RAG system into an application framework, this project demonstrates not only model and retrieval performance, but also system-level integration and usability considerations.

## 10. Limitations

## Limitations

While the system demonstrates effective end-to-end behavior and supports systematic experimentation, several limitations should be noted.

- **Fixed Model Choices**  
  To control computational cost and reduce experimental complexity, the embedding, reranking, and generation models were held fixed during hyperparameter optimization.  
  Exploring stronger or more specialized models at each stage may further improve performance.

- **Retrieval Depth Constraints**  
  The maximum retrieval depth (Top-K) was limited for efficiency reasons.  
  Increasing Top-K may improve recall for complex queries, particularly in multi-hop scenarios, at the cost of additional computation.

- **Multi-Hop Reasoning Limitations**  
  Multi-hop question answering is implemented using a classical RAG architecture.  
  Such architectures may struggle with deeper reasoning chains, as they rely on single-pass retrieval and generation rather than explicit multi-step reasoning or planning mechanisms.

- **Stateless Interaction (No Chat History)**  
  The current application does not maintain conversational state.  
  Each user query is processed independently, and the full retrieval pipeline is executed for every input.  
  While this simplifies system design, it limits the system’s ability to support contextual follow-up questions or multi-turn dialogue.

- **Prompt Design Constraints**  
  The generation prompt is currently static and manually designed.  
  Further prompt engineering and systematic prompt refinement may lead to improved answer quality, and controllability.

- **Cost Constraints**  
  Model selection, hyperparameter ranges, and evaluation scale were influenced by practical cost considerations.  
  More extensive experimentation and higher-capacity models could be explored under less restrictive resource constraints.

- **Lack of Human Evaluation**  
  All evaluations are performed using automated LLM-as-a-judge metrics.  
  Human evaluation could provide additional qualitative insight, particularly for nuanced or interpretive questions.
