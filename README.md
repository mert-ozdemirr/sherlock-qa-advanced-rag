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

### 7. Evaluation Methodology


### Installation

## Configuration

## Usage

## Data

## Evaluation

## Experiments

## Results (Optional / If Available)

## Project Status

## Roadmap (Optional)

## Limitations

## Notes

## License









