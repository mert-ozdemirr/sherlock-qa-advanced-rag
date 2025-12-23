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

## Setup

### Prerequisites
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








