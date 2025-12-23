# Sherlock QA Advanced RAG

## 1. Overview
This project implements a Retrieval-Augmented Generation (RAG) based question answering system over the four main Sherlock Holmes novels by Sir Arthur Conan Doyle.
The system retrieves relevant passages from the original texts and conditions a large language model on this context to produce grounded, context-aware answers.

This repository represents the first phase of a multi-part project exploring different AI interaction paradigms over the Sherlock Holmes books.

## 2. Motivation
In addition to studying the effectiveness of RAG for literary question answering, this project is motivated by the goal of systematically exploring and gaining hands-on experience with modern libraries and technologies commonly used in RAG-based systems, including retrieval, evaluation, and experiment tracking frameworks.

## 3. System Architecture

User Query  
     ↓  
Embedding Model (Embed the Query)  
     ↓  
Qdrant Vector DB (Dense Vector Similarity Search)  
     ↓  
Top-K Results (First Retrieval)  
     ↓  
Reranker (Cross Encoder)  
     ↓  
Final Retrieved Context (Top-K/5 Filtering After Reranking)  
     ↓  
Generator (Context + Query, Gemini-2.5-Flash-Lite)  
     ↓  
Final Answer  


## Project Structure

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




