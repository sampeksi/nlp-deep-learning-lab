# NLP Deep Learning Lab

This repository contains work for an NLP (Natural Language Processing) deep learning, covering various fundamental and advanced topics in natural language processing using deep learning techniques.

## Directories

### Sentiment Analysis
Implementation of sentiment classification models to analyze and predict sentiment in text data.

### Language Modeling
Building and training language models to predict next tokens and understand sequential patterns in text. Includes work on probabilistic language models.

### Machine Translation
Neural machine translation implementation with encoder-decoder architectures. Includes vocabulary files for German-English translation (`de_vocab.json`, `en_vocab.json`).

### RAG (Retrieval-Augmented Generation)
Exploring Retrieval-Augmented Generation with vector databases (Chroma) for enhanced language model responses with external knowledge retrieval.

### WMasked Language Modeling (MLM)
Training and fine-tuning BERT-based masked language models. Includes custom tokenizers and model checkpoints for MLM tasks.

###  Named Entity Recognition (NER)
Fine-tuning models for named entity recognition tasks to identify and classify entities in text (persons, organizations, locations, etc.).

### Chatbot
Conversational AI system integrating Dense Passage Retrieval (DPR), BERT-QA, and T5 for summarization. Retrieves relevant passages from Project Gutenberg books, answers questions, and generates summaries on agricultural topics.


## Technologies

- PyTorch / TensorFlow for deep learning
- Hugging Face Transformers for pre-trained models
- Jupyter Notebooks for interactive development
- ChromaDB for vector storage and retrieval

## Setup

1. Install Python dependencies (PyTorch, Transformers, etc.)
2. Run the Jupyter notebooks in each weekly folder
3. Model weights and vocabularies will be downloaded/generated during training
