# Natural Language Processing Projects

This repository contains two NLP projects developed as part of the CSE431 – Natural Language Processing with Machine Learning course at Aydin Adnan Menderes University.

## Project 1: Named Entity Recognition on Medical Text

This project focuses on performing Named Entity Recognition (NER) on medical transcriptions to identify diseases, drugs, and drug doses.

### Key Features
- Used multiple NLP models including:
  - en_core_sci_sm
  - en_core_sci_md 
  - en_ner_bc5cdr_md
- Implemented various machine learning models:
  - Multinomial Naïve Bayes
  - Random Forest
  - XGBoost
  - LightGBM
  - 1D CNN with LSTM and GRU
- Utilized both TF-IDF Vectorizer and Bag-of-Words (CountVectorizer) approaches
- Applied SMOTE for handling class imbalance
- Evaluated model performance using confusion matrices and classification metrics

### Results
- Achieved best performance with Random Forest using CountVectorizer
- Multinomial Naïve Bayes showed strong results with some test cases
- Successfully identified and classified medical entities in transcription texts

## Project 2: Word2Vec Models Analysis

This project involves analyzing and comparing different Word2Vec models using both Turkish and English lexicons.

### Key Features
- Trained custom Word2Vec model on Turkish Wikipedia dump
- Utilized multiple pre-trained models:
  - Custom Turkish Wikipedia model
  - 2018 trmodel
  - Fasttext vectors
  - Glove Twitter vectors
  - Word2Vec Google vectors
- Implemented lexicon-based analysis using:
  - NRC-VAD Lexicon
  - MTL-Grouped Lexicon
- Calculated mean values for emotional dimensions (Valence, Arousal, Dominance)
- Performed comparative analysis across models and languages

### Results
- Successfully generated word embeddings and found similar words
- Created comprehensive comparison tables for lexicon values
- Analyzed emotional dimensions across different semantic categories

## Setup and Installation

1. Install Jupyter Notebook
2. Install required libraries:
```bash
pip install spacy scispacey pandas numpy matplotlib seaborn nltk gensim
```

## Data Sources
- Medical transcriptions dataset from Kaggle
- Turkish Wikipedia dump
- NRC-VAD Lexicon
- MTL-Grouped Lexicon

## Authors
- Burak TÜZEL
- Talha Alper ASAV

## Course Information
CSE431 – Natural Language Processing with Machine Learning 2023/2024  
Lecturer: Asst. Prof. Dr. Fatih SOYGAZİ  
Aydin Adnan Menderes University, Engineering Faculty  
Computer Science Engineering Department

## References
1. [Medical Transcriptions Dataset](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
2. [Turkish Word2Vec](https://github.com/akoksal/Turkish-Word2Vec/wiki/)
3. [Turkish Wikipedia Dumps](https://dumps.wikimedia.org/trwiki/)
