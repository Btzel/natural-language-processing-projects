# Natural Language Processing Projects

A comprehensive implementation of NLP techniques focusing on Named Entity Recognition in medical texts and Word2Vec model analysis.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![spaCy](https://img.shields.io/badge/spaCy-latest-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-latest-red)

## 🎯 Overview

This repository contains two main NLP projects developed for advanced text analysis:
1. Medical Text NER: Identifying medical entities using various ML models
2. Word2Vec Analysis: Comparing different word embedding models across languages

## 🔧 Technical Components

### Project 1: Medical Text NER

#### Data Processing Pipeline
```python
def process_medical_text(text):
    """
    Process medical transcriptions for NER tasks
    Returns preprocessed text with identified entities
    """
```

#### Core Components
1. **Text Preprocessing**
   - Tokenization
   - Lemmatization
   - Named Entity Recognition

2. **Model Implementation**
   - Multinomial Naïve Bayes
   - Random Forest
   - XGBoost
   - LightGBM
   - 1D CNN with LSTM and GRU

### Project 2: Word2Vec Analysis

#### Word Embedding Pipeline
```python
def train_word2vec(corpus):
    """
    Train Word2Vec model on given corpus
    Returns trained model and word embeddings
    """
```

#### Key Features
1. **Model Integration**
   - Custom Turkish Wikipedia model
   - Pre-trained embeddings
   - Multi-language support

2. **Lexicon Analysis**
   - NRC-VAD implementation
   - MTL-Grouped analysis
   - Emotional dimension calculation

## 💻 Implementation Details

### Medical NER System

#### Vector Representations
```python
def create_vectors(text_data):
    """
    Create TF-IDF and Bag-of-Words representations
    """
```

#### Model Training
```python
def train_models(X_train, y_train):
    """
    Train multiple models for comparison
    """
```

### Word2Vec Analysis

#### Embedding Generation
```python
def generate_embeddings(wiki_dump):
    """
    Generate word embeddings from Wikipedia dump
    """
```

## 🛠️ Dependencies

- spacy: NER and linguistic features
- scikit-learn: Machine learning models
- tensorflow: Deep learning implementation
- gensim: Word2Vec models
- pandas: Data manipulation
- numpy: Numerical operations

## 🚀 Usage

1. Install dependencies:
```bash
pip install spacy scikit-learn tensorflow gensim pandas numpy
```

2. Download required models:
```bash
python -m spacy download en_core_sci_md
```

3. Run NER analysis:
```bash
python medical_ner.py
```

4. Run Word2Vec analysis:
```bash
python word2vec_analysis.py
```

## 📊 Results

### Medical NER Performance
- Random Forest: 97% accuracy
- XGBoost: 90% accuracy
- LightGBM: 93% accuracy

### Word2Vec Analysis
- Successfully mapped semantic relationships
- Achieved accurate cross-lingual word associations
- Generated comprehensive emotion lexicons

## 🔍 Technical Details

### NER Implementation
```python
def identify_entities(text):
    """
    Identify medical entities in text using trained models
    """
```

### Word2Vec Implementation
```python
def calculate_lexicon_values(embeddings):
    """
    Calculate emotional dimensions from word embeddings
    """
```

## 🔄 Future Improvements

1. **Model Enhancement**
   - Transformer architecture integration
   - Cross-lingual model support
   - Enhanced emotion detection

2. **Performance Optimization**
   - Parallel processing implementation
   - Memory usage optimization
   - Faster inference times

3. **Feature Addition**
   - Additional language support
   - Real-time processing
   - Advanced visualization tools

## 🤝 Contributing

Contributions are welcome! Key areas:
1. Model optimization
2. Additional language support
3. Documentation enhancement
4. New feature implementation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. Medical Transcriptions Dataset (Kaggle)
2. Turkish Word2Vec Implementation
3. NRC-VAD Lexicon Documentation
4. MTL-Grouped Lexicon Dataset

## 👥 Authors

- Burak TÜZEL
- Talha Alper ASAV

## 🏫 Academic Context

CSE431 – Natural Language Processing with Machine Learning (2023/2024)  
Aydin Adnan Menderes University  
Supervisor: Asst. Prof. Dr. Fatih SOYGAZİ
