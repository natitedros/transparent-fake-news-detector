# Fake News Detection

A machine learning pipeline that classifies news articles as **Fake** or **True** using NLP techniques and explains its prediction, trained on ~45,000 labeled articles.

---

## Project Structure

```
├── Datasets/               # Raw and processed data
│   ├── Fake.csv
│   ├── True.csv
│   ├── train.csv           # Cleaned, stratified training split
│   └── test.csv            # Cleaned, stratified test split
├── models/                 # Saved model artifacts
│   ├── best_vectorizer.pkl # Fitted TF-IDF vectorizer
│   ├── best_model.pkl      # Best trained model (Logistic Regression)
│   └── best_config.json    # Best feature/hyperparameter config
├── plots/                  # EDA and evaluation visualizations
├── task_1_preprocessing.ipynb
├── task_2_feature_selection.ipynb
├── task_3_model_training.ipynb
└── task_4_RAG_explainability.ipynb
```

---

## Pipeline Overview

### Notebook 1 — Data Preprocessing

- Loads `Fake.csv` (23,481 samples) and `True.csv` (21,417 samples), assigns labels, and concatenates into a single dataset of 44,898 articles
- Explores feature distributions (title length, text length, subject categories, class balance)
- Cleans text via lowercasing, URL/handle/hashtag removal, stopword removal, lemmatization, and photo credit stripping
- Performs a stratified 75/25 train/test split (33,043 train / 11,015 test) and exports cleaned CSVs

### Notebook 2 — Feature Selection

Compares two feature extraction strategies on a Logistic Regression baseline:

| Approach | Config     | Accuracy | F1     |
| -------- | ---------- | -------- | ------ |
| TF-IDF   | body-only  | 0.9953   | 0.9951 |
| TF-IDF   | title+body | 0.9939   | 0.9937 |
| Word2Vec | title+body | 0.9757   | 0.9749 |
| Word2Vec | body-only  | 0.9680   | 0.9670 |

**Best config:** TF-IDF with unigrams+bigrams (`ngram_range=[1,2]`), `max_features=50000`, `C=10.0`, using the combined title+body content field. Saved to `models/`.

### Notebook 3 — Model Training

Trains and compares three models on the TF-IDF feature matrix (50,000 features). SVM and MLP use TruncatedSVD (→ 300 dimensions) to reduce training time.

| Model                   | Accuracy   | Precision  | Recall     | F1         | ROC-AUC    |
| ----------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **Logistic Regression** | **0.9939** | **0.9928** | **0.9945** | **0.9937** | **0.9996** |
| SVM (RBF)               | 0.9923     | 0.9908     | 0.9932     | 0.9920     | 0.9992     |
| MLP (PyTorch)           | 0.9897     | 0.9915     | 0.9870     | 0.9892     | 0.9989     |

**Best model:** Logistic Regression (F1: 0.9937). Saved to `models/best_model.pkl`.

### Notebook 4 — RAG Explainability

Implements a Retrieval-Augmented Generation (RAG) pipeline to explain model predictions in natural language:

1. Vectorizes the input article with TF-IDF
2. Retrieves the top-3 similar training samples via a FAISS index (50,000-dimensional flat index over 33,043 vectors)
3. Extracts top TF-IDF key phrases from the article
4. Constructs a prompt and generates a human-readable explanation using `google/flan-t5-large` (HuggingFace, run locally)

---

## Key Dependencies

- `pandas`, `numpy`, `scikit-learn` — data processing and ML models
- `nltk` — text cleaning and lemmatization
- `gensim` — Word2Vec embeddings
- `faiss-cpu` — similarity search index
- `torch` — MLP model (PyTorch)
- `transformers` — Flan-T5 explainability model
