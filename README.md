# Explainable Fake News Detection

A machine learning pipeline that classifies news articles as **Fake** or **True** using NLP techniques and explains its predictions, trained on ~45,000 labeled articles.

---

## Project Overview

This project builds an end-to-end explainable fake news detection system using classical NLP and machine learning. The pipeline ingests raw labeled news articles, cleans and transforms them into numerical features, trains and benchmarks multiple classifiers, and finally wraps the best model in a Retrieval-Augmented Generation (RAG) layer that produces human-readable explanations for each prediction.

The dataset combines two sources, `Fake.csv` and `True.csv`, totaling 44,898 articles across a balanced binary classification task. After preprocessing and a stratified train/test split, a TF-IDF vectorizer with unigram and bigram features (50,000 dimensions) is used as the primary feature representation. Three classifiers are evaluated: Logistic Regression, SVM with RBF kernel, and a PyTorch MLP. Logistic Regression achieves the best performance (F1: 0.9937, ROC-AUC: 0.9996) and is saved as the production model.

Explainability is added via a FAISS similarity index over the training set, which retrieves the top-3 nearest neighbors for any input article. These, combined with the article's top TF-IDF key phrases, are passed as context to `google/flan-t5-large` to generate a natural language explanation of why the model classified the article as fake or true.

---

## Project Structure

```
├── Dataset/               # Raw and processed data
│   ├── Fake.csv
│   ├── True.csv
│   ├── train.csv           # Cleaned, stratified training split
│   └── test.csv            # Cleaned, stratified test split
├── documents/
│   ├── project_report      # IEEE-style 7-page detailed report
│   ├── project_proposal
│   └── presentation_slides
├── models/                 # Saved model artifacts
│   ├── best_vectorizer.pkl # Fitted TF-IDF vectorizer
│   ├── best_config.json    # Best feature/hyperparameter config
│   ├── best_model.pkl      # Best trained model (Logistic Regression)
│   └── best_model_meta.pkl # Best trained model name and it's hyperparameter values
├── plots/                  # EDA and evaluation visualizations
├── task_1_preprocessing.ipynb
├── task_2_feature_selection.ipynb
├── task_3_model_training.ipynb
└── task_4_RAG_explainability.ipynb
```

---

## Implementation Pipeline

The project is organized as four sequential notebooks, each building on the outputs of the previous.

### `task_1_preprocessing.ipynb` — Data Preprocessing

Loads `Fake.csv` (23,481 samples) and `True.csv` (21,417 samples), assigns binary labels, and concatenates them into a single dataset of 44,898 articles. Performs exploratory data analysis on feature distributions (title length, text length, subject categories, class balance), then cleans text via lowercasing, URL/handle/hashtag removal, stopword removal, lemmatization, and photo credit stripping. Outputs a stratified 75/25 train/test split (33,043 train / 11,015 test) saved as `Datasets/train.csv` and `Datasets/test.csv`.

### `task_2_feature_selection.ipynb` — Feature Selection

Compares TF-IDF and Word2Vec feature extraction strategies on a Logistic Regression baseline, evaluating body-only vs. title+body content fields.

| Approach | Config     | Accuracy | F1     |
| -------- | ---------- | -------- | ------ |
| TF-IDF   | body-only  | 0.9953   | 0.9951 |
| TF-IDF   | title+body | 0.9939   | 0.9937 |
| Word2Vec | title+body | 0.9757   | 0.9749 |
| Word2Vec | body-only  | 0.9680   | 0.9670 |

The best configuration — TF-IDF with unigrams+bigrams (`ngram_range=[1,2]`), `max_features=50000`, `C=10.0`, on the combined title+body field — is saved to `models/best_vectorizer.pkl` and `models/best_config.json`.

### `task_3_model_training.ipynb` — Model Training

Trains and evaluates three classifiers on the 50,000-dimensional TF-IDF feature matrix. SVM and MLP use TruncatedSVD (→ 300 dimensions) to reduce training time.

| Model                   | Accuracy   | Precision  | Recall     | F1         | ROC-AUC    |
| ----------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **Logistic Regression** | **0.9939** | **0.9928** | **0.9945** | **0.9937** | **0.9996** |
| SVM (RBF)               | 0.9923     | 0.9908     | 0.9932     | 0.9920     | 0.9992     |
| MLP (PyTorch)           | 0.9897     | 0.9915     | 0.9870     | 0.9892     | 0.9989     |

Logistic Regression wins across all metrics and is saved to `models/best_model.pkl`.

### `task_4_RAG_explainability.ipynb` — RAG Explainability

Wraps the trained classifier in a Retrieval-Augmented Generation pipeline that explains predictions in natural language:

1. Vectorizes the input article using the saved TF-IDF vectorizer
2. Retrieves the top-3 most similar training samples via a FAISS flat index (50,000-dimensional, 33,043 vectors)
3. Extracts the article's top TF-IDF key phrases as additional context
4. Constructs a prompt from the retrieved neighbors and key phrases, then generates a human-readable explanation using `google/flan-t5-large` (HuggingFace, run locally)

---

## How to Run

### 1. Clone the repository

> **Note:**: `git lfs` must be installed in the local machine for cloning the large dataset files in the folder `Dataset/`. Use the command `git lfs install` to install the library before cloning.

```bash
git lfs install
git clone https://github.com/your-username/explainable-fake-news-detection.git
cd explainable-fake-news-detection
```

### 2. Set up a virtual environment

```bash
python -m venv venv             # Or use conda environment commands
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

Install the core packages manually:

```bash
pip install pandas numpy scikit-learn nltk gensim faiss-cpu torch transformers jupyter
```

Then download the required NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4. Add the raw datasets

The starting dataset files are included in the repository and will load correctly given `git lfs` is installed.

If the cloning step fails due to the size of the dataset exceeding 100MB, place `Fake.csv` and `True.csv` manually by downloading from [this link](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/) into the `Dataset/` directory before running the notebooks.

### 5. Run the notebooks in order

Launch Jupyter and execute the notebooks sequentially:

```bash
jupyter notebook
```

| Step | Notebook                          | Outputs                                                 |
| ---- | --------------------------------- | ------------------------------------------------------- |
| 1    | `task_1_preprocessing.ipynb`      | `Dataset/train.csv`, `Dataset/test.csv`                 |
| 2    | `task_2_feature_selection.ipynb`  | `models/best_vectorizer.pkl`, `models/best_config.json` |
| 3    | `task_3_model_training.ipynb`     | `models/best_model.pkl`, `models/best_model_meta.json`  |
| 4    | `task_4_RAG_explainability.ipynb` | Prediction + natural language explanation               |

> **Note:** Notebook 4 downloads `google/flan-t5-large` (~3 GB) from HuggingFace on first run. Please ensure you have sufficient disk space and a stable internet connection.

---

## Key Dependencies

- `pandas`, `numpy`, `scikit-learn` — data processing and ML models
- `nltk` — text cleaning and lemmatization
- `gensim` — Word2Vec embeddings
- `faiss-cpu` — similarity search index
- `torch` — MLP model (PyTorch)
- `transformers` — Flan-T5 explainability model
