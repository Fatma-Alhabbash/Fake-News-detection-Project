# Fake News Detection Using Machine Learning

> An NLP-based machine learning system to classify Arabic news articles as **real** or **fake**, achieving up to **92.62% accuracy**.

---

## Team

| Name | Role |
|------|------|
| Roaa Alhaddad | Team Member |
| Saja Abdalal | Team Member |
| Fatma Alzahraa Alhabbash | Team Member |

**Supervisor:** Dr. Tareq ALTALMAS  
**Institution:** University College of Applied Sciences (UCAS)  
**Date:** July 2025

---

## Overview

The spread of misinformation in Arabic media poses serious challenges to public trust. This project applies Natural Language Processing (NLP) and machine learning to automatically classify Arabic news articles as **fake** or **real**.

Three preprocessing pipelines were developed and compared:

| Pipeline | Stemmer / Lemmatizer | Accuracy |
|----------|----------------------|----------|
| Pipeline 1 | ISRI Stemmer (NLTK) | **92.62%** (Best) |
| Pipeline 2 | Spark NLP Arabic Lemmatizer | 88.50% |
| Pipeline 3 | Light Stemmer (custom) | — |

All pipelines use **TF-IDF** feature extraction and **Logistic Regression** classification.

---

## 📂 Repository Structure

```
Fake-News-detection-Project/
│
├── Fake_News_Detectin_Project.ipynb   # Main Jupyter notebook
├── merged_cleaned.csv                 # Dataset (loaded from GitHub)
├── arabic_stopwords.txt               # Arabic stopword list
└── README.md
```

---

## Dataset

- **Source:** Loaded directly from this repository
- **Size:** 5,352 news articles
- **Language:** Arabic
- **Time span:** 2011 – 2025 (majority from 2023–2025)

### Fields

| Column | Description |
|--------|-------------|
| `Id` | Unique article identifier |
| `date` | Publication date |
| `platform` | News source (e.g., Aljazeera, Misbar) |
| `title` | News headline |
| `News content` | Full article body |
| `Label` | `real` or `fake` |

### Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Real | 3,913 | 73.1% |
| Fake | 1,439 | 26.9% |

> **Note:** The dataset is imbalanced (3:1 real-to-fake ratio). This may reduce recall for fake news detection.

**Top platforms:** Aljazeera (64%), followed by Misbar, Tibyan, and others.

---

## Methodology

### 1. Data Preprocessing

- **Null check:** No missing values found
- **Field merging:** `title` + `News content` → `content`
- **Text cleaning** (via `tnkeeh` + regex):
  - Removed URLs, Twitter links, hashtags, mentions
  - Removed English words and special characters
  - Normalized Arabic digits and text
- **Stopwords removal:** Based on [Mohataher's Arabic stopword list](https://github.com/mohataher/arabic-stop-words)
- **Text normalization:** ISRI Stemmer / Spark NLP Lemmatizer / Light Stemmer
- **Platform appended** to content for source-aware prediction

### 2. Feature Extraction

- **TF-IDF Vectorization** on the processed `content` column

### 3. Model

- **Logistic Regression** — chosen for its efficiency with high-dimensional sparse TF-IDF vectors and strong interpretability

---

## Pipeline Flowchart

```
Load Dataset (merged_cleaned.csv)
        │
        ▼
   EDA (Label / Platform / Temporal Distribution)
        │
        ▼
   Merge title + News content → content
        │
        ▼
   Clean text (URLs, English words, special chars)
        │
        ▼
   Remove Arabic Stopwords
        │
        ▼
   Choose Preprocessing:
   ├── ISRI Stemmer
   ├── Spark NLP Lemmatizer
   └── Light Stemmer
        │
        ▼
   Append platform to content
        │
        ▼
   TF-IDF Vectorization
        │
        ▼
   Label Encoding: fake → 1, real → 0
        │
        ▼
   Train / Test Split
        │
        ▼
   Logistic Regression
        │
        ▼
   Evaluate (Accuracy, Confusion Matrix, Classification Report)
```

---

## Results (Best Pipeline: ISRI Stemmer + Logistic Regression)

**Accuracy: 92.62%**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real (0) | 0.96 | 0.93 | 0.95 | 783 |
| Fake (1) | 0.83 | 0.91 | 0.87 | 288 |
| **Weighted Avg** | **0.93** | **0.93** | **0.93** | 1071 |

**Confusion Matrix:**
```
[[731  52]
 [ 27 261]]
```

---

## Getting Started

### Prerequisites

```bash
pip install tnkeeh spark-nlp farasapy joblib swifter nltk light_stem
pip install pandas scikit-learn matplotlib seaborn
```

### Run the Notebook

The notebook is divided into two sections:

1. **Section 1 — Explanation:** Step-by-step walkthrough of each preprocessing stage *(do not run — for understanding only)*
2. **Section 2 — Pipelines:** The main training and evaluation pipelines *(run these)*

```bash
jupyter notebook Fake_News_Detectin_Project.ipynb
```

> The dataset is loaded automatically from GitHub — no manual download needed.

---

## Future Work

1. **Address class imbalance** using SMOTE, oversampling, or class-weighted loss
2. **Explore transformer models** like AraBERT for richer semantic understanding
3. **Add metadata features** such as source credibility scores or sentiment analysis
4. **Expand the dataset** with more diverse sources and topics
5. **Hybrid preprocessing** combining stemming and lemmatization approaches

---

## 📚 References

- [Mohataher Arabic Stopwords](https://github.com/mohataher/arabic-stop-words)
- [NLTK ISRI Stemmer](https://www.nltk.org/api/nltk.stem.isri.html)
- [Spark NLP](https://nlp.johnsnowlabs.com/)
- [tnkeeh — Arabic Text Normalization](https://github.com/ARBML/tnkeeh)
