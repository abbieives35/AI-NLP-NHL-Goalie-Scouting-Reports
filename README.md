# AI-NLP-NHL-Goalie-Scouting-Reports

# 🧊 Evaluating NHL Goalie Scouting Reports Using NLP and Machine Learning

This project applies Natural Language Processing (NLP), machine learning, and clustering techniques to evaluate the effectiveness of NHL pre-draft scouting reports on goaltenders. The goal: to determine whether the language used in these reports actually correlates with long-term goaltending success.

> 🔬 **Major Insight**: The results strongly suggest that current scouting reports fail to differentiate between successful and unsuccessful goalies—indicating a systemic flaw in how goaltending talent is evaluated.

---

## 📌 Project Overview

- **Dataset**: Pre-draft scouting reports for NHL goalies drafted between 2010–2020 (sourced from The Hockey Writers).
- **Goal**: Test whether language in these reports predicts career success.
- **Success Metrics**: NHL games played, save percentage; grouped into "Elite", "Successful", "Average", or "Unsuccessful".
- **Approach**:
  - Cleaned and preprocessed text data
  - Created binary success labels
  - Applied TF-IDF and trained a Logistic Regression classifier
  - Performed lexical analysis and semantic clustering using SentenceTransformers and K-Means

---

## 🛠️ Tools & Technologies

- **Languages**: Python
- **Libraries**: `pandas`, `nltk`, `scikit-learn`, `sentence-transformers`, `matplotlib`, `seaborn`
- **Models & Techniques**:
  - TF-IDF vectorization
  - Logistic Regression classifier
  - Sentence Embedding (`all-MiniLM-L6-v2`)
  - K-Means clustering
- **Visuals**:
  - Cosine similarity heatmaps
  - Word usage trends across draft years

---

## 📈 Key Findings

- **Model Performance**: The binary classification model achieved only **39% accuracy** with poor precision and recall—indicating the text does **not** meaningfully predict success.
- **Lexical Analysis**: Common traits like "quick", "calm", and "athletic" appeared in reports for both successful and unsuccessful goalies.
- **Clustering Results**: K-Means clusters showed **no meaningful separation** based on future success.
- **Semantic Similarity**: Reports were semantically similar regardless of outcome—pointing to a lack of depth and differentiation in scouting language.

---

## 🔍 Project Files

```bash
.
├── notebooks/                    # Jupyter Notebooks with modeling & visualizations
├── docs/                         # Full implementation paper (PDF)
├── presentation/                 # PowerPoint slide deck
└── README.md
