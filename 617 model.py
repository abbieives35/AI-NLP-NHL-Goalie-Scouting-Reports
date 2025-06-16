import pandas as pd 
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Download stopwords if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Add custom domain-specific stopwords
custom_stop_words = {'goalie', 'goaltender', 'net', 'he', 'hes', 'his', 'puck', 'goal', 'saves', 'good', 'well', 'game', 'gets', 'plays', 'vasilevski','gibson', 'gillies', 'shots', 'guy', 'john', 'however', 'need' }
stop_words = set(stopwords.words('english')).union(custom_stop_words)

# === 1. Load your CSV ===
df = pd.read_csv(r"\\ISCFS01\RedirectedFolders\aives\Downloads\617 project.csv", encoding='latin1')
df.columns = df.columns.str.strip()

# === 2. Define success categories with detailed logic ===
def classify_success(row):
    games = row['NHL Games Played']
    year = row['Draft Year']
    if games >= 500:
        return 'Elite'
    elif (year > 2015 and games >= 50) or (year <= 2015 and games >= 100):
        return 'Successful'
    elif games <= 20:
        return 'Unsuccessful'
    else:
        return 'Average'

df['SuccessCategory'] = df.apply(classify_success, axis=1)

# Create a binary success flag for classification (Elite + Successful = 1, else 0)
df['Success'] = df['SuccessCategory'].apply(lambda x: 1 if x in ['Elite', 'Successful'] else 0)

print("Success Category Breakdown:\n", df['SuccessCategory'].value_counts())

# === 3. Clean scouting report text ===
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['cleaned_report'] = df['Pre-Draft Scouting Report'].apply(clean_text)

# === 4. Vectorize with TF-IDF ===
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_report'])
y = df['Success']

# === 5. Train/test split and classifier ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# === 6. Analyze most common words for each SuccessCategory group ===
print("\nTop 10 words in reports by SuccessCategory:")

for category in ['Elite', 'Successful', 'Average', 'Unsuccessful']:
    texts = " ".join(df[df['SuccessCategory'] == category]['cleaned_report']).split()
    counts = Counter(texts)
    print(f"\n{category}:")
    print(counts.most_common(10))

# === 7. Sentence Embeddings and Clustering ===
model = SentenceTransformer('all-MiniLM-L6-v2')
df['embedding'] = df['Pre-Draft Scouting Report'].fillna("").apply(lambda x: model.encode(x))

# Convert list of embeddings to numpy array
X_emb = np.vstack(df['embedding'])

# KMeans clustering to show similarity
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_emb)
df['cluster'] = kmeans.labels_

# === Sample reports from each cluster ===
print("\n=== Sample Reports from Each Cluster ===")
for i in range(kmeans.n_clusters):
    print(f"\n--- Cluster {i} ---")
    sample_texts = df[df['cluster'] == i]['Pre-Draft Scouting Report'].dropna().sample(3, random_state=42)
    for idx, text in enumerate(sample_texts, 1):
        print(f"{idx}. {text[:300]}...")  # Print first 300 characters

print("\nCluster breakdown by success:")
print(df.groupby(['cluster', 'Success']).size())

# Optional: Plot similarity heatmap
sim_matrix = cosine_similarity(X_emb)
sns.heatmap(sim_matrix)
plt.title("Scouting Report Semantic Similarity Heatmap")
plt.show()

# === 8. Plot usage of a sample word over draft years ===
def word_over_time(word, df):
    year_counts = {}
    if 'Draft Year' not in df.columns:
        return year_counts
    for year in sorted(df['Draft Year'].dropna().unique()):
        reports = df[df['Draft Year'] == year]['cleaned_report']
        count = " ".join(reports).split().count(word)
        year_counts[year] = count
    return year_counts

sample_words = ['athletic', 'quick', 'positioning', 'reflexes']

plt.figure(figsize=(10,6))
for word in sample_words:
    counts = word_over_time(word, df)
    if counts:
        plt.plot(list(counts.keys()), list(counts.values()), label=word)

plt.title("Scouting Report Word Frequency Over Draft Years")
plt.xlabel("Draft Year")
plt.ylabel("Count of Word in Reports")
plt.legend()
plt.show()
