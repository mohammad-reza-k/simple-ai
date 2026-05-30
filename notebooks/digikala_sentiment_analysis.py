
import requests
import pandas as pd
import time
import os
import re
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

raw_csv_path = "raw_reviews.csv"

def scrape_digikala_reviews(product_id, max_pages=50):

    reviews_data = []

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for page in range(1, max_pages + 1):

        url = f"https://api.digikala.com/v1/product/{product_id}/comments/?page={page}&mode=newest"

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            break

        try:
            data = response.json()
        except:
            break

        comments = data.get("data", {}).get("comments", [])

        if not comments:
            break

        for c in comments:

            review_text = c.get("body", "").strip()

            rating = c.get("rate")

            if isinstance(rating, dict):
                rating = rating.get("rate")

            review_date = c.get("created_at", None)

            reviews_data.append({
                "review": review_text,
                "rating": rating,
                "date": review_date
            })

        time.sleep(1)

    return pd.DataFrame(reviews_data)

if os.path.exists(raw_csv_path):
    os.remove(raw_csv_path)

df_reviews = scrape_digikala_reviews(
    product_id=19321474,
    max_pages=5
)

df_reviews.to_csv(
    raw_csv_path,
    index=False,
    encoding="utf-8-sig"
)

cleaned_csv_path = "cleaned_reviews.csv"

df = pd.read_csv(raw_csv_path)

df.dropna(inplace=True)

stopwords = set([
    'و', 'در', 'که', 'با', 'از', 'به',
    'این', 'آن', 'را', 'برای',
    'شد', 'می', 'تا', 'یک',
    'بر', 'هم', 'ان', 'های',
    'کن', 'کرد', 'است'
])

def clean_text(text):

    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    words = [w for w in words if w not in stopwords]

    return ' '.join(words)

df["cleaned_text"] = df["review"].apply(clean_text)

df = df[df["cleaned_text"].str.strip() != ""]

def label_sentiment(star):

    if star >= 4:
        return 1

    elif star <= 2:
        return 0

    else:
        return None

df["label"] = df["rating"].apply(label_sentiment)

df.dropna(inplace=True)

df.to_csv(cleaned_csv_path, index=False)

sentiment_counts = df["label"].value_counts()

plt.figure(figsize=(6, 5))

plt.bar(
    sentiment_counts.index.astype(str),
    sentiment_counts.values
)

plt.title("Sentiment Distribution")

plt.xlabel("Sentiment")

plt.ylabel("Count")

plt.show()

all_words = " ".join(df["cleaned_text"]).split()

word_counts = Counter(all_words)

most_common = word_counts.most_common(10)

words = [w[0] for w in most_common]

counts = [w[1] for w in most_common]

plt.figure(figsize=(10, 5))

plt.bar(words, counts)

plt.title("Top 10 Frequent Words")

plt.xlabel("Words")

plt.ylabel("Frequency")

plt.xticks(rotation=30)

plt.show()

X = df["cleaned_text"]

y = df["label"]

vectorizer = TfidfVectorizer(max_features=5000)

X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

lr_model = LogisticRegression(max_iter=2000)

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_predictions)

print("LOGISTIC REGRESSION")

print("Accuracy:", lr_accuracy)

print(classification_report(y_test, lr_predictions))

svm_model = LinearSVC()

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)

print("SVM")

print("Accuracy:", svm_accuracy)

print(classification_report(y_test, svm_predictions))

kmeans_model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

kmeans_model.fit(X_vectorized)

df["cluster"] = kmeans_model.labels_

print(df[["cleaned_text", "cluster"]].head())

cluster_counts = df["cluster"].value_counts()

plt.figure(figsize=(6, 5))

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.title("K-Means Clusters")

plt.xlabel("Cluster")

plt.ylabel("Count")

plt.show()

sample_review = "کیفیت محصول خیلی عالی بود"

sample_cleaned = clean_text(sample_review)

sample_vectorized = vectorizer.transform([sample_cleaned])

prediction = svm_model.predict(sample_vectorized)

print("Review:", sample_review)

if prediction[0] == 1:
    print("Predicted Sentiment: Positive")
else:
    print("Predicted Sentiment: Negative")
