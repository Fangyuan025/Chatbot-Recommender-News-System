import pandas as pd
import json
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split as surprise_split
import hashlib
import random
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE


def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def load_data(file_path, sample_size=50000):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    news_df = pd.DataFrame(data)
    print(f"Loaded dataset with {news_df.shape[0]} records")

    news_df.dropna(subset=['headline', 'short_description', 'category'], inplace=True)
    print(f"After dropping NAs: {news_df.shape[0]} records")

    if sample_size:
        news_df = news_df.sample(n=sample_size, random_state=42)
        print(f"Using a subset of {sample_size} records")

    top_categories = news_df['category'].value_counts().head(15).index
    return news_df[news_df['category'].isin(top_categories)]


def preprocess_data(news_df):
    news_df['text'] = (news_df['headline'] + ' ' + news_df['short_description']).str.lower()
    news_df['text'] = news_df['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

    # Generate IDs
    news_df['news_id'] = news_df['headline'].apply(
        lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10 ** 6)
    )

    # Simulate user preferences
    np.random.seed(42)
    user_base_size = 10000
    user_prefs = {
        uid: np.random.choice(news_df['category'].unique(),
                              size=np.random.randint(1, 4),
                              replace=False
                              )
        for uid in range(1, user_base_size + 1)
    }

    news_df['user_id'] = np.random.randint(1, user_base_size + 1, size=len(news_df))

    # Optimized rating calculation
    news_df['rating'] = news_df.apply(lambda row:
                                      4 if row['category'] in user_prefs[row['user_id']] else 2, axis=1
                                      )
    news_df['rating'] += np.random.randint(-1, 2, size=len(news_df))
    news_df['rating'] = np.clip(news_df['rating'], 1, 5)

    return news_df


def extract_features(news_df):
    print("\nExtracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,  # Reduced from 3000
        ngram_range=(1, 1),  # Removed bigrams
        min_df=10,  # Increased from 5
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(news_df['text'])
    print(f"Feature matrix shape: {X.shape}")
    return X, vectorizer


def train_classification_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Convert sparse matrices to dense arrays
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    # Get class distribution
    class_counts = np.bincount(y_train)
    majority_class_size = np.max(class_counts)

    # Create sampling strategy dictionary for 50% of majority class size
    sampling_strategy = {
        class_idx: max(int(majority_class_size * 0.5), count)
        for class_idx, count in enumerate(class_counts)
        if count < majority_class_size
    }

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_res, y_res = smote.fit_resample(X_train_dense, y_train)

    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=200, solver='lbfgs', n_jobs=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=20, n_jobs=-1
        ),
        'Linear SVM': LinearSVC(C=0.5, max_iter=1000, dual=False)
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()

        if name == 'Logistic Regression':
            scaler = StandardScaler()
            X_res_scaled = scaler.fit_transform(X_res)
            X_test_scaled = scaler.transform(X_test_dense)
            model.fit(X_res_scaled, y_res)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_res, y_res)
            y_pred = model.predict(X_test_dense)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")
        results[name] = accuracy

    return results


def train_recommender(news_df):
    if not {'user_id', 'news_id', 'rating'}.issubset(news_df.columns):
        return None

    print("\nTraining recommender...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(news_df[['user_id', 'news_id', 'rating']], reader)
    algo = SVD(n_factors=20, n_epochs=10, lr_all=0.01)  # Simplified model
    return cross_validate(algo, data, measures=['rmse'], cv=2, verbose=True)


if __name__ == "__main__":
    file_path = r"D:\Collage\DS\FINAL\Classification\news_dataset.json"

    # Data pipeline
    news_data = load_data(file_path, sample_size=50000)
    processed_data = preprocess_data(news_data)

    # Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(processed_data['category'])

    # Feature extraction
    X, vectorizer = extract_features(processed_data)

    # Classification
    model_results = train_classification_models(X, y)
    print("\nFinal Results:")
    for model, acc in model_results.items():
        print(f"{model}: {acc:.4f}")

    # Recommendation
    recommender_results = train_recommender(processed_data)