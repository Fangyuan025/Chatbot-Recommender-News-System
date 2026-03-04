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
from sklearn.metrics import accuracy_score, classification_report
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split as surprise_split
import hashlib
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Load JSON Data
def load_data(file_path, sample_size=50000):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    print(f"Loaded dataset with {df.shape[0]} records")

    df.dropna(subset=['headline', 'short_description', 'category'], inplace=True)
    print(f"After dropping NAs: {df.shape[0]} records")

    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Using a subset of {sample_size} records")

    top_categories = df['category'].value_counts().head(15).index
    df = df[df['category'].isin(top_categories)]
    print(f"Filtered dataset to top 15 categories: {df.shape[0]} records")

    return df


# Preprocess Data
def preprocess_data(df):
    df['text'] = (df['headline'] + ' ' + df['short_description']).str.lower()
    df['text'] = df['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

    # Generate unique IDs
    df['news_id'] = df['headline'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10 ** 6))

    # Simulate user preferences
    np.random.seed(42)
    random.seed(42)

    # Create user preference database
    user_base_size = 10000
    user_prefs = {
        uid: np.random.choice(
            df['category'].unique(),
            size=np.random.randint(1, 4),
            replace=False
        )
        for uid in range(1, user_base_size + 1)
    }

    # Assign users with preference bias
    df['user_id'] = np.random.randint(1, user_base_size + 1, size=len(df))

    # Generate meaningful ratings
    def calculate_rating(row):
        base = 3.0
        if row['category'] in user_prefs[row['user_id']]:
            return min(5.0, base + np.random.choice([1.2, 1.5, 0.8]))
        else:
            return max(1.0, base - np.random.choice([1.0, 0.7, 1.3]))

    df['rating'] = df.apply(calculate_rating, axis=1)
    df['rating'] = df['rating'].apply(lambda x: np.clip(x + np.random.normal(0, 0.3), 1.0, 5.0))
    df['rating'] = df['rating'].round().astype(int)

    print("\nSimulated rating distribution:")
    print(df['rating'].value_counts())

    print("\nTop categories with average ratings:")
    print(df.groupby('category')['rating'].mean().sort_values(ascending=False).head(10))

    return df


# Convert text data to numerical features
def extract_features(df):
    print("\nExtracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=3000,
        ngram_range=(1, 2),  # Added bigrams
        min_df=5,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(df['text'])
    print(f"Feature matrix shape: {X.shape}")
    return X, vectorizer


# Train Classification Model
def train_classification_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Add confusion matrix visualization
    def plot_confusion_matrix(y_true, y_pred, classes, model_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(C=0.5, max_iter=5000, solver='saga', n_jobs=-1,
                                                  class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=25,
                                                class_weight='balanced', n_jobs=-1),
        'Linear SVM': LinearSVC(C=0.5, max_iter=1000, dual=False, class_weight='balanced')
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        X_train_curr = X_train_scaled if name in ['Logistic Regression', 'Linear SVM'] else X_train
        X_test_curr = X_test_scaled if name in ['Logistic Regression', 'Linear SVM'] else X_test

        start_time = time.time()
        model.fit(X_train_curr, y_train)
        y_pred = model.predict(X_test_curr)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        # Plot confusion matrix for best model
        if name == 'Logistic Regression':
            class_names = label_encoder.classes_
            plot_confusion_matrix(y_test, y_pred, class_names, name)

        results[name] = {'model': model, 'accuracy': accuracy}
        print(f"Training time: {time.time() - start_time:.2f} seconds")

    return results


# Collaborative Filtering using Surprise Library
def train_recommender(df):
    if not {'user_id', 'news_id', 'rating'}.issubset(df.columns):
        print("\nSkipping recommender system - Required columns missing")
        return None

    print("\nTraining recommender system...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'news_id', 'rating']], reader)
    trainset, testset = surprise_split(data, test_size=0.2)

    # Improved model parameters
    algo = SVD(n_factors=100, n_epochs=25, lr_all=0.007, reg_all=0.03)
    algo.fit(trainset)

    results = cross_validate(algo, data, measures=['rmse', 'mae'], cv=3, verbose=True)
    print("Recommender training complete.")

    # Plot rating distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rating', data=df)
    plt.title('Rating Distribution')
    plt.show()

    return algo


if __name__ == "__main__":
    file_path = r"D:\Collage\DS\FINAL\Classification\news_dataset.json"
    df = load_data(file_path, sample_size=50000)
    df = preprocess_data(df)

    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])
    category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("\nCategory mapping:", category_mapping)

    X, vectorizer = extract_features(df)
    models = train_classification_models(X, df['category_encoded'])

    best_model = max(models.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.4f}")

    recommender = train_recommender(df)