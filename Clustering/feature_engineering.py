import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import re
import pickle


df = pd.read_csv('processed_news_data.csv', encoding='utf-8')

# 1. BASIC TEXT CLEANING (if not already done in your data prep)
def clean_text(text):
    if isinstance(text, str):
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Tokenize
        tokens = word_tokenize(text)
        # Remove short words (optional)
        tokens = [t for t in tokens if len(t) > 2]
        return ' '.join(tokens)
    return ''

# Select which text column to use - combined_text seems most appropriate
# but you can adjust based on your data preparation results
text_column = 'combined_text' 
if text_column in df.columns:
    # Apply cleaning if needed
    df['cleaned_text'] = df[text_column].apply(clean_text)
else:
    # Fall back to processed_content if combined_text doesn't exist
    text_column = 'processed_content'
    df['cleaned_text'] = df[text_column].apply(clean_text)

# Also prepare tokenized version for Word2Vec
df['tokenized_text'] = df['cleaned_text'].apply(lambda x: word_tokenize(x) if isinstance(x, str) else [])

# 2. TF-IDF VECTORIZATION
print("Applying TF-IDF vectorization...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,  # Limit features to top 5000 terms
    min_df=5,           # Ignore terms that appear in less than 5 documents
    max_df=0.85,        # Ignore terms that appear in more than 85% of documents
    stop_words='english'
)

# Fit and transform the data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# Save the vectorizer for future use
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# 3. DIMENSIONALITY REDUCTION
# Option 1: TruncatedSVD (similar to PCA but works on sparse matrices)
print("Applying dimensionality reduction with TruncatedSVD...")
n_components = 100  # Reduce to 100 components
svd = TruncatedSVD(n_components=n_components, random_state=42)
tfidf_svd = svd.fit_transform(tfidf_matrix)
print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.2f}")
print(f"Reduced TF-IDF matrix shape: {tfidf_svd.shape}")

# Save the SVD model and the reduced matrix
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(svd, f)
np.save('tfidf_svd_features.npy', tfidf_svd)

# 4. WORD2VEC EMBEDDINGS (alternative approach)
print("Training Word2Vec model...")
# Get list of tokenized texts, filtering out empty lists
tokenized_texts = [tokens for tokens in df['tokenized_text'] if len(tokens) > 0]

# Train Word2Vec model
w2v_model = Word2Vec(
    sentences=tokenized_texts,
    vector_size=100,  # Embedding dimension
    window=5,         # Context window size
    min_count=2,      # Ignore words with frequency below this
    sg=1,             # Skip-gram model (1) vs CBOW (0)
    workers=4         # Number of threads
)

# Save the Word2Vec model
w2v_model.save('news_word2vec.model')

# Generate document vectors by averaging word vectors
def document_vector(doc, model):
    # Remove out-of-vocabulary words
    doc = [word for word in doc if word in model.wv]
    if len(doc) == 0:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[word] for word in doc], axis=0)

# Create document vectors for each article
print("Generating document vectors from Word2Vec...")
doc_vectors = []
for tokens in df['tokenized_text']:
    # Only process if there are tokens
    if len(tokens) > 0:
        doc_vectors.append(document_vector(tokens, w2v_model))
    else:
        # Handle empty documents
        doc_vectors.append(np.zeros(w2v_model.vector_size))

# Convert to numpy array
w2v_features = np.array(doc_vectors)
print(f"Word2Vec document vectors shape: {w2v_features.shape}")

# Save the Word2Vec document vectors
np.save('w2v_features.npy', w2v_features)

# 5. COMBINING WITH METADATA (optional)
# You might want to include categorical features like category
# Convert category to one-hot encoding
if 'category' in df.columns:
    print("Adding category one-hot encoding...")
    category_dummies = pd.get_dummies(df['category'], prefix='cat')
    
    # Scale the one-hot features to be in similar range as TF-IDF features
    category_features = category_dummies.values
    
    # You could save these separately or combine with text features
    np.save('category_features.npy', category_features)

# 6. SAVE PROCESSED DATAFRAME
# Save the processed DataFrame with added columns
df.to_csv('feature_engineered_news.csv', index=False)

print("Feature engineering complete!")