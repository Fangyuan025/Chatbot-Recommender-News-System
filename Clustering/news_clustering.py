import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import pickle
from wordcloud import WordCloud
from collections import Counter
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class NewsClusteringModel:
    def __init__(self, features_path='tfidf_svd_features.npy', 
                data_path='feature_engineered_news.csv',
                vectorizer_path='tfidf_vectorizer.pkl'):
        """
        Initialize the clustering model
        
        Parameters:
        -----------
        features_path : str
            Path to the features file (TF-IDF features or Word2Vec features)
        data_path : str
            Path to the original data CSV file
        vectorizer_path : str
            Path to the TF-IDF vectorizer
        """
        self.features_path = features_path
        self.data_path = data_path
        self.vectorizer_path = vectorizer_path
        
        # Create output directory
        os.makedirs('clustering_results', exist_ok=True)
        
        # Load data and features
        self._load_data()
        
    def _load_data(self):
        """Load data and features"""
        print("Loading data and features...")
        
        # Load feature matrix
        self.features = np.load(self.features_path)
        print(f"Feature matrix shape: {self.features.shape}")
        
        # Load original data
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        
        # Load TF-IDF vectorizer (for keyword extraction later)
        try:
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("TF-IDF vectorizer loaded successfully")
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
            self.vectorizer = None
    
    def kmeans_clustering(self, n_clusters=15, random_state=42):
        """
        Perform K-Means clustering
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        random_state : int
            Random seed
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with cluster labels
        """
        print(f"\nPerforming K-Means clustering (k={n_clusters})...")
        
        # Initialize and train K-Means model
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.kmeans_labels = self.kmeans.fit_predict(self.features)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(self.features, self.kmeans_labels)
        print(f"K-Means clustering silhouette score: {silhouette_avg:.4f}")
        
        # Add labels to DataFrame
        self.df['kmeans_cluster'] = self.kmeans_labels
        
        # Count cluster sizes
        cluster_sizes = self.df['kmeans_cluster'].value_counts().sort_index()
        print("\nCluster sizes:")
        for cluster, size in cluster_sizes.items():
            print(f"  Cluster {cluster}: {size} documents")
        
        # Save clustering results
        self.df.to_csv('clustering_results/kmeans_clustered_data.csv', index=False)
        
        # Save model
        with open('clustering_results/kmeans_model.pkl', 'wb') as f:
            pickle.dump(self.kmeans, f)
        
        return self.df
    
    def dbscan_clustering(self, eps=0.5, min_samples=5):
        """
        Perform DBSCAN density clustering
        
        Parameters:
        -----------
        eps : float
            Neighborhood radius
        min_samples : int
            Minimum number of samples required to be a core point
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with cluster labels
        """
        print(f"\nPerforming DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
        
        # Initialize and train DBSCAN model
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.dbscan_labels = self.dbscan.fit_predict(self.features)
        
        # Count number of clusters and noise points
        n_clusters = len(set(self.dbscan_labels)) - (1 if -1 in self.dbscan_labels else 0)
        n_noise = list(self.dbscan_labels).count(-1)
        
        print(f"DBSCAN number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise} ({n_noise/len(self.dbscan_labels):.2%})")
        
        # Calculate silhouette score if there's more than one cluster
        if n_clusters > 1:
            # Only calculate for non-noise points
            mask = self.dbscan_labels != -1
            if sum(mask) > n_clusters:  # Ensure there are enough samples
                silhouette_avg = silhouette_score(
                    self.features[mask], 
                    self.dbscan_labels[mask]
                )
                print(f"DBSCAN clustering silhouette score (excluding noise points): {silhouette_avg:.4f}")
        
        # Add labels to DataFrame
        self.df['dbscan_cluster'] = self.dbscan_labels
        
        # Count cluster sizes
        cluster_sizes = self.df['dbscan_cluster'].value_counts().sort_index()
        print("\nCluster sizes:")
        for cluster, size in cluster_sizes.items():
            label = "Noise points" if cluster == -1 else f"Cluster {cluster}"
            print(f"  {label}: {size} documents")
        
        # Save clustering results
        self.df.to_csv('clustering_results/dbscan_clustered_data.csv', index=False)
        
        return self.df
    
    def topic_modeling(self, n_topics=15, random_state=42):
        """
        Perform topic modeling using NMF
        
        Parameters:
        -----------
        n_topics : int
            Number of topics
        random_state : int
            Random seed
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with topic labels
        """
        print(f"\nPerforming topic modeling (topics={n_topics})...")
        
        # Check if the features contain negative values
        has_negative = np.any(self.features < 0)
        
        if has_negative:
            print("Input features contain negative values. Applying transformation to make them non-negative...")
            
            # Option 1: Shift data to make all values non-negative
            min_val = np.min(self.features)
            if min_val < 0:
                # Add the absolute minimum value to make all values non-negative
                features_non_negative = self.features - min_val
                print(f"Shifted all values by {-min_val} to ensure non-negativity")
            else:
                features_non_negative = self.features
        else:
            features_non_negative = self.features
        
        # Initialize and train NMF model
        print("Using NMF for topic modeling...")
        self.nmf = NMF(n_components=n_topics, random_state=random_state)
        self.topic_distributions = self.nmf.fit_transform(features_non_negative)
        
        # Assign topics to documents (take the topic with highest value)
        self.topic_labels = np.argmax(self.topic_distributions, axis=1)
        
        # Add labels to DataFrame
        self.df['nmf_topic'] = self.topic_labels
        
        # Count topic sizes
        topic_sizes = self.df['nmf_topic'].value_counts().sort_index()
        print("\nTopic sizes:")
        for topic, size in topic_sizes.items():
            print(f"  Topic {topic}: {size} documents")
        
        # Save topic modeling results
        self.df.to_csv('clustering_results/nmf_topic_data.csv', index=False)
        
        # Save model
        with open('clustering_results/nmf_model.pkl', 'wb') as f:
            pickle.dump(self.nmf, f)
        
        return self.df
    
    def extract_cluster_keywords(self, cluster_column='kmeans_cluster', n_keywords=10):
        """
        Extract keywords for each cluster
        
        Parameters:
        -----------
        cluster_column : str
            Name of the column containing cluster labels
        n_keywords : int
            Number of keywords to extract for each cluster
            
        Returns:
        --------
        dict
            Mapping from cluster ID to list of keywords
        """
        print(f"\nExtracting keywords for each cluster in {cluster_column}...")
        
        # Check if vectorizer is available
        if self.vectorizer is None:
            print("TF-IDF vectorizer not available, cannot extract keywords")
            return {}
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Initialize keywords dictionary
        cluster_keywords = {}
        
        # Get unique cluster labels
        unique_clusters = sorted(self.df[cluster_column].unique())
        
        # For each cluster, extract keywords
        for cluster in unique_clusters:
            # Skip noise points (such as -1 in DBSCAN)
            if cluster == -1:
                continue
                
            # Get document indices for this cluster
            cluster_indices = self.df[self.df[cluster_column] == cluster].index.tolist()
            
            # Check if there are documents in this cluster
            if len(cluster_indices) == 0:
                continue
            
            # Get documents for this cluster
            cluster_docs = self.df.loc[cluster_indices, 'combined_text'].tolist()
            
            # Use the TF-IDF vectorizer to process these documents
            cluster_tfidf = self.vectorizer.transform(cluster_docs)
            
            # Sum TF-IDF scores to get the most important words in this cluster
            cluster_tfidf_sum = cluster_tfidf.sum(axis=0)
            
            # Convert to array and flatten
            # Check if we need to convert to array or if it's already an array-like object
            if hasattr(cluster_tfidf_sum, 'toarray'):
                cluster_tfidf_scores = cluster_tfidf_sum.toarray().flatten()
            else:
                # If it's already a matrix or array-like object
                cluster_tfidf_scores = np.array(cluster_tfidf_sum).flatten()
            
            # Get indices of words with highest TF-IDF scores
            top_indices = cluster_tfidf_scores.argsort()[-n_keywords:][::-1]
            
            # Get keywords
            keywords = [feature_names[i] for i in top_indices]
            
            # Store keywords
            cluster_keywords[cluster] = keywords
            
            print(f"Cluster {cluster} ({len(cluster_indices)} documents) keywords: {', '.join(keywords)}")
        
        # Save keywords
        with open(f'clustering_results/{cluster_column}_keywords.txt', 'w') as f:
            for cluster, keywords in cluster_keywords.items():
                f.write(f"Cluster {cluster}: {', '.join(keywords)}\n")
        
        return cluster_keywords
    
    def generate_wordclouds(self, cluster_column='kmeans_cluster'):
        """
        Generate word clouds for each cluster
        
        Parameters:
        -----------
        cluster_column : str
            Name of the column containing cluster labels
        """
        print(f"\nGenerating word clouds for each cluster in {cluster_column}...")
        
        # Create word cloud directory
        wordcloud_dir = f'clustering_results/{cluster_column}_wordclouds'
        os.makedirs(wordcloud_dir, exist_ok=True)
        
        # Get unique cluster labels
        unique_clusters = sorted(self.df[cluster_column].unique())
        
        # For each cluster, generate a word cloud
        for cluster in unique_clusters:
            # Skip noise points (such as -1 in DBSCAN)
            if cluster == -1:
                continue
                
            # Get documents for this cluster
            cluster_docs = self.df[self.df[cluster_column] == cluster]['combined_text'].tolist()
            
            # Check if there are documents in this cluster
            if len(cluster_docs) == 0:
                continue
            
            # Combine all documents
            cluster_text = ' '.join(cluster_docs)
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100
            ).generate(cluster_text)
            
            # Save word cloud image
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud for Cluster {cluster} ({len(cluster_docs)} documents)')
            plt.tight_layout()
            plt.savefig(f'{wordcloud_dir}/cluster_{cluster}_wordcloud.png')
            plt.close()
        
        print(f"Word clouds saved to {wordcloud_dir} directory")
    
    def visualize_clusters_tsne(self, cluster_column='kmeans_cluster', random_state=42):
        """
        Visualize clustering results using t-SNE
        
        Parameters:
        -----------
        cluster_column : str
            Name of the column containing cluster labels
        random_state : int
            Random seed
        """
        print(f"\nVisualizing {cluster_column} clustering results using t-SNE...")
        
        # Check if the data is too large, if so, sample it
        max_samples = 5000
        if self.features.shape[0] > max_samples:
            print(f"Data too large, sampling {max_samples} points for visualization...")
            indices = np.random.choice(self.features.shape[0], max_samples, replace=False)
            features_sample = self.features[indices]
            labels_sample = self.df.loc[indices, cluster_column].values
        else:
            features_sample = self.features
            labels_sample = self.df[cluster_column].values
        
        # Apply t-SNE dimensionality reduction
        print("Applying t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
        features_tsne = tsne.fit_transform(features_sample)
        
        # Create scatter plot
        print("Creating t-SNE scatter plot...")
        plt.figure(figsize=(12, 10))
        
        # Get unique labels and assign colors
        unique_labels = np.unique(labels_sample)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each cluster
        for i, label in enumerate(unique_labels):
            # Assign gray color for noise points in DBSCAN
            if label == -1:
                color = 'gray'
                marker = 'x'
                label_name = 'Noise'
            else:
                color = colors[i]
                marker = 'o'
                label_name = f'Cluster {label}'
            
            # Get indices for this cluster
            idx = labels_sample == label
            
            # Plot points
            plt.scatter(
                features_tsne[idx, 0], 
                features_tsne[idx, 1],
                c=[color], 
                marker=marker,
                label=f'{label_name} ({sum(idx)})',
                alpha=0.7,
                s=50
            )
        
        plt.title(f't-SNE visualization of {cluster_column} clustering')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'clustering_results/{cluster_column}_tsne_visualization.png', dpi=300)
        plt.close()
        
        print(f"t-SNE visualization saved to clustering_results/{cluster_column}_tsne_visualization.png")
    
    def compare_with_categories(self, cluster_column='kmeans_cluster'):
        """
        Compare clustering results with original categories
        
        Parameters:
        -----------
        cluster_column : str
            Name of the column containing cluster labels
        """
        # Check if there's a category column
        if 'category' not in self.df.columns:
            print("No category column in the dataset, cannot compare")
            return
        
        print(f"\nComparing {cluster_column} with original categories...")
        
        # Create cross-tabulation
        crosstab = pd.crosstab(
            self.df[cluster_column], 
            self.df['category'],
            normalize='index'  # Normalize by row
        )
        
        # Plot heatmap
        plt.figure(figsize=(15, 10))
        sns.heatmap(crosstab, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=.5)
        plt.title(f'Relationship between {cluster_column} and categories')
        plt.tight_layout()
        plt.savefig(f'clustering_results/{cluster_column}_category_comparison.png', dpi=300)
        plt.close()
        
        # Save cross-tabulation
        crosstab.to_csv(f'clustering_results/{cluster_column}_category_crosstab.csv')
        
        print(f"Category comparison saved to clustering_results/{cluster_column}_category_comparison.png")
    
    def evaluate_optimal_k(self, k_range=range(5, 21), random_state=42):
        """
        Evaluate the optimal K value (number of clusters)
        
        Parameters:
        -----------
        k_range : range
            Range of K values to try
        random_state : int
            Random seed
        """
        print("\nEvaluating optimal number of clusters for K-Means...")
        
        # Initialize lists to store silhouette scores
        silhouette_scores = []
        inertia_values = []
        
        # For each K value, perform K-Means clustering and calculate evaluation metrics
        for k in k_range:
            print(f"Trying k={k}...")
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(self.features)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(self.features, labels)
            silhouette_scores.append(silhouette_avg)
            
            # Calculate inertia (sum of squared distances to nearest centroid)
            inertia_values.append(kmeans.inertia_)
            
            print(f"  k={k}, silhouette score={silhouette_avg:.4f}, inertia={kmeans.inertia_:.4f}")
        
        # Plot silhouette score
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(list(k_range), silhouette_scores, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Scores for different k values')
        plt.grid(True)
        
        # Plot elbow curve (inertia)
        plt.subplot(1, 2, 2)
        plt.plot(list(k_range), inertia_values, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Curve')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('clustering_results/kmeans_optimal_k_evaluation.png', dpi=300)
        plt.close()
        
        # Save evaluation results
        eval_df = pd.DataFrame({
            'k': list(k_range),
            'silhouette_score': silhouette_scores,
            'inertia': inertia_values
        })
        eval_df.to_csv('clustering_results/kmeans_optimal_k_evaluation.csv', index=False)
        
        # Return the K value with highest silhouette score
        best_k = k_range[np.argmax(silhouette_scores)]
        print(f"\nOptimal number of clusters based on silhouette score: k={best_k} (score={max(silhouette_scores):.4f})")
        
        return best_k


def main():
    """Main function to execute the clustering analysis pipeline"""
    # Set input file paths
    tfidf_features_path = 'tfidf_svd_features.npy'
    w2v_features_path = 'w2v_features.npy'
    data_path = 'feature_engineered_news.csv'
    vectorizer_path = 'tfidf_vectorizer.pkl'
    
    # Specify which feature file to use (TF-IDF or Word2Vec)
    features_path = tfidf_features_path  # Can also be changed to w2v_features_path
    
    print(f"Using features from: {features_path}")
    
    # Initialize clustering model
    clustering = NewsClusteringModel(
        features_path=features_path,
        data_path=data_path,
        vectorizer_path=vectorizer_path
    )
    
    # Evaluate optimal K value (optional)
    # For large datasets, this step might be time-consuming and could be commented out
    best_k = clustering.evaluate_optimal_k(k_range=range(5, 21))
    
    # Perform K-Means clustering
    # If optimal K is known, specify directly; otherwise use preset K (e.g., 15)
    # or use the best_k returned by evaluate_optimal_k function
    clustering.kmeans_clustering(n_clusters=best_k)
    
    # Extract keywords for K-Means clusters
    clustering.extract_cluster_keywords(cluster_column='kmeans_cluster')
    
    # Generate word clouds for K-Means clusters
    clustering.generate_wordclouds(cluster_column='kmeans_cluster')
    
    # Visualize K-Means clustering results using t-SNE
    clustering.visualize_clusters_tsne(cluster_column='kmeans_cluster')
    
    # Compare K-Means clustering with original categories
    clustering.compare_with_categories(cluster_column='kmeans_cluster')
    
    # Perform DBSCAN density clustering
    # Note: DBSCAN parameters eps and min_samples need to be adjusted based on data characteristics
    clustering.dbscan_clustering(eps=0.5, min_samples=10)
    
    # Extract keywords for DBSCAN clusters
    clustering.extract_cluster_keywords(cluster_column='dbscan_cluster')
    
    # Visualize DBSCAN clustering results using t-SNE
    clustering.visualize_clusters_tsne(cluster_column='dbscan_cluster')
    
    # Perform topic modeling
    clustering.topic_modeling(n_topics=best_k)
    
    # Extract keywords for NMF topics
    clustering.extract_cluster_keywords(cluster_column='nmf_topic')
    
    # Visualize NMF topics using t-SNE
    clustering.visualize_clusters_tsne(cluster_column='nmf_topic')
    
    print("\nClustering analysis complete! All results have been saved to the clustering_results directory")


if __name__ == "__main__":
    main()