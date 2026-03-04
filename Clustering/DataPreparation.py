import pandas as pd
import numpy as np
import re
import nltk
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class NewsDataPreparation:
    def __init__(self, input_file_path):
        """
        Initialize the data preparation class
        
        Parameters:
        -----------
        input_file_path : str
            Path to the input JSON file containing news data
        """
        self.input_file_path = input_file_path
        self.download_nltk_resources()
        
    def download_nltk_resources(self):
        """Download required NLTK resources"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
    
    def load_data(self):
        """Load the dataset and perform initial checks"""
        print(f"Loading data from {self.input_file_path}...")
        
        # Load the JSON dataset
        try:
            self.news_df = pd.read_json(self.input_file_path, lines=True)
        except Exception as e:
            print(f"Error loading JSON with lines=True: {e}")
            print("Trying alternative JSON loading method...")
            try:
                # Try loading as a regular JSON file
                self.news_df = pd.read_json(self.input_file_path)
            except Exception as e2:
                print(f"Error with alternative method: {e2}")
                # If both methods fail, try reading the file line by line
                print("Trying to load JSON line by line...")
                with open(self.input_file_path, 'r', encoding='utf-8') as f:
                    data = []
                    for line in f:
                        try:
                            data.append(json.loads(line))
                        except:
                            continue
                self.news_df = pd.DataFrame(data)
        
        # Print basic information
        print(f"Dataset loaded with shape: {self.news_df.shape}")
        print("\nFirst 5 rows:")
        print(self.news_df.head())
        
        # Check for column names
        print("\nColumns in the dataset:")
        print(self.news_df.columns.tolist())
        
        # Check for missing values
        print("\nMissing values per column:")
        print(self.news_df.isnull().sum())
        
        # Check for duplicates
        duplicates = self.news_df.duplicated().sum()
        print(f"\nNumber of duplicate rows: {duplicates}")
        
        return self.news_df
    
    def filter_top_categories(self, n=15):
        """
        Filter to only include the top N categories
        
        Parameters:
        -----------
        n : int
            Number of top categories to include
        """
        print(f"\nFiltering to include only the top {n} categories...")
        
        # Get category counts
        category_counts = self.news_df['category'].value_counts()
        
        # Display top categories
        print(f"Top {n} categories:")
        print(category_counts.head(n))
        
        # Get the names of the top categories
        top_categories = category_counts.head(n).index.tolist()
        
        # Filter the dataset
        initial_count = len(self.news_df)
        self.news_df = self.news_df[self.news_df['category'].isin(top_categories)]
        final_count = len(self.news_df)
        
        print(f"Filtered from {initial_count} to {final_count} articles")
        
        return self.news_df
    
    def clean_data(self, content_column='short_description', title_column='headline'):
        """
        Clean the data by handling missing values and duplicates
        
        Parameters:
        -----------
        content_column : str
            Name of the column containing the main text content
        title_column : str
            Name of the column containing the news title
        """
        print("\nCleaning data...")
        
        # Ensure content_column and title_column exist
        if content_column not in self.news_df.columns:
            raise ValueError(f"Column '{content_column}' not found in the dataset")
        
        # Handle missing values
        initial_count = len(self.news_df)
        self.news_df = self.news_df.dropna(subset=[content_column])
        after_na_count = len(self.news_df)
        print(f"Removed {initial_count - after_na_count} rows with missing content")
        
        # Fill missing values in other columns if any
        for col in self.news_df.columns:
            if col != content_column and self.news_df[col].isnull().sum() > 0:
                if self.news_df[col].dtype == 'object':
                    self.news_df[col] = self.news_df[col].fillna('')
                else:
                    self.news_df[col] = self.news_df[col].fillna(0)
        
        # Remove exact duplicates (much faster than similar article detection)
        initial_count = len(self.news_df)
        self.news_df = self.news_df.drop_duplicates()
        after_dup_count = len(self.news_df)
        print(f"Removed {initial_count - after_dup_count} exact duplicate rows")
        
        # Reset index
        self.news_df = self.news_df.reset_index(drop=True)
        
        return self.news_df
    
    def preprocess_text(self, content_column='short_description', title_column='headline', sample_size=None):
        """
        Preprocess text content for clustering
        
        Parameters:
        -----------
        content_column : str
            Name of the column containing the main text content
        title_column : str
            Name of the column containing the news title
        sample_size : int or None
            Size of random sample to use (None to use full dataset)
        """
        print("\nPreprocessing text...")
        
        # Use random sample if specified (for faster processing)
        if sample_size is not None and sample_size < len(self.news_df):
            print(f"Using random sample of {sample_size} articles for faster processing")
            self.news_df = self.news_df.sample(sample_size, random_state=42).reset_index(drop=True)
        
        # Create processed content column
        print("Processing content...")
        self.news_df['processed_content'] = self.news_df[content_column].apply(
            lambda x: self._clean_and_tokenize_text(x) if isinstance(x, str) else ""
        )
        
        # Process titles if available
        print("Processing titles...")
        if title_column in self.news_df.columns:
            self.news_df['processed_title'] = self.news_df[title_column].apply(
                lambda x: self._clean_and_tokenize_text(x) if isinstance(x, str) else ""
            )
            
            # Combine title and content for better clustering results
            print("Combining title and content...")
            self.news_df['combined_text'] = self.news_df['processed_title'] + ' ' + self.news_df['processed_content']
        else:
            self.news_df['combined_text'] = self.news_df['processed_content']
        
        # Count words after processing
        self.news_df['word_count'] = self.news_df['processed_content'].apply(
            lambda x: len(x.split())
        )
        
        # Filter out articles that are too short
        min_words = 3  # Set lower for short descriptions
        initial_count = len(self.news_df)
        self.news_df = self.news_df[self.news_df['word_count'] >= min_words]
        after_filter_count = len(self.news_df)
        print(f"Removed {initial_count - after_filter_count} articles with fewer than {min_words} words")
        
        return self.news_df
    
    def _clean_and_tokenize_text(self, text):
        """
        Clean and tokenize text data
        
        Parameters:
        -----------
        text : str
            The text to process
            
        Returns:
        --------
        str
            Processed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Return processed text
        return ' '.join(tokens)
    
    def standardize_format(self, keep_columns=None):
        """
        Standardize the data format for team use
        
        Parameters:
        -----------
        keep_columns : list
            List of columns to keep in the final dataset
            
        Returns:
        --------
        pandas.DataFrame
            Standardized dataset
        """
        print("\nStandardizing data format...")
        
        # Define columns to keep if not specified
        if keep_columns is None:
            # Check which columns are available in the dataset
            available_columns = set(self.news_df.columns)
            
            # Define potential columns of interest
            potential_columns = ['headline', 'short_description', 'category', 
                                'processed_content', 'processed_title', 'combined_text',
                                'word_count', 'date', 'authors', 'link']
            
            # Find intersection of available and potential columns
            keep_columns = list(available_columns.intersection(potential_columns))
            
            # Always include processed text columns
            text_columns = ['processed_content', 'processed_title', 'combined_text']
            for col in text_columns:
                if col in available_columns and col not in keep_columns:
                    keep_columns.append(col)
        
        # Keep only necessary columns
        columns_to_keep = [col for col in keep_columns if col in self.news_df.columns]
        self.final_df = self.news_df[columns_to_keep].copy()
        
        # Reset index
        self.final_df = self.final_df.reset_index(drop=True)
        
        print(f"Final dataset shape: {self.final_df.shape}")
        print(f"Final columns: {self.final_df.columns.tolist()}")
        
        return self.final_df
    
    def analyze_data(self):
        """Analyze the processed data and generate summary statistics"""
        print("\nGenerating data analysis...")
        
        # Create an output directory for analysis
        os.makedirs('data_analysis', exist_ok=True)
        
        # Word count distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.news_df, x='word_count', bins=50)
        plt.title('Distribution of Word Count in Articles')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.savefig('data_analysis/word_count_distribution.png')
        plt.close()
        
        # Category distribution
        if 'category' in self.news_df.columns:
            plt.figure(figsize=(12, 8))
            category_counts = self.news_df['category'].value_counts()
            
            # Limit to top 15 categories if there are too many
            if len(category_counts) > 15:
                category_counts = category_counts.head(15)
                
            sns.barplot(x=category_counts.values, y=category_counts.index)
            plt.title('Distribution of News Categories (Top 15)')
            plt.xlabel('Count')
            plt.tight_layout()
            plt.savefig('data_analysis/category_distribution.png')
            plt.close()
        
        # Generate summary statistics
        summary_stats = {
            'Total articles': len(self.news_df),
            'Average word count': self.news_df['word_count'].mean(),
            'Median word count': self.news_df['word_count'].median(),
            'Min word count': self.news_df['word_count'].min(),
            'Max word count': self.news_df['word_count'].max()
        }
        
        # Add category counts if available
        if 'category' in self.news_df.columns:
            summary_stats['Number of categories'] = self.news_df['category'].nunique()
            summary_stats['Most common category'] = self.news_df['category'].mode()[0]
        
        with open('data_analysis/summary_statistics.txt', 'w') as f:
            for key, value in summary_stats.items():
                f.write(f"{key}: {value}\n")
        
        return summary_stats
    
    def save_processed_data(self, output_file_path='processed_news_data.csv'):
        """
        Save the processed dataset
        
        Parameters:
        -----------
        output_file_path : str
            Path where to save the processed CSV file
        """
        print(f"\nSaving processed data to {output_file_path}...")
        self.final_df.to_csv(output_file_path, index=False)
        print("Data preparation completed successfully!")
        
        return output_file_path


def main():
    """Main function to execute the data preparation pipeline"""
    # Set path to your JSON dataset
    input_file_path = r"E:\Graduate study\2025winter\5125\Final Project\Clustering\News_Category_Dataset_v3.json"
    output_file_path = 'processed_news_data.csv'
    
    # Sample size for faster processing (set to None to use full dataset)
    # For initial testing, use a smaller sample (e.g., 10000)
    sample_size = 10000  # Change to None for final run
    
    # Initialize data preparation
    data_prep = NewsDataPreparation(input_file_path)
    
    # Execute data preparation pipeline
    data_prep.load_data()
    
    # Filter to top 15 categories
    data_prep.filter_top_categories(n=15)
    
    # Continue with data preparation (skip similarity detection to save time)
    data_prep.clean_data(content_column='short_description', title_column='headline')
    
    # Use sample_size for faster processing
    data_prep.preprocess_text(
        content_column='short_description', 
        title_column='headline',
        sample_size=sample_size
    )
    
    data_prep.standardize_format()
    data_prep.analyze_data()
    data_prep.save_processed_data(output_file_path)


if __name__ == "__main__":
    main()