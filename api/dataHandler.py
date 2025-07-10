import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from html2text import HTML2Text
import numpy as np
from config import get_config
from typing import List, Tuple
class DataHandler:
    PATH="data"
    def __init__(self):
        self.saved_path = self.PATH
        self.config = get_config()
        self._initialize_data()
    def _initialize_data(self):
        """
        Initialize the data handler with necessary components.
        This method sets up the tokenizer, lemmatizer, stop words, and model for embeddings
        """
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained model for embeddings
        
        # Initialize HTML to text converter
        self.html2text = HTML2Text()
        self.html2text.ignore_links = True
        self.html2text.ignore_images = True
        
        # Configuration from centralized config
        self.chunk_size = self.config.CHUNK_SIZE
        self.chunk_overlap = self.config.CHUNK_OVERLAP
        self.max_context_length = self.config.DATA_MAX_CONTEXT_LENGTH
        
        os.makedirs(self.saved_path, exist_ok=True)
        
    def clean_html_text(self, html_content):
        """
        Clean HTML content and convert to readable text.
        Minimally processes text to preserve original content meaning.
        """
        if isinstance(html_content, bytes):
            html_content = html_content.decode('utf-8', errors='ignore')
        
        # Convert HTML to markdown/text
        text = self.html2text.handle(html_content)
        
        # Basic normalization - just normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text.strip()

    def chunk_text(self, text, chunk_size=None, overlap=None):
        """
        Split text into chunks for better processing and retrieval.
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.chunk_overlap
            
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence ending within reasonable distance
                sentence_end = text.rfind('.', start, end + 100)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
        return chunks

    def filter_content_quality(self, text_chunks):
        """
        Minimal content filtering to preserve original meaning.
        Only filters out completely empty chunks and exact duplicates.
        """
        filtered_chunks = []
        seen_chunks = set()
        
        for chunk in text_chunks:
            # Skip empty chunks
            if not chunk.strip():
                continue
                
            # Skip exact duplicate chunks (keep casing and whitespace)
            if chunk in seen_chunks:
                continue
                
            seen_chunks.add(chunk)
            filtered_chunks.append(chunk)
        
        return filtered_chunks
    def process_query(self, query):
        """
        Simplified query processing that preserves original query meaning.
        Only performs basic normalization.
        """
        # Basic normalization - just normalize whitespace
        normalized_query = re.sub(r'\s+', ' ', query).strip()
        
        # Get embedding for the original query
        _, query_embedding = self.get_embeddings(normalized_query)
        
        # Return original tokens (split by whitespace) and embedding
        tokens = normalized_query.split()
        
        return tokens, query_embedding

    def process_html(self, html_text_content, no_html_cleaning=False):
        """
        Simplified HTML processing that preserves original content meaning.
        Minimal preprocessing to maintain content integrity.
        """
        # Clean HTML content first
        if not no_html_cleaning:
            clean_text = self.clean_html_text(html_text_content)
        else:
            clean_text = html_text_content
            
        # Chunk the text
        text_chunks = self.chunk_text(clean_text)
        
        # Minimal filtering - just remove empty chunks and exact duplicates
        filtered_chunks = self.filter_content_quality(text_chunks)
        
        if not filtered_chunks:
            return [], np.array([])
            
        # Embed the original chunks without excessive processing
        return self.get_embeddings(filtered_chunks)
    
    def get_embeddings(self, chunks):
        """
        Get embeddings for the provided list of text documents.
        
        Args:
            list: List of text chunks

        Returns:
            list: List of processed text chunks
            np.array: Corresponding embeddings
        """
        data_embeddings = self.model.encode(chunks)
        return chunks, data_embeddings
    
    def compute_similarity(self, text_content, query, top_k=10):
        """
        Enhanced similarity computation with ranking and filtering.
        
        Args:
            text_content: Raw HTML or text content
            query: User query
            top_k: Number of top results to return
            
        Returns:
            list: Top k similar chunks with their similarity scores
        """
        # Process the query to get its embedding
        _, query_embedding = self.process_query(query)
        
        # Process the HTML content to get its embeddings
        processed_chunks, data_embeddings = self.process_html(text_content)
        
        if len(processed_chunks) == 0:
            return []
        
        # Compute similarities
        similarities = cosine_similarity([query_embedding], data_embeddings)[0]
        
        # Create ranked results
        chunk_scores = list(zip(processed_chunks, similarities))
        
        # Sort by similarity score (descending)
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out very low similarity scores (< 0.1)
        chunk_scores = [(chunk, score) for chunk, score in chunk_scores if score > 0.1]
        
        # Return top k results
        return chunk_scores[:top_k]

    def get_relevant_context(self, text_content, query, max_context_length=None):
        """
        Get the most relevant context for a query, respecting length limits.
        
        Args:
            text_content: Raw content to search
            query: User query
            max_context_length: Maximum characters in returned context (uses default if None)
            
        Returns:
            str: Concatenated relevant context
        """
        if max_context_length is None:
            max_context_length = self.max_context_length
            
        similar_chunks = self.compute_similarity(text_content, query, top_k=20)
        
        if not similar_chunks:
            return ""
        
        # Build context from most similar chunks
        context_parts = []
        total_length = 0
        
        for chunk, score in similar_chunks:
            if total_length + len(chunk) <= max_context_length:
                context_parts.append(f"[Score: {score:.3f}] {chunk}")
                total_length += len(chunk)
            else:
                # Add partial chunk if it fits
                remaining_length = max_context_length - total_length
                if remaining_length > 200:  # Only if meaningful amount remains
                    context_parts.append(f"[Score: {score:.3f}] {chunk[:remaining_length]}...")
                break
        
        return "\n\n".join(context_parts)
    def save_data(self, data):
        """
        Save the processed data to a file or database.
        In Html5, this method would typically use a library like pymongo for MongoDB
        In a mongoDB or a file I/O operation.
        """
        # Placeholder for data saving logic
        for i , item in enumerate(data):
            # Save inside the saved_path directory
            with open(os.path.join(self.saved_path, f"item_{i}.txt"), "w") as file:
                print(f"Saving item {i} to {self.saved_path}")
                file.write(item)
        print(f"Data saved successfully to {self.saved_path}")
        return True            

    def load_data(self):
        """
        Load the data from a file or database.
        In Html5, this method would typically use a library like SQLite or a file I/O operation
        """
        # Placeholder for data loading logic
        return "Loaded data."
    
    def clean_data(self, data):
        """
        Clean the data by removing duplicates or irrelevant information.
        In Html5, this method would typically use a library like pandas for data manipulation
        """
        # Placeholder for data cleaning logic
        return "Data cleaned successfully."
    
    def get_k_nearest_neighbors(self, query, k=5):
        """
        Get the k-nearest neighbors for a given query.
        In Html5, this method would typically use a library like scikit-learn for nearest neighbor search
        """
        # Placeholder for nearest neighbor logic
        return f"Retrieved {k} nearest neighbors for query: {query}"
    
    def enhance_query_preprocessing(self, query: str) -> str:
        """
        Enhanced query preprocessing for better matching.
        
        Args:
            query (str): Original query
            
        Returns:
            str: Enhanced query
        """
        # Expand contractions
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not"
        }
        
        enhanced_query = query.lower()
        for contraction, expansion in contractions.items():
            enhanced_query = enhanced_query.replace(contraction, expansion)
        
        # Add synonyms for common terms
        synonyms = {
            "how to": "how do I",
            "what is": "what are",
            "tell me": "explain",
            "show me": "demonstrate"
        }
        
        for original, synonym in synonyms.items():
            if original in enhanced_query:
                enhanced_query += f" {synonym}"
        
        return enhanced_query

    def improve_chunk_quality(self, chunks: List[str]) -> List[str]:
        """
        Improve chunk quality by filtering and enhancing.
        
        Args:
            chunks (List[str]): Original chunks
            
        Returns:
            List[str]: Enhanced chunks
        """
        improved_chunks = []
        
        for chunk in chunks:
            # Skip very short or very long chunks
            if len(chunk.strip()) < 30 or len(chunk) > 10000:
                continue
            
            # Clean up chunk
            clean_chunk = self._clean_chunk(chunk)
            
            # Check if chunk has meaningful content
            if self._has_meaningful_content(clean_chunk):
                improved_chunks.append(clean_chunk)
        
        return improved_chunks
    
    def _clean_chunk(self, chunk: str) -> str:
        """Clean and normalize a text chunk."""
        # Remove excessive whitespace
        chunk = re.sub(r'\s+', ' ', chunk)
        
        # Remove repeated punctuation
        chunk = re.sub(r'[.]{3,}', '...', chunk)
        chunk = re.sub(r'[-]{3,}', '---', chunk)
        
        # Fix common encoding issues
        chunk = chunk.replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
        
        # Ensure proper sentence endings
        chunk = chunk.strip()
        if chunk and not chunk.endswith(('.', '!', '?', ':')):
            chunk += '.'
        
        return chunk
    
    def _has_meaningful_content(self, chunk: str) -> bool:
        """Check if chunk contains meaningful content."""
        # Check word count
        words = chunk.split()
        if len(words) < 5:
            return False
        
        # Check for meaningful words (not just numbers and symbols)
        alpha_words = [word for word in words if word.isalpha()]
        if len(alpha_words) < 3:
            return False
        
        # Check for repeated patterns
        if len(set(words)) / len(words) < 0.3:  # Too much repetition
            return False
        
        return True

