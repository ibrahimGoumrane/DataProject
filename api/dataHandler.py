import os
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Import libraries for cosine similarity and vectorization
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from html2text import HTML2Text
import numpy as np

class DataHandler:
    PATH="data"
    def __init__(self):
        self.saved_path = self.PATH
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
        
        # Chunk size for splitting long texts
        self.chunk_size = 500  # characters per chunk
        self.chunk_overlap = 50  # overlap between chunks
        
        os.makedirs(self.saved_path, exist_ok=True)
        
    def clean_html_text(self, html_content):
        """
        Clean HTML content and convert to readable text.
        """
        if isinstance(html_content, bytes):
            html_content = html_content.decode('utf-8', errors='ignore')
        
        # Convert HTML to markdown/text
        text = self.html2text.handle(html_content)
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
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
        Filter out low-quality chunks (too short, repetitive, etc.)
        """
        filtered_chunks = []
        seen_chunks = set()
        
        for chunk in text_chunks:
            # Skip very short chunks
            if len(chunk.strip()) < 50:
                continue
                
            # Skip chunks that are mostly numbers or special characters
            alpha_ratio = sum(c.isalpha() for c in chunk) / len(chunk)
            if alpha_ratio < 0.6:
                continue
                
            # Skip duplicate chunks
            chunk_normalized = re.sub(r'\s+', ' ', chunk.lower().strip())
            if chunk_normalized in seen_chunks:
                continue
                
            seen_chunks.add(chunk_normalized)
            filtered_chunks.append(chunk)
        
        return filtered_chunks
    def process_query(self, query):
        """
        Process the query to extract relevant information.
        Enhanced with better preprocessing.
        """
        # Clean the query
        query = re.sub(r'[^\w\s]', ' ', query)  # Remove special characters
        query = re.sub(r'\s+', ' ', query).strip()  # Normalize whitespace
        
        tokens = word_tokenize(query)
        tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in self.stop_words]
        # Also remove punctuation or special characters
        tokens = [token for token in tokens if token.isalpha() and len(token) > 2]  # Remove very short tokens
        
        # Create both processed and original query embeddings
        processed_query = ' '.join(tokens)
        query_embedding = self.model.encode(processed_query if processed_query else query)

        return tokens, query_embedding

    def process_html(self, html_text_content):
        """
        Enhanced HTML processing with chunking and quality filtering.
        """
        # Clean HTML content first
        clean_text = self.clean_html_text(html_text_content)
        
        # Chunk the text
        text_chunks = self.chunk_text(clean_text)
        
        # Filter quality
        filtered_chunks = self.filter_content_quality(text_chunks)
        
        if not filtered_chunks:
            return [], np.array([])
        
        # Process each chunk
        processed_chunks = []
        for chunk in filtered_chunks:
            sentences = sent_tokenize(chunk)
            processed_sentences = []
            
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in self.stop_words]
                tokens = [token for token in tokens if token.isalpha() and len(token) > 2]
                
                if tokens:  # Only add if we have meaningful tokens
                    processed_sentences.append(' '.join(tokens))
            
            if processed_sentences:
                processed_chunks.append(' '.join(processed_sentences))
        
        if not processed_chunks:
            return [], np.array([])
            
        # Embed the processed chunks
        data_embeddings = self.model.encode(processed_chunks)
        
        return processed_chunks, data_embeddings
    def compute_similarity(self, text_content, query, top_k=5):
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

    def get_relevant_context(self, text_content, query, max_context_length=2000):
        """
        Get the most relevant context for a query, respecting length limits.
        
        Args:
            text_content: Raw content to search
            query: User query
            max_context_length: Maximum characters in returned context
            
        Returns:
            str: Concatenated relevant context
        """
        similar_chunks = self.compute_similarity(text_content, query, top_k=10)
        
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
                if remaining_length > 100:  # Only if meaningful amount remains
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

