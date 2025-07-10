from vector_store import VectorStore
from typing import Dict, List, Optional
import json
import os
import numpy as np

class StorageManager:
    """
    High-level storage manager that coordinates vector storage and retrieval.
    """
    
    def __init__(self, storage_dir="storage"):
        """
        Initialize storage manager.
        
        Args:
            storage_dir (str): Directory for all storage files
        """
        self.storage_dir = storage_dir
        self.vector_store = VectorStore(storage_dir)
        
        # Cache for frequently accessed data
        self._query_cache = {}
        
    def store_scrape_result(self, scrape_result: Dict) -> str:
        """
        Store a complete scraping result.
        
        Args:
            scrape_result (Dict): Result from scraper.scrape()
        
        Returns:
            str: Session ID
        """
        if not self._validate_scrape_result(scrape_result):
            raise ValueError("Invalid scrape result format")
        
        return self.vector_store.add_scrape_session(scrape_result)
    
    def search_content(self, query_embedding: np.ndarray, k: int = 5, 
                      session_id: Optional[str] = None) -> List[Dict]:
        """
        Search for relevant content.
        
        Args:
            query_embedding (np.ndarray): Query embedding
            k (int): Number of results
            session_id (str, optional): Filter by session
        
        Returns:
            List[Dict]: Search results
        """
        return self.vector_store.similarity_search(query_embedding, k, session_id)
    
    def get_context_for_query(self, query_embedding: np.ndarray, 
                            max_length: int = 2000) -> str:
        """
        Get relevant context for LLM generation.
        
        Args:
            query_embedding (np.ndarray): Query embedding
            max_length (int): Maximum context length
        
        Returns:
            str: Relevant context
        """
        return self.vector_store.get_relevant_context(query_embedding, max_length)
    
    def _validate_scrape_result(self, scrape_result: Dict) -> bool:
        """Validate scrape result format."""
        required_fields = [
            'processed_chunks', 
            'chunk_embeddings', 
            'query',
            'source_url'
        ]
        
        for field in required_fields:
            if field not in scrape_result:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Validate data consistency
        chunks = scrape_result['processed_chunks']
        embeddings = scrape_result['chunk_embeddings']
        
        if len(chunks) != len(embeddings):
            print(f"❌ Data length mismatch: chunks={len(chunks)}, embeddings={len(embeddings)}")
            return False
        
        return True
    
    def get_storage_stats(self) -> Dict:
        """Get comprehensive storage statistics."""
        stats = self.vector_store.get_stats()
        stats['cache_size'] = len(self._query_cache)
        return stats
    
    def list_all_sessions(self) -> List[Dict]:
        """List all stored sessions with details."""
        return self.vector_store.list_sessions()
    
    def clear_cache(self):
        """Clear query cache."""
        self._query_cache = {}
        print("🧹 Query cache cleared")