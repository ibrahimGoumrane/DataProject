import faiss
import numpy as np
import pickle
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import uuid

class VectorStore:
    """
    FAISS-based vector storage system for RAG chatbot.
    Handles embedding storage, similarity search, and metadata management.
    """
    
    def __init__(self, storage_dir="storage", embedding_dim=384):
        """
        Initialize the vector store.
        
        Args:
            storage_dir (str): Directory to store FAISS index and metadata
            embedding_dim (int): Dimension of embeddings (384 for sentence-transformers)
        """
        self.storage_dir = storage_dir
        self.embedding_dim = embedding_dim
        
        # Create storage directory
        os.makedirs(storage_dir, exist_ok=True)
        
        # FAISS index for similarity search
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity)
        
        # Metadata storage (separate from FAISS)
        self.metadata = []  # List of metadata dicts
        self.chunks = []    # List of text chunks
        
        # File paths
        self.index_path = os.path.join(storage_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(storage_dir, "metadata.pkl")
        self.chunks_path = os.path.join(storage_dir, "chunks.pkl")
        self.sessions_path = os.path.join(storage_dir, "sessions.json")
        
        # Load existing data if available
        self._load_existing_data()
        
        print(f"📦 VectorStore initialized with {self.index.ntotal} embeddings")
    
    def _load_existing_data(self):
        """Load existing FAISS index and metadata if they exist."""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                print(f"✅ Loaded existing FAISS index with {self.index.ntotal} vectors")
            
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"✅ Loaded {len(self.metadata)} metadata entries")
            
            if os.path.exists(self.chunks_path):
                with open(self.chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                print(f"✅ Loaded {len(self.chunks)} text chunks")
                    
        except Exception as e:
            print(f"⚠️ Error loading existing data: {e}")
            print("Starting with empty storage")
    
    def add_scrape_session(self, scrape_result: Dict, session_id: Optional[str] = None) -> str:
        """
        Add a complete scraping session to the vector store.
        
        Args:
            scrape_result (Dict): Complete result from scraper.scrape()
            session_id (str, optional): Existing session ID to append to
        
        Returns:
            str: Session ID for tracking
        """
        if not scrape_result or not scrape_result.get('processed_chunks'):
            raise ValueError("Invalid scrape result provided")
        
        # Generate a new session_id if not provided
        is_new_session = session_id is None
        if not session_id:
            session_id = str(uuid.uuid4())
            print(f"💾 Creating new scrape session: {session_id}")
        else:
            print(f"💾 Appending to existing session: {session_id}")
        
        print(f"📊 Chunks to store: {len(scrape_result['processed_chunks'])}")
        
        # Normalize embeddings for cosine similarity
        embeddings = scrape_result['chunk_embeddings']
        if len(embeddings) > 0:
            embeddings = self._normalize_embeddings(embeddings)
        
        # Store in FAISS index
        start_idx = self.index.ntotal
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks with minimal metadata
        for i, chunk in enumerate(scrape_result['processed_chunks']):
            # Create simple metadata
            metadata = {
                'session_id': session_id,
                'vector_index': start_idx + i,
                'source_url': scrape_result['source_url'],
                'query': scrape_result['query'],
                'storage_timestamp': datetime.now().isoformat()
            }
            
            self.chunks.append(chunk)
            self.metadata.append(metadata)
        
        # Save session info
        self._save_session_info(session_id, scrape_result, is_append=not is_new_session)
        
        # Persist to disk
        self._save_to_disk()
        
        print(f"✅ Session stored: {len(scrape_result['processed_chunks'])} chunks added")
        return session_id
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5, 
                         session_id: Optional[str] = None) -> List[Dict]:
        """
        Search for similar chunks using FAISS.
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            k (int): Number of results to return
            session_id (str, optional): Filter by session ID
        
        Returns:
            List[Dict]: Search results with chunks, scores, and metadata
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_embedding = self._normalize_embeddings(query_embedding.reshape(1, -1))
        
        # If filtering by session, search more results to account for filtering
        search_k = k * 3 if session_id else k
        search_k = min(search_k, self.index.ntotal)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):  # Valid index
                result = {
                    'chunk': self.chunks[idx],
                    'metadata': self.metadata[idx],
                    'similarity_score': float(score),
                    'vector_index': idx
                }
                
                # Filter by session if requested
                if session_id is None or self.metadata[idx].get('session_id') == session_id:
                    results.append(result)
                    
                # Stop if we have enough results
                if len(results) >= k:
                    break
        
        # Debug logging
        if session_id:
            print(f"🔍 Session {session_id}: Found {len(results)} results out of {self.index.ntotal} total vectors")
        
        return results
    
    def get_relevant_context(self, query_embedding: np.ndarray, 
                           max_context_length: int = 10000) -> str:
        """
        Get relevant context for RAG generation.
        
        Args:
            query_embedding (np.ndarray): Query embedding
            max_context_length (int): Maximum context length
        
        Returns:
            str: Combined relevant context
        """
        results = self.similarity_search(query_embedding, k=10)
        
        context_parts = []
        total_length = 0
        
        for result in results:
            chunk = result['chunk']
            if total_length + len(chunk) <= max_context_length:
                context_parts.append(chunk)
                total_length += len(chunk)
            else:
                # Add partial chunk if there's room
                remaining = max_context_length - total_length
                if remaining > 50:  # Only if meaningful space left
                    context_parts.append(chunk[:remaining] + "...")
                break
        
        return "\n\n".join(context_parts)
    
    def _save_session_info(self, session_id: str, scrape_result: Dict, is_append: bool = False):
        """Save session information for tracking."""
        sessions = {}
        if os.path.exists(self.sessions_path):
            with open(self.sessions_path, 'r') as f:
                sessions = json.load(f)
        
        # If appending to existing session, update with new website info
        if is_append and session_id in sessions:
            # Keep track of multiple websites in the same session
            if 'websites' not in sessions[session_id]:
                sessions[session_id]['websites'] = []
            
            # Add new website to the list
            new_website = {
                'source_url': scrape_result['source_url'],
                'query': scrape_result['query'],
                'chunks_added': scrape_result['total_chunks'],
                'added_timestamp': datetime.now().isoformat()
            }
            sessions[session_id]['websites'].append(new_website)
            sessions[session_id]['last_updated'] = datetime.now().isoformat()
        else:
            # Create new session info
            sessions[session_id] = {
                'query': scrape_result['query'],
                'source_url': scrape_result['source_url'],
                'total_chunks': scrape_result['total_chunks'],
                'storage_timestamp': datetime.now().isoformat(),
                'websites': [
                    {
                        'source_url': scrape_result['source_url'],
                        'query': scrape_result['query'],
                        'chunks_added': scrape_result['total_chunks'],
                        'added_timestamp': datetime.now().isoformat()
                    }
                ]
            }
        
        with open(self.sessions_path, 'w') as f:
            json.dump(sessions, f, indent=2)
    
    def _save_to_disk(self):
        """Persist all data to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata and chunks
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
                
            print(f"💾 Data persisted: {self.index.ntotal} vectors, {len(self.chunks)} chunks")
            
        except Exception as e:
            print(f"❌ Error saving to disk: {e}")
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get information about a specific session."""
        if os.path.exists(self.sessions_path):
            with open(self.sessions_path, 'r') as f:
                sessions = json.load(f)
                return sessions.get(session_id)
        return None
    
    def list_sessions(self) -> List[Dict]:
        """List all stored sessions."""
        if os.path.exists(self.sessions_path):
            with open(self.sessions_path, 'r') as f:
                sessions = json.load(f)
                return [{'session_id': sid, **info} for sid, info in sessions.items()]
        return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a specific session and its data."""
        # This is complex with FAISS - for now, mark as deleted
        # Full implementation would require rebuilding the index
        print(f"⚠️ Session deletion not fully implemented: {session_id}")
        return False
    
    def get_stats(self) -> Dict:
        """Get storage statistics."""
        return {
            'total_vectors': self.index.ntotal,
            'total_chunks': len(self.chunks),
            'total_metadata': len(self.metadata),
            'embedding_dimension': self.embedding_dim,
            'storage_size_mb': self._get_storage_size(),
            'total_sessions': len(self.list_sessions())
        }
    


    def clear_index_store_by_session(self, session_id: str) -> bool:
        """
        Clear all data associated with a specific session ID.
        
        Args:
            session_id (str): Session ID to clear
        
        Returns:
            bool: True if cleared successfully, False otherwise
        """
        if not os.path.exists(self.sessions_path):
            print(f"⚠️ No sessions found to clear for ID: {session_id}")
            return False
        
        # Load existing sessions
        with open(self.sessions_path, 'r') as f:
            sessions = json.load(f)
        
        # Filter out the session to clear
        if session_id in sessions:
            del sessions[session_id]
            with open(self.sessions_path, 'w') as f:
                json.dump(sessions, f, indent=2)
            
            # Remove associated metadata and chunks
            self.metadata = [m for m in self.metadata if m.get('session_id') != session_id]
            self.chunks = [c for c in self.chunks if c.get('session_id') != session_id]
            
            # Rebuild FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            embeddings = np.array([m['embedding'] for m in self.metadata], dtype='float32')
            if len(embeddings) > 0:
                self.index.add(embeddings)
            
            # Save updated data
            self._save_to_disk()
            
            print(f"✅ Cleared session {session_id} and associated data")
            return True
        
        print(f"⚠️ Session ID {session_id} not found")
        return False

    def get_session_metadata(self, session_id: str) -> List[Dict]:
        """
        Get all metadata entries for a specific session.
        
        Args:
            session_id (str): The session ID to filter by
            
        Returns:
            List[Dict]: All metadata entries for the session
        """
        if not session_id:
            return []
            
        session_metadata = []
        for meta in self.metadata:
            if meta.get('session_id') == session_id:
                session_metadata.append(meta)
                
        return session_metadata
    
    def _get_storage_size(self) -> float:
        """Calculate total storage size in MB."""
        total_size = 0
        for file_path in [self.index_path, self.metadata_path, self.chunks_path, self.sessions_path]:
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
        return total_size / (1024 * 1024)


