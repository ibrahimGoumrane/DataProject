"""
RAG Enhancement Module
Provides advanced features to improve RAG accuracy and performance.
"""

import numpy as np
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class RAGEnhancer:
    """
    Advanced enhancements for RAG pipeline accuracy.
    """
    
    def __init__(self):
        """Initialize the RAG enhancer."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.query_cache = {}
        
    def rerank_search_results(self, query: str, search_results: List[Dict], 
                            method: str = 'hybrid') -> List[Dict]:
        """
        Re-rank search results using advanced scoring methods.
        
        Args:
            query (str): Original query
            search_results (List[Dict]): Initial search results
            method (str): Reranking method ('hybrid', 'tfidf', 'bm25')
            
        Returns:
            List[Dict]: Re-ranked search results
        """
        if not search_results or len(search_results) < 2:
            return search_results
            
        if method == 'hybrid':
            return self._hybrid_rerank(query, search_results)
        elif method == 'tfidf':
            return self._tfidf_rerank(query, search_results)
        else:
            return search_results
    
    def _hybrid_rerank(self, query: str, search_results: List[Dict]) -> List[Dict]:
        """Hybrid re-ranking combining semantic and keyword matching."""
        
        # Extract chunks and original scores
        chunks = [result.get('chunk', '') for result in search_results]
        semantic_scores = [result.get('similarity_score', 0) for result in search_results]
        
        # Calculate TF-IDF scores
        try:
            all_texts = [query] + chunks
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            query_tfidf = tfidf_matrix[0]
            chunk_tfidf = tfidf_matrix[1:]
            
            tfidf_scores = cosine_similarity(query_tfidf, chunk_tfidf).flatten()
        except:
            tfidf_scores = [0.0] * len(chunks)
        
        # Calculate keyword overlap scores
        keyword_scores = self._calculate_keyword_overlap(query, chunks)
        
        # Combine scores with weights
        combined_scores = []
        for i in range(len(search_results)):
            semantic_weight = 0.5
            tfidf_weight = 0.3
            keyword_weight = 0.2
            
            combined_score = (
                semantic_weight * semantic_scores[i] +
                tfidf_weight * tfidf_scores[i] +
                keyword_weight * keyword_scores[i]
            )
            combined_scores.append(combined_score)
        
        # Re-rank based on combined scores
        ranked_indices = np.argsort(combined_scores)[::-1]
        
        reranked_results = []
        for idx in ranked_indices:
            result = search_results[idx].copy()
            result['rerank_score'] = combined_scores[idx]
            result['original_rank'] = idx + 1
            reranked_results.append(result)
            
        return reranked_results
    
    def _tfidf_rerank(self, query: str, search_results: List[Dict]) -> List[Dict]:
        """Re-rank using TF-IDF similarity."""
        chunks = [result.get('chunk', '') for result in search_results]
        
        try:
            all_texts = [query] + chunks
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            query_tfidf = tfidf_matrix[0]
            chunk_tfidf = tfidf_matrix[1:]
            
            tfidf_scores = cosine_similarity(query_tfidf, chunk_tfidf).flatten()
            
            # Re-rank based on TF-IDF scores
            ranked_indices = np.argsort(tfidf_scores)[::-1]
            
            reranked_results = []
            for idx in ranked_indices:
                result = search_results[idx].copy()
                result['tfidf_score'] = tfidf_scores[idx]
                result['original_rank'] = idx + 1
                reranked_results.append(result)
                
            return reranked_results
        except:
            return search_results
    
    def _calculate_keyword_overlap(self, query: str, chunks: List[str]) -> List[float]:
        """Calculate basic keyword overlap scores between query and chunks."""
        # Split by whitespace instead of regex to preserve more words
        query_words = set(query.lower().split())
        
        scores = []
        for chunk in chunks:
            # Split by whitespace to preserve more words
            chunk_words = set(chunk.lower().split())
            
            if not query_words or not chunk_words:
                scores.append(0.0)
                continue
                
            # Count the words that appear in both
            overlap_count = len(query_words.intersection(chunk_words))
            
            # Simple overlap score
            score = overlap_count / len(query_words) if query_words else 0.0
            scores.append(score)
            
        return scores
    
    def enhance_context_selection(self, query: str, search_results: List[Dict], 
                                max_context_length: int = 10000) -> str:
        """
        Intelligently select and combine context from search results,
        preserving more of the original content.
        
        Args:
            query (str): User query
            search_results (List[Dict]): Search results
            max_context_length (int): Maximum context length
            
        Returns:
            str: Enhanced context
        """
        if not search_results:
            return ""
            
        # Only re-rank the results, but use more results overall
        reranked_results = self.rerank_search_results(query, search_results)
        
        # Collect chunks with minimal redundancy filtering
        selected_chunks = []
        total_length = 0
        seen_chunks = set()
        
        for result in reranked_results:
            chunk = result.get('chunk', '')
            
            # Skip empty chunks and exact duplicates
            if not chunk or chunk in seen_chunks:
                continue
                
            # Add chunk if it fits within the max length
            if total_length + len(chunk) <= max_context_length:
                selected_chunks.append({
                    'chunk': chunk,
                    'rerank_score': result.get('rerank_score', result.get('similarity_score', 0)),
                })
                seen_chunks.add(chunk)
                total_length += len(chunk)
            else:
                # Add partial chunk if there's significant room left
                remaining_length = max_context_length - total_length
                if remaining_length > 200:
                    partial_chunk = chunk[:remaining_length]
                    selected_chunks.append({
                        'chunk': partial_chunk,
                        'rerank_score': result.get('rerank_score', result.get('similarity_score', 0)),
                    })
                break
        
        # Combine chunks with relevance indicators
        context_parts = []
        for chunk_data in selected_chunks:
            chunk = chunk_data['chunk']
            score = chunk_data['rerank_score']
            context_parts.append(f"[Relevance: {score:.3f}] {chunk}")
        
        return "\n\n".join(context_parts)
    
    def _is_redundant(self, new_chunk: str, existing_chunks: List[str], 
                     threshold: float = 0.95) -> bool:
        """
        Check if a chunk is too similar to existing chunks.
        Only considers exact or near-exact duplicates as redundant.
        """
        if not existing_chunks:
            return False
            
        # Use a higher threshold - only consider extremely similar content as redundant
        # Split by whitespace to preserve more content
        new_words = set(new_chunk.lower().split())
        
        if not new_words:
            return False
            
        for existing_chunk in existing_chunks:
            existing_words = set(existing_chunk.lower().split())
            
            if not existing_words:
                continue
                
            # Only check overlap in one direction (what percentage of the new chunk is in the existing chunk)
            overlap = len(new_words.intersection(existing_words)) / len(new_words)
            
            # Only filter out chunks that are nearly identical
            if overlap > threshold:
                return True
                
        return False
    
    def analyze_query_complexity(self, query: str) -> Dict:
        """
        Analyze query complexity to adjust search strategy.
        
        Args:
            query (str): User query
            
        Returns:
            Dict: Query analysis results
        """
        analysis = {
            'word_count': len(query.split()),
            'character_count': len(query),
            'question_type': self._detect_question_type(query),
            'complexity_score': 0.0,
            'entities': self._extract_simple_entities(query),
            'suggested_k': 10  # Default number of results
        }
        
        # Calculate complexity score
        complexity_factors = [
            analysis['word_count'] / 10,  # More words = more complex
            len(analysis['entities']) / 5,  # More entities = more complex
            1 if '?' in query else 0.5,  # Questions are more complex
        ]
        
        analysis['complexity_score'] = min(sum(complexity_factors) / len(complexity_factors), 1.0)
        
        # Adjust suggested k based on complexity
        if analysis['complexity_score'] > 0.7:
            analysis['suggested_k'] = 10
        elif analysis['complexity_score'] > 0.4:
            analysis['suggested_k'] = 7
        else:
            analysis['suggested_k'] = 5
            
        return analysis
    
    def _detect_question_type(self, query: str) -> str:
        """Detect the type of question being asked."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'define', 'explain']):
            return 'definition'
        elif any(word in query_lower for word in ['how', 'steps', 'process']):
            return 'procedural'
        elif any(word in query_lower for word in ['why', 'because', 'reason']):
            return 'causal'
        elif any(word in query_lower for word in ['when', 'time', 'date']):
            return 'temporal'
        elif any(word in query_lower for word in ['where', 'location', 'place']):
            return 'spatial'
        elif any(word in query_lower for word in ['compare', 'difference', 'vs']):
            return 'comparative'
        else:
            return 'general'
    
    def _extract_simple_entities(self, query: str) -> List[str]:
        """
        Extract simple entities with minimal processing.
        Focuses on preserving key terms from the query.
        """
        entities = []
        
        # Keep capitalized words as potential entities
        for word in query.split():
            if word and word[0].isupper():
                entities.append(word)
        
        # Find quoted terms
        entities.extend(re.findall(r'"([^"]*)"', query))
        entities.extend(re.findall(r"'([^']*)'", query))
        
        # Add important keywords (longer words are often more significant)
        keywords = [word for word in query.split() if len(word) > 5]
        entities.extend(keywords)
        
        return list(set(entities))
    
    def suggest_context_improvements(self, query: str, context: str, 
                                   answer_quality: float = 0.0) -> Dict:
        """
        Suggest improvements for context selection.
        
        Args:
            query (str): User query
            context (str): Current context
            answer_quality (float): Estimated answer quality (0-1)
            
        Returns:
            Dict: Improvement suggestions
        """
        suggestions = {
            'needs_more_context': False,
            'context_too_long': False,
            'missing_keywords': [],
            'redundant_content': False,
            'suggestions': []
        }
        
        # Check context length
        if len(context) < 500:
            suggestions['needs_more_context'] = True
            suggestions['suggestions'].append("Consider retrieving more context chunks")
        elif len(context) > 3000:
            suggestions['context_too_long'] = True
            suggestions['suggestions'].append("Context might be too long, consider filtering")
        
        # Check for missing keywords
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        context_words = set(re.findall(r'\b\w+\b', context.lower()))
        
        missing_keywords = query_words - context_words
        if missing_keywords:
            suggestions['missing_keywords'] = list(missing_keywords)
            suggestions['suggestions'].append(f"Missing keywords: {', '.join(list(missing_keywords)[:3])}")
        
        # Check for redundancy
        sentences = context.split('. ')
        if len(sentences) > 5:
            unique_sentences = len(set(sentences))
            if unique_sentences / len(sentences) < 0.8:
                suggestions['redundant_content'] = True
                suggestions['suggestions'].append("Context contains redundant information")
        
        # Quality-based suggestions
        if answer_quality < 0.5:
            suggestions['suggestions'].append("Consider expanding search criteria or adjusting query")
        
        return suggestions
