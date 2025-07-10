import numpy as np
import os
from scraper import Scraper
from dataHandler import DataHandler
from storage_manager import StorageManager
from config import get_config, load_env_file
from llm import LLM
from rag_enhancer import RAGEnhancer
from typing import Dict, List, Optional


class RAGPipeline:
    """
    Complete RAG pipeline: Scrape → Store → Retrieve → Generate
    """
    
    def __init__(self, storage_dir=None, openai_api_key=None, config_file='.env'):
        """Initialize the RAG pipeline."""
        # Load configuration from .env file if it exists
        load_env_file(config_file)
        self.config = get_config()
        
        # Validate configuration
        validation = self.config.validate()
        if not validation['valid']:
            print("❌ Configuration errors:")
            for error in validation['errors']:
                print(f"  - {error}")
            raise ValueError("Invalid configuration")
        
        if validation['warnings']:
            print("⚠️ Configuration warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        # Initialize components
        self.scraper = Scraper()
        self.data_handler = DataHandler()
        self.storage_manager = StorageManager(storage_dir or self.config.STORAGE_DIR)
        
        # Initialize LLM and enhancer
        self.llm = LLM(openai_api_key)
        self.enhancer = RAGEnhancer()
        
        print("🤖 RAG Pipeline initialized")
        print(f"📊 Configuration: max_context={self.config.MAX_CONTEXT_LENGTH}, model={self.config.MODEL_NAME}")
        print(f"🔑 OpenAI: {'✅ Available' if self.llm.is_available() else '❌ Not configured'}")
        print("🚀 Enhanced RAG features enabled")
    
    def process_website(self, url: str, query: str, session_id: Optional[str] = None) -> Dict:
        """
        Complete pipeline: scrape website and store in vector database.
        
        Args:
            url (str): Website URL to scrape
            query (str): User query for context
            session_id (str, optional): Existing session ID to append to for combining multiple websites
        
        Returns:
            Dict: Processing result with session info
        """
        print(f"🚀 Processing website: {url}")
        print(f"📝 Query: {query}")
        
        # 1. Scrape the website
        scrape_result = self.scraper.scrape(url=url, query=query)
        
        if not scrape_result:
            return {"error": "Failed to scrape website", "success": False}
        
        # 2. Store in vector database
        try:
            # Use existing session ID if provided (for multi-website context)
            if session_id:
                print(f"📌 Appending to existing session: {session_id}")
                
            session_id = self.storage_manager.store_scrape_result(
                scrape_result,
                session_id=session_id
            )
            
            # 3. Get immediate relevant context
            relevant_context = self.storage_manager.get_context_for_query(
                scrape_result['query_embedding']
            )
            
            return {
                "success": True,
                "session_id": session_id,
                "chunks_stored": scrape_result['total_chunks'],
                "source_url": url,
                "query": query,
                "immediate_context": relevant_context[:500] + "..." if len(relevant_context) > 500 else relevant_context
            }
            
        except Exception as e:
            print(f"❌ Storage error: {e}")
            return {"error": f"Storage failed: {e}", "success": False}
    
    def answer_question(self, query: str, session_id: Optional[str] = None) -> Dict:
        """
        Answer a question using stored content with enhanced accuracy.
        
        Args:
            query (str): User question
            session_id (str, optional): Specific session to search
        
        Returns:
            Dict: Answer with sources and metadata
        """
        print(f"❓ Answering question: {query}")
        
        # 1. Analyze query complexity
        query_analysis = self.enhancer.analyze_query_complexity(query)
        print(f"🔍 Query complexity: {query_analysis['complexity_score']:.2f}, type: {query_analysis['question_type']}")
        
        # 1.5 Enhance query with preprocessing
        enhanced_query = self.preprocess_query(query)
        print(f"🔄 Enhanced query: '{enhanced_query}'")
        
        # 2. Process the enhanced query
        _, query_embedding = self.data_handler.process_query(enhanced_query)
        
        # 3. Search for relevant content with adaptive k
        search_results = self.storage_manager.search_content(
            query_embedding, 
            k=query_analysis['suggested_k'], 
            session_id=session_id
        )
        
        if not search_results:
            return {
                "query": query,
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "confidence": 0.0,
                "query_analysis": query_analysis,
                "original_query": query,
                "enhanced_query": enhanced_query
            }
        
        # 4. Re-rank search results for better accuracy
        enhanced_results = self.enhancer.rerank_search_results(query, search_results)
        print(f"🔄 Re-ranked {len(enhanced_results)} results")
        
        # 5. Get enhanced context
        enhanced_context = self.enhancer.enhance_context_selection(
            query, enhanced_results, self.config.MAX_CONTEXT_LENGTH
        )
        
        # 6. Analyze context quality
        context_suggestions = self.enhancer.suggest_context_improvements(
            query, enhanced_context
        )
        
        # 7. Generate answer using LLM
        if self.llm.is_available():
            answer_result = self.llm.generate_answer_with_context(query, enhanced_context, enhanced_results)
            
            # Add enhancement metadata
            answer_result.update({
                "search_results": enhanced_results,
                "context_used": len(enhanced_context),
                "sources": [result['metadata'] for result in enhanced_results],
                "query_analysis": query_analysis,
                "context_suggestions": context_suggestions,
                "enhancement_applied": True,
                "original_query": query,
                "enhanced_query": enhanced_query,
                "context_quality": {
                    "missing_keywords": context_suggestions.get('missing_keywords', []),
                    "redundant_content": context_suggestions.get('redundant_content', False),
                    "needs_more_context": context_suggestions.get('needs_more_context', False)
                }
            })
            
            # Run answer quality validation
            quality_metrics = self.validate_answer_quality(
                query, answer_result['answer'], enhanced_results
            )
            answer_result['answer_quality'] = quality_metrics
            
            # Print quality metrics
            print(f"📊 Answer quality score: {quality_metrics.get('overall_score', 0):.2f}")
            if quality_metrics.get('suggestions'):
                print("💡 Quality improvement suggestions:")
                for suggestion in quality_metrics.get('suggestions', []):
                    print(f"  - {suggestion}")
            
            return answer_result
        else:
            # Fallback without LLM
            return {
                "query": query,
                "answer": "LLM not available. Here's the enhanced context I found:",
                "context": enhanced_context[:1000] + "..." if len(enhanced_context) > 1000 else enhanced_context,
                "search_results": enhanced_results,
                "confidence": self._calculate_confidence(enhanced_results),
                "sources": [result['metadata'] for result in enhanced_results],
                "llm_available": False,
                "query_analysis": query_analysis,
                "context_suggestions": context_suggestions,
                "enhancement_applied": True,
                "original_query": query,
                "enhanced_query": enhanced_query,
                "context_quality": {
                    "missing_keywords": context_suggestions.get('missing_keywords', []),
                    "redundant_content": context_suggestions.get('redundant_content', False),
                    "needs_more_context": context_suggestions.get('needs_more_context', False)
                }
            }
    
    def get_pipeline_stats(self) -> Dict:
        """Get comprehensive pipeline statistics."""
        storage_stats = self.storage_manager.get_storage_stats()
        
        # Check OpenAI status
        openai_status = "✅ Ready" if self.llm.is_available() else "❌ Not configured"
        
        config_summary = self.config.get_summary()
        llm_status = self.llm.get_status()
        
        return {
            **storage_stats,
            "pipeline_ready": True,
            "components": {
                "scraper": "✅ Ready",
                "storage": "✅ Ready", 
                "data_handler": "✅ Ready",
                "openai_llm": openai_status
            },
            "openai_available": self.llm.is_available(),
            "configuration": config_summary,
            "llm_status": llm_status
        }
    
    def ask_question_with_llm(self, query: str, url: str = None, session_id: str = None) -> Dict:
        """
        Complete pipeline: process website (if needed) and answer question with LLM.
        
        Args:
            query (str): User question
            url (str, optional): Website URL to process (if not already processed)
            session_id (str, optional): Existing session ID to use or append to
        
        Returns:
            Dict: Complete answer with sources and metadata
        """
        print(f"🔍 Processing question: {query}")
        
        # If URL provided, process it first
        if url:
            print(f"🌐 Processing website: {url}")
            process_result = self.process_website(url, query, session_id)
            if not process_result.get('success'):
                return {
                    "error": "Failed to process website",
                    "details": process_result.get('error'),
                    "success": False
                }
            session_id = process_result['session_id']
        
        # Answer the question
        answer_result = self.answer_question(query, session_id)
        
        return {
            **answer_result,
            "session_id": session_id,
            "success": True
        }
    
    def _calculate_confidence(self, search_results: List[Dict]) -> float:
        """Calculate confidence score based on search results quality."""
        if not search_results:
            return 0.0
        
        scores = []
        for result in search_results:
            # Use rerank score if available, otherwise similarity score
            score = result.get('rerank_score', result.get('similarity_score', 0))
            scores.append(score)
        
        if not scores:
            return 0.0
        
        # Calculate weighted average with higher weight for top results
        weights = [1.0 / (i + 1) for i in range(len(scores))]
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def preprocess_query(self, query: str) -> str:
        """
        Advanced query preprocessing for better retrieval accuracy.
        
        Args:
            query (str): Raw user query
        
        Returns:
            str: Preprocessed and enhanced query
        """
        # 1. Expand contractions
        contractions = {
            "what's": "what is", "how's": "how is", "where's": "where is",
            "when's": "when is", "why's": "why is", "who's": "who is",
            "can't": "cannot", "won't": "will not", "shouldn't": "should not",
            "wouldn't": "would not", "couldn't": "could not", "doesn't": "does not",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "haven't": "have not", "hasn't": "has not",
            "hadn't": "had not", "didn't": "did not", "don't": "do not"
        }
        
        processed_query = query.lower()
        for contraction, expansion in contractions.items():
            processed_query = processed_query.replace(contraction, expansion)
        
        # 2. Get query type from enhancer
        query_analysis = self.enhancer.analyze_query_complexity(query)
        question_type = query_analysis.get('question_type', 'general')
        
        # 3. Add context keywords based on query type
        if question_type == 'procedural' or any(word in processed_query for word in ['how to', 'how do', 'steps']):
            processed_query += " tutorial guide instructions steps process methodology"
        elif question_type == 'definition' or any(word in processed_query for word in ['what is', 'define', 'definition']):
            processed_query += " explanation meaning concept definition terminology"
        elif question_type == 'causal' or any(word in processed_query for word in ['why', 'because', 'reason']):
            processed_query += " reason explanation cause effect result"
        elif question_type == 'comparative' or any(word in processed_query for word in ['compare', 'vs', 'versus', 'difference']):
            processed_query += " comparison differences pros cons advantages disadvantages"
        elif any(word in processed_query for word in ['best', 'recommend', 'should']):
            processed_query += " recommendation advice tips best practices optimal"
        
        # 4. Add entities found in the query
        entities = query_analysis.get('entities', [])
        if entities:
            entity_text = " ".join(entities)
            processed_query += f" {entity_text}"
        
        return processed_query

    def validate_answer_quality(self, query: str, answer: str, search_results: List[Dict]) -> Dict:
        """
        Validate the quality of generated answers.
        
        Args:
            query (str): Original query
            answer (str): Generated answer
            search_results (List[Dict]): Search results used
        
        Returns:
            Dict: Quality assessment with scores and suggestions
        """
        quality_metrics = {}
        
        # 1. Length appropriateness
        answer_length = len(answer.split())
        if answer_length < 10:
            quality_metrics['length_score'] = 0.3
            quality_metrics['length_issue'] = "Answer too short"
        elif answer_length > 500:
            quality_metrics['length_score'] = 0.7
            quality_metrics['length_issue'] = "Answer might be too long"
        else:
            quality_metrics['length_score'] = 1.0
            quality_metrics['length_issue'] = None
        
        # 2. Query term coverage in answer
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        coverage = len(query_words.intersection(answer_words)) / len(query_words) if query_words else 0
        quality_metrics['query_coverage'] = coverage
        
        # 3. Source attribution
        has_specific_info = any(word in answer.lower() for word in ['according to', 'based on', 'the document states'])
        quality_metrics['has_attribution'] = has_specific_info
        
        # 4. Answer structure
        has_structure = any(marker in answer for marker in [':', '•', '-', '1.', '2.', '#'])
        quality_metrics['has_structure'] = has_structure
        
        # 5. Confidence from search results
        if search_results:
            avg_score = np.mean([r.get('rerank_score', r.get('similarity_score', 0)) for r in search_results])
            quality_metrics['source_confidence'] = avg_score
        else:
            quality_metrics['source_confidence'] = 0.0
        
        # Overall quality score
        overall_score = (
            quality_metrics['length_score'] * 0.2 +
            quality_metrics['query_coverage'] * 0.3 +
            (1.0 if quality_metrics['has_attribution'] else 0.5) * 0.2 +
            (1.0 if quality_metrics['has_structure'] else 0.7) * 0.1 +
            quality_metrics['source_confidence'] * 0.2
        )
        
        quality_metrics['overall_score'] = overall_score
        
        # Generate improvement suggestions
        suggestions = []
        if quality_metrics['length_score'] < 0.5:
            suggestions.append("Consider providing more detailed information")
        if quality_metrics['query_coverage'] < 0.3:
            suggestions.append("Answer should better address the specific query terms")
        if not quality_metrics['has_attribution']:
            suggestions.append("Consider adding source references")
        if quality_metrics['source_confidence'] < 0.5:
            suggestions.append("Low source confidence - consider expanding search")
        
        quality_metrics['suggestions'] = suggestions
        
        return quality_metrics
