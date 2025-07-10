import numpy as np
import os
from scraper import Scraper
from dataHandler import DataHandler
from storage_manager import StorageManager
from config import get_config, load_env_file
from llm import LLM
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
        
        # Initialize LLM
        self.llm = LLM(openai_api_key)
        
        print("🤖 RAG Pipeline initialized")
        print(f"📊 Configuration: max_context={self.config.MAX_CONTEXT_LENGTH}, model={self.config.MODEL_NAME}")
        print(f"🔑 OpenAI: {'✅ Available' if self.llm.is_available() else '❌ Not configured'}")
    
    def process_website(self, url: str, query: str) -> Dict:
        """
        Complete pipeline: scrape website and store in vector database.
        
        Args:
            url (str): Website URL to scrape
            query (str): User query for context
        
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
            session_id = self.storage_manager.store_scrape_result(scrape_result)
            
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
        Answer a question using stored content.
        
        Args:
            query (str): User question
            session_id (str, optional): Specific session to search
        
        Returns:
            Dict: Answer with sources and metadata
        """
        print(f"❓ Answering question: {query}")
        
        # 1. Process the query
        _, query_embedding = self.data_handler.process_query(query)
        
        # 2. Search for relevant content
        search_results = self.storage_manager.search_content(
            query_embedding, 
            k=self.config.TOP_K_RESULTS, 
            session_id=session_id
        )
        
        if not search_results:
            search_results = [{"metadata": {"source_url": "No results found"}}]
            context = "No relevant content found."
        else :
            # 3. Get context for generation
            context = self.storage_manager.get_context_for_query(query_embedding)
            
        # 4. Generate answer using LLM
        if self.llm.is_available():
            answer_result = self.llm.generate_answer_with_context(query, context, search_results)
            return {
                **answer_result,
                "search_results": search_results,
                "context_used": len(context),
                "sources": [result['metadata'] for result in search_results]
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
            session_id (str, optional): Existing session ID to use
        
        Returns:
            Dict: Complete answer with sources and metadata
        """
        print(f"🔍 Processing question: {query}")
        
        # If URL provided, process it first
        if url:
            print(f"🌐 Processing website: {url}")
            process_result = self.process_website(url, query)
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
