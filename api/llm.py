"""
LLM Module for RAG Pipeline
Handles all language model interactions including prompt engineering and response generation.
"""

from typing import Dict, List, Optional
from config import get_config
from openai import OpenAI



class LLM:
    """
    Language Model handler for the RAG pipeline.
    Manages OpenAI API interactions, prompt engineering, and response generation.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM with configuration.
        
        Args:
            api_key (str, optional): OpenAI API key. If not provided, uses config.
        """
        self.config = get_config()
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=api_key or self.config.OPENAI_API_KEY)
        self.modal = self.config.MODEL_NAME

    def is_available(self) -> bool:
        """Check if LLM is available for use."""
        return self.openai_client is not None
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Dict:
        """
        Generate a response from the language model.
        
        Args:
            prompt (str): The user prompt
            system_prompt (str, optional): System prompt to set context
        
        Returns:
            Dict: Response with metadata
        """
        if not self.is_available():
            return {
                "response": "LLM not available. Please configure OpenAI API key.",
                "success": False,
                "error": "LLM not configured"
            }
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.openai_client.responses.create(
                model=self.modal,
                input=messages,
                temperature=self.config.TEMPERATURE,
                max_output_tokens=self.config.MAX_TOKENS
            )
            return {
                "response": response.output_text.strip(),
                "success": True,
                "model": self.config.MODEL_NAME,
                "tokens_used": response.usage.total_tokens if response.usage else None
            }
            
        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def generate_answer_with_context(self, query: str, context: str, search_results: List[Dict]) -> Dict:
        """
        Generate an answer using OpenAI LLM with enhanced prompt and context.
        
        Args:
            query (str): User question
            context (str): Retrieved context
            search_results (List[Dict]): Search results with metadata
        
        Returns:
            Dict: Generated answer with metadata
        """
        if not self.is_available():
            return {
                "query": query,
                "answer": "LLM not available. Please configure OpenAI API key.",
                "confidence": 0.0,
                "llm_available": False,
                "error": "LLM not configured"
            }
        
        try:
            # Create enhanced prompt with context and instructions
            enhanced_prompt = self.create_enhanced_prompt(query, context, search_results)
            system_prompt = self.get_system_prompt()
            
            # Generate response
            response_result = self.generate_response(enhanced_prompt, system_prompt)
            
            if response_result['success']:
                return {
                    "query": query,
                    "answer": response_result['response'],
                    "confidence": self._calculate_confidence(search_results),
                    "llm_model": response_result['model'],
                    "llm_available": True,
                    "tokens_used": response_result.get('tokens_used')
                }
            else:
                return {
                    "query": query,
                    "answer": f"Sorry, I encountered an error generating the answer: {response_result['error']}",
                    "confidence": 0.0,
                    "llm_available": False,
                    "error": response_result['error']
                }
                
        except Exception as e:
            print(f"❌ LLM generation error: {e}")
            return {
                "query": query,
                "answer": f"Sorry, I encountered an error generating the answer: {str(e)}",
                "confidence": 0.0,
                "llm_available": False,
                "error": str(e)
            }
    
    def create_enhanced_prompt(self, query: str, context: str, search_results: List[Dict]) -> str:
        """
        Create an enhanced prompt with context and search results.
        
        Args:
            query (str): User question
            context (str): Retrieved context
            search_results (List[Dict]): Search results
        
        Returns:
            str: Enhanced prompt for the LLM
        """
        # Limit context length to avoid token limits
        if len(context) > self.config.MAX_CONTEXT_LENGTH:
            context = context[:self.config.MAX_CONTEXT_LENGTH] + "..."
        
        # Extract source URLs for citations
        sources = []
        for result in search_results[:3]:  # Top 3 sources
            if 'source_url' in result['metadata']:
                sources.append(result['metadata']['source_url'])
        
        sources_text = "\n".join([f"- {url}" for url in set(sources)]) if sources else "- No specific sources available"
        
        prompt = f"""
QUESTION: {query}

RELEVANT CONTEXT:
{context}

SOURCES:
{sources_text}

Please provide a comprehensive and accurate answer to the question based on the context provided. 

Instructions:
1. Format your response in clean Markdown with proper headings, bullet points, and code blocks where appropriate
2. Answer directly and concisely using only the information from the provided context
3. If the context doesn't contain enough information, say so clearly
4. Include relevant details and examples when available
5. Use proper Markdown formatting:
   - Use ## for main headings
   - Use ### for subheadings
   - Use bullet points (-) for lists
   - Use `code blocks` for technical terms
   - Use **bold** for emphasis
   - Use > for important quotes or notes
6. Don't make up information that's not in the context
7. Structure your response with clear sections if the topic is complex
8. Always give more priority to the provided context over general knowledge.
9. Always choose the most relevant and recent information from the context.
Answer in Markdown format:"""
        
        return prompt
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the LLM.
        
        Returns:
            str: System prompt
        """
        return """You are a helpful AI assistant that answers questions based on provided context from web content. 

Your role:
- Answer questions accurately using only the provided context
- Be concise but comprehensive
- Acknowledge when information is insufficient
- Structure answers clearly using proper Markdown formatting
- Use headings, bullet points, code blocks, and emphasis appropriately
- Don't hallucinate or add information not in the context
- If asked about something not in the context, politely explain the limitation

Always format your responses in clean, readable Markdown and base your answers on the provided context while citing limitations when appropriate."""
    
    def _calculate_confidence(self, search_results: List[Dict]) -> float:
        """
        Calculate confidence score based on similarity scores.
        
        Args:
            search_results (List[Dict]): Search results with similarity scores
        
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        if not search_results:
            return 0.0
        
        # Import numpy here to avoid dependency issues
        try:
            import numpy as np
            scores = [result['similarity_score'] for result in search_results]
            return float(np.mean(scores))
        except ImportError:
            # Fallback without numpy
            scores = [result['similarity_score'] for result in search_results]
            return sum(scores) / len(scores) if scores else 0.0
    
    def get_status(self) -> Dict:
        """
        Get LLM status and configuration.
        
        Returns:
            Dict: Status information
        """
        return {
            "available": self.is_available(),
            "api_key_configured": bool(self.config.OPENAI_API_KEY),
            "model": self.config.MODEL_NAME,
            "max_tokens": self.config.MAX_TOKENS,
            "temperature": self.config.TEMPERATURE,
            "max_context_length": self.config.MAX_CONTEXT_LENGTH
        }