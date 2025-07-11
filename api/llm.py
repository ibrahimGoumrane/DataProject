"""
LLM Module for RAG Pipeline
Handles all language model interactions including prompt engineering and response generation.
"""

from typing import Dict, List, Optional
from config import RAGConfig
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
        self.config = RAGConfig()
        
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
        
        # Extract metadata for analysis
        metadata_list = []
        for result in search_results[:5]:  # Top 5 sources
            if 'metadata' in result:
                metadata_list.append(result['metadata'])
        
        # Get question type hints
        question_type = ""
        if search_results and 'query_analysis' in search_results[0]:
            question_type = search_results[0]['query_analysis'].get('question_type', '')
            
        # Add specific instructions based on question type
        type_specific_instructions = ""
        if question_type == 'procedural':
            type_specific_instructions = "Since this is a procedural question, organize your answer as a step-by-step guide with clear instructions based on the context."
        elif question_type == 'definition':
            type_specific_instructions = "Since this is a definition question, start with a clear, concise definition based on the context before providing more details."
        elif question_type == 'comparative':
            type_specific_instructions = "Since this is a comparative question, organize your answer to clearly contrast the items being compared using information from the context."
        
        prompt = f"""
QUESTION: {query}

RELEVANT CONTEXT:
{context}

CRITICAL INSTRUCTIONS - CONTEXT DEPENDENCY:
You MUST base your answer primarily on the provided context above. Follow this strict hierarchy:
- 80% of your answer MUST come from the provided context
- Only 20% can come from your general knowledge to fill minor gaps
- If the context doesn't contain sufficient information, clearly state what's missing rather than inventing details
- NEVER provide information that contradicts the context
- Always indicate when you're using general knowledge vs. context information

FORMATTING REQUIREMENTS:
1. Format your response in clean Markdown with proper headings, bullet points, and code blocks where appropriate
2. Use ## for main headings, ### for subheadings
3. Use bullet points (-) for lists
4. Use `code blocks` for technical terms
5. Use **bold** for emphasis
6. Use > for important quotes or notes from the context
7. Structure your response with clear sections if the topic is complex
{type_specific_instructions}

ANSWER STRUCTURE:
- Start with information directly from the context
- Clearly mark any general knowledge additions with phrases like "Based on general knowledge:" or "Additionally, from standard practice:"
- If context is insufficient, state: "The provided context doesn't contain information about [specific topic]"
- Provide only what can be reliably derived from the context

Answer in Markdown format based PRIMARILY on the provided context:"""
        
        return prompt
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the LLM.
        
        Returns:
            str: System prompt
        """
        return """You are a context-dependent AI assistant that answers questions based PRIMARILY on provided context.

STRICT GUIDELINES:
- 80% of your response MUST be based on the provided context
- Only 20% can come from general knowledge to fill minor gaps
- You are NOT a general knowledge assistant - you are a context-based assistant
- When context is insufficient, clearly state what's missing rather than providing general information
- NEVER contradict or override information from the context
- Always distinguish between context-based information and general knowledge

RESPONSE REQUIREMENTS:
- Be transparent about information sources (context vs. general knowledge)
- If context doesn't contain needed information, state: "The provided context doesn't include information about [topic]"
- Provide comprehensive answers ONLY when context supports it
- Use proper Markdown formatting for clarity
- Focus on practical, actionable information from the context
- When general knowledge is used, clearly mark it as such

Remember: You are primarily a context interpreter, not a general knowledge provider."""
    
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