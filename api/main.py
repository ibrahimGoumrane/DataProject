"""
Test the RAG pipeline with enhanced context size (10,000 characters)
"""

from rag_pipeline import RAGPipeline
from config import get_config
from result_formatter import ResultFormatter
import os

def test_enhanced_context():
    """Test the RAG pipeline with enhanced context size."""
    # Initialize the pipeline
    pipeline = RAGPipeline()
    
    # Check the configuration
    config = get_config()
    
    # Initialize the result formatter
    formatter = ResultFormatter(output_dir="results")
    
    # Print the context size
    print(f"\n🔍 CONTEXT SIZE TEST")
    print(f"====================")
    print(f"Maximum context length: {config.MAX_CONTEXT_LENGTH} characters")
    print(f"Chunk size: {config.CHUNK_SIZE} characters")
    print(f"Chunk overlap: {config.CHUNK_OVERLAP} characters")
    print(f"Max tokens: {config.MAX_TOKENS} tokens")
    
    # Load a test HTML file (or use a live website)
    test_url = "https://laravel.com/docs/12.x/scheduling"
    query = " How does Laravel's task scheduling work and what are its key features?"
    # Process the website
    print(f"\n🌐 Processing website: {test_url}")
    result = pipeline.process_website(
        url=test_url,
        query=query
    )
    
    if not result.get('success', False):
        print(f"❌ Failed to process website: {result.get('error')}")
        return
    
    print(f"✅ Website processed successfully")
    print(f"Session ID: {result['session_id']}")
    print(f"Chunks stored: {result['chunks_stored']}")
    
    # Now ask a complex question that requires extensive context
    complex_query = "How do i create a scheduled task in Laravel, and what are the key features of its task scheduling system?"
    
    print(f"\n❓ Asking complex question: {complex_query}")
    answer = pipeline.answer_question(complex_query, result['session_id'])
    
    # Print the answer metadata to analyze context usage
    print(f"\n📊 CONTEXT USAGE ANALYSIS")
    print(f"=========================")
    print(f"Context length used: {answer.get('context_used', 0)} characters")
    print(f"Query complexity: {answer.get('query_analysis', {}).get('complexity_score', 0)}")
    print(f"Suggested k: {answer.get('query_analysis', {}).get('suggested_k', 0)}")
    print(f"Search results retrieved: {len(answer.get('search_results', []))}")
    
    # Extract the context provided to the LLM
    enhanced_context = pipeline.enhancer.enhance_context_selection(
        complex_query, 
        answer.get('search_results', []), 
        config.MAX_CONTEXT_LENGTH
    )
    
    # Add the context to the answer for saving
    answer['full_context'] = enhanced_context
    
    # Create the results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Use the result formatter to save results in both formats
    saved_files = formatter.save_both_formats(answer, "enhanced_context_test_result")
    
    # Additionally, save the raw context that was given to the LLM
    context_file_path = os.path.join("results", "enhanced_context_raw.txt")
    with open(context_file_path, "w", encoding="utf-8") as f:
        f.write(f"# Raw Context Provided to LLM\n\n")
        f.write(f"## Query\n{complex_query}\n\n")
        f.write(f"## Full Context ({len(enhanced_context)} characters)\n\n")
        f.write(enhanced_context)
    
    print(f"\n✅ Test complete!")
    print(f"Results saved to:")
    print(f"  - Markdown: {saved_files['markdown']}")
    print(f"  - HTML: {saved_files['html']}")
    print(f"  - Raw Context: {context_file_path}")

if __name__ == "__main__":
    test_enhanced_context()
