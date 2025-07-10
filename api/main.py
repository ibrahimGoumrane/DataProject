"""
Test the RAG pipeline with enhanced context size (10,000 characters)
and multi-website processing for broader context
"""

from rag_pipeline import RAGPipeline
from config import get_config
from result_formatter import ResultFormatter
config = get_config()
# Initialize the result formatter
formatter = ResultFormatter(output_dir="results")
# Initialize the RAG pipeline
pipeline = RAGPipeline()

def test_enhanced_context():
    """Test the RAG pipeline with enhanced context size."""

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
    
    # Use the result formatter to save results in both formats
    saved_files = formatter.save_both_formats(answer, "enhanced_context_test_result")
    
    print(f"\n✅ Test complete!")
    print(f"Results saved to:")
    print(f"  - Markdown: {saved_files['markdown']}")
    print(f"  - HTML: {saved_files['html']}")

def test_multi_website_context():
    """Test processing multiple websites under a single session ID."""

    # Print the context size
    print(f"\n🔍 MULTI-WEBSITE CONTEXT TEST")
    print(f"============================")
    print(f"Maximum context length: {config.MAX_CONTEXT_LENGTH} characters")
    
    # Process the first website
    first_url = "https://laravel.com/docs/12.x/scheduling"
    first_query = "How does Laravel's task scheduling work?"
    
    print(f"\n🌐 Processing first website: {first_url}")
    first_result = pipeline.process_website(
        url=first_url,
        query=first_query
    )
    
    if not first_result.get('success', False):
        print(f"❌ Failed to process first website: {first_result.get('error')}")
        return
    
    print(f"✅ First website processed successfully")
    print(f"Session ID: {first_result['session_id']}")
    print(f"Chunks stored: {first_result['chunks_stored']}")
    
    # Store the session ID to use for the second website
    combined_session_id = first_result['session_id']
    
    # Process a second related website and add to the same session
    second_url = "https://laravel.com/docs/12.x/queues"
    second_query = "How do Laravel queues work with scheduled tasks?"
    
    print(f"\n🌐 Processing second website: {second_url}")
    print(f"📌 Appending to session: {combined_session_id}")
    
    second_result = pipeline.process_website(
        url=second_url,
        query=second_query,
        session_id=combined_session_id  # Reuse the same session ID
    )
    
    if not second_result.get('success', False):
        print(f"❌ Failed to process second website: {second_result.get('error')}")
        return
    
    print(f"✅ Second website processed successfully")
    print(f"Combined session ID: {second_result['session_id']}")
    print(f"Additional chunks stored: {second_result['chunks_stored']}")
    
    # Now ask a complex question that requires context from both websites
    complex_query = "How can I create a scheduled task in Laravel that dispatches jobs to a queue, and what are the advantages of this approach?"
    
    print(f"\n❓ Asking question requiring context from both websites: {complex_query}")
    answer = pipeline.answer_question(complex_query, combined_session_id)
    
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
    
    # Use the result formatter to save results in both formats
    saved_files = formatter.save_both_formats(answer, "multi_website_context_test")
    
    print(f"\n✅ Test complete!")
    print(f"Results saved to:")
    print(f"  - Markdown: {saved_files['markdown']}")
    print(f"  - HTML: {saved_files['html']}")

if __name__ == "__main__":
    print("Choose a test to run:")
    print("1. Enhanced Context Test (single website)")
    print("2. Multi-Website Context Test")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        test_enhanced_context()
    elif choice == "2":
        test_multi_website_context()
    else:
        print("Invalid choice. Please run again and select 1 or 2.")
