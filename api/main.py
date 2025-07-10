"""
Interactive RAG CLI interface with enhanced context size (10,000 characters)
and multi-website processing for broader context
"""

from rag_pipeline import RAGPipeline
from config import get_config
from result_formatter import ResultFormatter


def run_interactive_cli(existing_session_id=None):
    """Run an interactive CLI for the RAG pipeline with direct website input."""
    # Initialize components
    config = get_config()
    formatter = ResultFormatter(output_dir="results")
    pipeline = RAGPipeline()
    
    print("\n🤖 ENHANCED RAG PIPELINE - INTERACTIVE MODE")
    print("===========================================")
    print(f"Maximum context length: {config.MAX_CONTEXT_LENGTH} characters")
    print(f"Chunk size: {config.CHUNK_SIZE} characters")
    
    # Use existing session if provided
    session_id = existing_session_id
    websites_processed = []
    
    # If resuming a session, get the list of websites already processed
    if session_id:
        # Try to get session info and any stored metadata
        session_info = pipeline.storage_manager.vector_store.get_session_info(session_id)
        if session_info:
            print(f"\n📋 RESUMING EXISTING SESSION: {session_id}")
            source_url = session_info.get('source_url')
            if source_url and source_url not in websites_processed:
                websites_processed.append(source_url)
                print(f"Found website in session: {source_url}")
            
            # Try to find other websites associated with this session
            for metadata in pipeline.storage_manager.vector_store.get_session_metadata(session_id):
                url = metadata.get('source_url')
                if url and url not in websites_processed:
                    websites_processed.append(url)
            
            print(f"Websites already processed in this session: {len(websites_processed)}")
            for i, website in enumerate(websites_processed, 1):
                print(f"  {i}. {website}")
    
    # Step 1: Process websites
    print("\n📚 WEBSITE PROCESSING")
    print("===================")
    print("Enter the websites you want to process (enter an empty URL when finished):")
    
    website_num = 1
    while True:
        url = input(f"\nWebsite {website_num} URL (or press Enter to finish): ")
        if not url.strip():
            if website_num == 1 and not websites_processed:
                print("⚠️ You need to process at least one website.")
                continue
            else:
                break
                
        query = input(f"Focus query for {url}: ")
        if not query.strip():
            query = "What is this page about?"
            print(f"Using default query: '{query}'")
            
        print(f"\n🌐 Processing website: {url}")
        if session_id:
            print(f"📌 Appending to existing session: {session_id}")
            
        result = pipeline.process_website(
            url=url,
            query=query,
            session_id=session_id
        )
        
        if not result.get('success', False):
            print(f"❌ Failed to process website: {result.get('error')}")
            continue
        
        session_id = result['session_id']
        websites_processed.append(url)
        
        print(f"✅ Website processed successfully")
        print(f"Session ID: {result['session_id']}")
        print(f"Chunks stored: {result['chunks_stored']}")
        
        website_num += 1
        
        # Ask if user wants to process another website
        if website_num > 3:  # After 3 websites, explicitly confirm
            another = input("\nProcess another website? (y/n): ").lower()
            if another != 'y':
                break
    
    # Step 2: Display summary of processed websites
    print("\n📋 PROCESSED WEBSITES SUMMARY")
    print("===========================")
    print(f"Total websites processed: {len(websites_processed)}")
    for i, website in enumerate(websites_processed, 1):
        print(f"  {i}. {website}")
    print(f"Session ID: {session_id}")
    
    # Step 3: Ask questions
    print("\n❓ ASK QUESTIONS")
    print("==============")
    print("Now you can ask questions about the processed websites (enter an empty question to exit):")
    
    question_num = 1
    while True:
        query = input(f"\nQuestion {question_num}: ")
        if not query.strip():
            break
            
        print(f"\n🔍 Searching for answer to: {query}")
        answer = pipeline.answer_question(query, session_id)
        
        # Print the answer
        print("\n📝 ANSWER:")
        print("==========")
        print(answer.get('answer', 'No answer available'))
        
        # Print the answer metadata
        print(f"\n📊 CONTEXT USAGE:")
        print(f"Context length: {answer.get('context_used', 0)} characters")
        print(f"Sources: {len(answer.get('sources', []))} chunks from {len(set([s.get('source_url', '') for s in answer.get('sources', [])]))} websites")
        
        # Ask if user wants to save the result
        save_result = input("\nSave this result? (y/n): ").lower()
        if save_result == 'y':
            result_name = f"result_{question_num}"
            
            # Extract the context provided to the LLM
            enhanced_context = pipeline.enhancer.enhance_context_selection(
                query, 
                answer.get('search_results', []), 
                config.MAX_CONTEXT_LENGTH
            )
            
            # Add the context to the answer for saving
            answer['full_context'] = enhanced_context
            
            # Save results
            saved_files = formatter.save_both_formats(answer, result_name)
            print(f"Results saved to:")
            print(f"  - Markdown: {saved_files['markdown']}")
            print(f"  - HTML: {saved_files['html']}")
        
        question_num += 1
        
        # After 3 questions, confirm if they want to continue
        if question_num > 3 and question_num % 3 == 1:
            another = input("\nAsk another question? (y/n): ").lower()
            if another != 'y':
                break
    
    # Exit message
    print("\n👋 Thank you for using the Enhanced RAG Pipeline!")
    print(f"Session ID: {session_id} (Save this if you want to resume this session later)")
    
    return session_id

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
    query = " How does Laravel's task scheduling work and what are the key features?"
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
    # Initialize the pipeline
    pipeline = RAGPipeline()
    
    # Check the configuration
    config = get_config()
    
    # Initialize the result formatter
    formatter = ResultFormatter(output_dir="results")
    
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
    print("\nEnhanced RAG Pipeline with Multi-Website Support")
    print("===============================================")
    print("1. Interactive CLI Mode (New Session)")
    print("2. Resume Existing Session")
    print("3. Enhanced Context Test (single website)")
    print("4. Multi-Website Context Test")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ")
    
    if choice == "1":
        run_interactive_cli()
    elif choice == "2":
        session_id = input("\nEnter the session ID to resume: ")
        if session_id.strip():
            print(f"\nResuming session: {session_id}")
            # Initialize the necessary components
            pipeline = RAGPipeline()
            
            # Check if session exists
            session_info = pipeline.storage_manager.vector_store.get_session_info(session_id)
            if session_info:
                print(f"✅ Session found: {session_id}")
                print(f"Original URL: {session_info.get('source_url', 'Unknown')}")
                print(f"Original query: {session_info.get('query', 'Unknown')}")
                print(f"Created: {session_info.get('storage_timestamp', 'Unknown')}")
                run_interactive_cli()
            else:
                print(f"❌ Session not found: {session_id}")
        else:
            print("❌ No session ID provided.")
    elif choice == "3":
        test_enhanced_context()
    elif choice == "4":
        test_multi_website_context()
    elif choice == "5":
        print("Exiting...")
    else:
        print("Invalid choice. Please run again and select a valid option.")
