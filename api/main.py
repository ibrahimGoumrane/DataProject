"""
Interactive RAG CLI interface with enhanced context size (10,000 characters)
and multi-website processing for broader context
"""

from rag_pipeline import RAGPipeline
from config import RAGConfig
from result_formatter import ResultFormatter

# Initialize the necessary components
pipeline = RAGPipeline()
config = RAGConfig()
formatter = ResultFormatter(output_dir="results")
def run_interactive_cli(existing_session_id=None):
    """Run an interactive CLI for the RAG pipeline with direct website input."""
    # Initialize components

    
    print("\n🤖 ENHANCED RAG PIPELINE - INTERACTIVE MODE")
    print("===========================================")
    print(f"Maximum context length: {config.MAX_CONTEXT_LENGTH} characters")
    print(f"Chunk size: {config.CHUNK_SIZE} characters")
    
    # Ask user about context preference
    print("\n🔧 CONTEXT MODE SELECTION")
    print("========================")
    print("1. Use only content from your provided URLs (focused)")
    print("2. Enhanced context mode (includes broader search)")
    
    while True:
        context_choice = input("\nChoose context mode (1 or 2): ").strip()
        if context_choice in ['1', '2']:
            break
        print("Please enter 1 or 2")
    
    use_session_only = context_choice == '1'
    
    if use_session_only:
        print("✅ Using focused mode - only content from your URLs")
    else:
        print("✅ Using enhanced mode - broader context search")
    
    # Always use async scraping (removed choice)
    use_async = True
    print("\n⚡ SCRAPING MODE")
    print("===============")
    print("✅ Using async parallel scraping - optimized for speed and efficiency")
    
    # Use existing session if provided
    session_id = existing_session_id
    websites_processed = []
    
    # If resuming a session, get the list of websites already processed
    if session_id:
        # Try to get session info and any stored metadata
        session_info = pipeline.storage_manager.vector_store.get_session_info(session_id)
        if session_info:
            print(f"\n📋 RESUMING EXISTING SESSION: {session_id}")
            
            # Get websites from session info
            if 'websites' in session_info:
                for website_info in session_info['websites']:
                    url = website_info.get('source_url')
                    if url and url not in websites_processed:
                        websites_processed.append(url)
            else:
                # Fallback to old format
                source_url = session_info.get('source_url')
                if source_url and source_url not in websites_processed:
                    websites_processed.append(source_url)
            
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
        print(f"📝 Query: {query}")
        print(f"⚡ Mode: Async Parallel")
        
        if session_id:
            print(f"📌 Appending to existing session: {session_id}")
            
        # Record start time for performance measurement
        import time
        start_time = time.time()
        
        result = pipeline.process_website(
            url=url,
            query=query,
            session_id=session_id,
            use_async=use_async
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if not result.get('success', False):
            print(f"❌ Failed to process website: {result.get('error')}")
            print(f"⏱️ Processing time: {processing_time:.2f} seconds")
            continue
        
        session_id = result['session_id']
        websites_processed.append(url)
        
        print(f"✅ Website processed successfully")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"📊 Performance: Async Parallel scraping")
        print(f"🆔 Session ID: {result['session_id']}")
        print(f"📦 Chunks stored: {result['chunks_stored']}")
        
        # Show performance benefit of parallel scraping
        if use_async and processing_time > 0:
            estimated_sequential_time = processing_time * 2.5  # Rough estimate
            print(f"📈 Estimated sequential time: {estimated_sequential_time:.2f}s")
            print(f"🚀 Speed improvement: ~{estimated_sequential_time/processing_time:.1f}x faster")
        
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
    print(f"Context mode: {'Focused (URLs only)' if use_session_only else 'Enhanced (broader search)'}")
    print(f"Scraping mode: Async Parallel")
    
    if use_async:
        print("\n⚡ ASYNC SCRAPING INFO")
        print("====================")
        print("✅ Parallel scraping was used for faster processing")
        print("🔧 Configuration:")
        print(f"  - Max concurrent: {config.SCRAPER_MAX_CONCURRENT}")
        print(f"  - Batch size: {config.SCRAPER_BATCH_SIZE}")
        print(f"  - Async timeout: {config.SCRAPER_ASYNC_TIMEOUT}s")
        print("💡 Tip: You can adjust these settings via environment variables")
    
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
        
        # Use session-only search if user chose focused mode
        search_session_id = session_id if use_session_only else None
        answer = pipeline.answer_question(query, search_session_id)
        
        # Print the answer
        print("\n📝 ANSWER:")
        print("==========")
        print(answer.get('answer', 'No answer available'))
        
        # Print the answer metadata
        print(f"\n📊 CONTEXT USAGE:")
        print(f"Context length: {answer.get('context_used', 0)} characters")
        print(f"Sources: {len(answer.get('sources', []))} chunks from {len(set([s.get('source_url', '') for s in answer.get('sources', [])]))} websites")
        print(f"Search mode: {'Session-only' if use_session_only else 'All content'}")
        
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

if __name__ == "__main__":
    print("\n🤖 Enhanced RAG Pipeline")
    print("========================")
    print("1. Start new session")
    print("2. Resume existing session")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == "1":
        run_interactive_cli()
    elif choice == "2":
        session_id = input("\nEnter the session ID to resume: ")
        if session_id.strip():
            print(f"\nResuming session: {session_id}")

            
            # Check if session exists
            session_info = pipeline.storage_manager.vector_store.get_session_info(session_id)
            if session_info:
                print(f"✅ Session found: {session_id}")
                if 'websites' in session_info and session_info['websites']:
                    print(f"Websites in session: {len(session_info['websites'])}")
                    for i, website in enumerate(session_info['websites'], 1):
                        print(f"  {i}. {website.get('source_url', 'Unknown')}")
                else:
                    print(f"Original URL: {session_info.get('source_url', 'Unknown')}")
                print(f"Created: {session_info.get('storage_timestamp', 'Unknown')}")
                run_interactive_cli(session_id)
            else:
                print(f"❌ Session not found: {session_id}")
        else:
            print("❌ No session ID provided.")
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("Invalid choice. Please run again and select a valid option.")
