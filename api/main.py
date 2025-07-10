"""
Test RAG Pipeline with Enhanced Output Formatting
Saves results in both Markdown and HTML formats for better readability.
"""
from rag_pipeline import RAGPipeline
from result_formatter import ResultFormatter
import os

if __name__ == "__main__":
    query = "Which file do I need to edit to add custom styles in Tailwind CSS?"
    url = "https://tailwindcss.com/docs/adding-custom-styles"

    print("🚀 Starting RAG Pipeline with Enhanced Formatting")
    print(f"📝 Query: {query}")
    print(f"🌐 URL: {url}")
    print("=" * 60)

    # Initialize pipeline and formatter
    rag_pipeline = RAGPipeline()
    formatter = ResultFormatter()

    # Process the website
    print("🔄 Processing website...")
    result = rag_pipeline.process_website(url=url, query=query)

    if result.get("success"):
        print(f"✅ Website processed successfully!")
        print(f"📊 Session ID: {result.get('session_id')}")
        print(f"📦 Chunks stored: {result.get('chunks_stored')}")

        # Get the answer to the query
        print("\n🤖 Generating answer with LLM...")
        answer = rag_pipeline.answer_question(query=query, session_id=result.get("session_id"))

        if answer.get("llm_available"):
            print(f"✅ LLM Response Generated!")
            print(f"🧠 Model: {answer.get('llm_model')}")
            print(f"🎯 Tokens Used: {answer.get('tokens_used')}")
            print(f"📊 Confidence: {answer.get('confidence'):.2f}")
            
            # Display answer preview
            answer_preview = answer.get('answer', '')[:200]
            print(f"\n📝 Answer Preview:\n{answer_preview}...")
            
            # Save results in both formats
            print("\n💾 Saving results...")
            saved_files = formatter.save_both_formats(answer, "tailwind_css_customization")
            
            print(f"✅ Results saved:")
            print(f"  📄 Markdown: {saved_files['markdown']}")
            print(f"  🌐 HTML: {saved_files['html']}")
            
            # Show file sizes
            for format_name, filepath in saved_files.items():
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    print(f"     {format_name.upper()}: {size:,} bytes")
            
        else:
            print(f"❌ LLM not available: {answer.get('error')}")
            
            # Still save the context-based result
            print("💾 Saving context-based result...")
            saved_files = formatter.save_both_formats(answer, "tailwind_css_context_only")
            print(f"📄 Fallback results saved: {saved_files}")

    else:
        print(f"❌ Failed to process website: {result.get('error')}")

    print(f"\n🎉 Process completed! Check the 'results' folder for output files.")

     