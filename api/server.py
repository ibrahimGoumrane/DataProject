from mcp.server.fastmcp import FastMCP
from rag_pipeline import RAGPipeline
from typing import List
import sys
import os

# Set UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Initialize the FastMCP server
mcp = FastMCP("urlContextHelper")

# Lazy initialization of RAG pipeline
rag_pipeline = None

def get_rag_pipeline():
    """Get RAG pipeline instance (lazy loading)"""
    global rag_pipeline
    if rag_pipeline is None:
        print("[INIT] Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline()
        print("[READY] RAG pipeline ready")
    return rag_pipeline

@mcp.tool("register_url")
async def register_url(url : str , contentToLookFFor : str , sessionId : str):
    """
    Register a URL with the RAG pipeline.
    """
    try:
        print(f"[REGISTER] URL: {url} for session: {sessionId}")
        pipeline = get_rag_pipeline()
        
        # Process the URL
        result = pipeline.process_website(url, contentToLookFFor, sessionId)
        
        return f"URL registered successfully. Processed {result.get('chunks_stored', 0)} chunks."
    except UnicodeEncodeError as e:
        print(f"[ERROR] Encoding error: {e}")
        return f"URL registration failed due to encoding error. Please check your content."
    except Exception as e:
        print(f"[ERROR] General error: {e}")
        return f"URL registration failed: {str(e)}"


@mcp.tool("search_url")
async def search_url(query: str, sessionId: str , useOwnContextOnly: bool = False , urls: List[str] = []):
    """
    Search for a URL using the RAG pipeline.
    """
    try:
        print(f"[SEARCH] Query: {query} in session: {sessionId}")
        pipeline = get_rag_pipeline()
        
        # Search for content
        search_session_id = sessionId if useOwnContextOnly else None
        results = pipeline.answer_question(query, search_session_id, urls)
        
        return results
    except UnicodeEncodeError as e:
        print(f"[ERROR] Encoding error: {e}")
        return {"error": "Search failed due to encoding error", "details": str(e)}
    except Exception as e:
        print(f"[ERROR] General error: {e}")
        return {"error": "Search failed", "details": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")