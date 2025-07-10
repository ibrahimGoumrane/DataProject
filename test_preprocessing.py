"""
Test the enhanced DataHandler preprocessing capabilities
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.dataHandler import DataHandler

def test_preprocessing():
    """Test the new preprocessing features"""
    
    # Sample HTML content (simulating scraped content)
    sample_html = """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <nav>Navigation menu here</nav>
        <h1>Machine Learning Fundamentals</h1>
        <p>Machine learning is a subset of artificial intelligence that focuses on algorithms 
        that can learn from data. It involves training models on datasets to make predictions 
        or decisions without being explicitly programmed for every scenario.</p>
        
        <h2>Types of Machine Learning</h2>
        <p>There are three main types: supervised learning, unsupervised learning, and 
        reinforcement learning. Supervised learning uses labeled data to train models.</p>
        
        <p>Unsupervised learning finds patterns in data without labels. Reinforcement 
        learning uses reward systems to improve performance over time.</p>
        
        <footer>Copyright 2024</footer>
        <script>console.log('some javascript');</script>
    </body>
    </html>
    """
    
    handler = DataHandler()
    
    print("🧪 Testing Enhanced Preprocessing")
    print("=" * 50)
    
    # Test 1: HTML cleaning
    print("\n1. HTML Cleaning:")
    clean_text = handler.clean_html_text(sample_html)
    print(f"Cleaned text length: {len(clean_text)}")
    print(f"Preview: {clean_text[:200]}...")
    
    # Test 2: Text chunking
    print("\n2. Text Chunking:")
    chunks = handler.chunk_text(clean_text, chunk_size=150)
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk[:100]}...")
    
    # Test 3: Content processing
    print("\n3. Content Processing:")
    processed_chunks, embeddings = handler.process_html(sample_html)
    print(f"Processed chunks: {len(processed_chunks)}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test 4: Query similarity
    print("\n4. Query Similarity:")
    query = "What is machine learning?"
    similar_chunks = handler.compute_similarity(sample_html, query, top_k=3)
    
    print(f"Query: '{query}'")
    print("Top similar chunks:")
    for i, (chunk, score) in enumerate(similar_chunks):
        print(f"  {i+1}. Score: {score:.3f}")
        print(f"     Text: {chunk[:100]}...")
    
    # Test 5: Context extraction
    print("\n5. Relevant Context:")
    context = handler.get_relevant_context(sample_html, query, max_context_length=500)
    print(f"Context length: {len(context)}")
    print(f"Context: {context[:300]}...")
    
    print("\n✅ All preprocessing tests completed!")

if __name__ == "__main__":
    # Download required NLTK data first
    import nltk
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("📥 NLTK data downloaded")
    except:
        print("⚠️  NLTK download failed - you may need to run this manually")
    
    test_preprocessing()
