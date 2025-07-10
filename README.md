# 🧠 Ask a Website Anything - Live Web Retrieval RAG Chatbot

> A powerful chatbot that allows users to input any website URL, dynamically crawls and extracts content, embeds it into a vector store, and answers natural language queries based on the **live content** of that website using a language model.

## 🎯 Project Goal

Build an intelligent chatbot system that can:

- Accept any website URL as input
- Dynamically crawl and extract content (HTML pages, subpages, code blocks)
- Embed content into a vector store for efficient retrieval
- Answer natural language queries based on live website content
- Provide contextual responses with source attribution

## 📐 System Architecture

```plaintext
[User Input URL] ---> [Web Scraper]
                          |
                          v
                [HTML Cleaner + Chunker]
                          |
                          v
               [Embedder → Vector Store]
                          |
                          v
[User Query] ---> [Retriever + Prompt Builder] ---> [LLM]
                          |
                          v
                   [Final Answer Output]
```

## 🧱 Project Structure

```
ask-website-anything/
│
├── backend/
│   ├── scraper.py           # Crawl website and extract raw HTML/text
│   ├── parser.py            # Clean HTML, extract code/text, chunk
│   ├── embedder.py          # Embed & store chunks in FAISS
│   ├── retriever.py         # Similarity search on vector store
│   ├── generator.py         # Build prompt & get LLM response
│   ├── api.py               # FastAPI app (optional)
│
├── interface/
│   ├── cli.py               # Terminal interface (MVP)
│   └── streamlit_app.py     # Streamlit web frontend
│
├── data/
│   └── (Temporary scraped pages and embeddings)
│
├── requirements.txt
└── README.md
```

## 📆 Development Phases

### ✅ Phase 1: Basic Functionality (MVP)

**Goal:** Build a CLI chatbot that takes a URL and answers based on that site.

1. **🌐 Website Scraper**

   - Use `requests`, `BeautifulSoup`, and `urljoin` to crawl the base page and linked subpages
   - Respect same domain / depth limits

2. **🧹 HTML Cleaner + Chunker**

   - Clean HTML: strip navbars, footers, scripts
   - Extract content blocks (headers, code blocks, lists)
   - Chunk into 300–500 token segments

3. **🔐 Embedding + Storage**

   - Use `sentence-transformers` (e.g., `"all-MiniLM-L6-v2"`) or OpenAI embeddings
   - Store vectors in FAISS or ChromaDB

4. **🔍 Query + Retrieval**

   - Convert user question into embedding
   - Retrieve top K chunks using cosine similarity

5. **🧠 Prompt & LLM Answer**

   - Construct prompt: `Context:\n... + Question: ...`
   - Call OpenAI (or local model like Mistral) to get an answer

6. **🖥️ CLI Interface**
   - Build a simple command-line interface:
   ```bash
   python cli.py
   > Enter URL: https://tailwindcss.com/docs
   > Enter question: How do I add custom colors?
   > Answer: ...
   ```

### ✅ Phase 2: Advanced UX + Caching

1. **📁 In-memory or local cache**

   - Avoid re-scraping same URL by saving index in `data/`

2. **📊 Streamlit UI**

   - Build interactive web interface with:
     - URL input field
     - Question input box
     - Real-time answer display
     - Context chunks + source URL visualization

3. **🧪 Test Cases**
   - Run on Tailwind docs, React docs, Django docs, etc.

### ✅ Phase 3: Live Enhancements

1. **🔄 Recursive Smart Crawler**

   - Use `requests-html` or `Playwright` for JS-heavy sites
   - Filter for documentation-like paths

2. **🌐 Link Indexing with Metadata**

   - Track content source (URL, title, section header)

3. **📎 Answer with Source Links**

   - Add source URLs to each answer chunk

4. **🧠 Use Local LLM (Optional)**
   - Use `Mistral`, `TinyLLaMA`, or `OpenHermes` via `llama-cpp-python` or `ollama`

### ✅ Phase 4: Deployment (Optional)

1. **🔌 API Backend**

   - Use FastAPI to expose endpoints:
     - `/scrape`
     - `/query`
     - `/answer`

2. **🌍 Deploy**
   - **Backend**: Render, Railway, or Hugging Face Spaces
   - **Frontend**: Vercel (if using Next.js)

## 🧰 Tech Stack & Libraries (Optimized Choices)

| Purpose        | Selected Tool                 | Why This Choice                                    |
| -------------- | ----------------------------- | -------------------------------------------------- |
| Web Scraping   | `requests` + `BeautifulSoup`  | Lightweight, reliable, handles 90% of sites        |
| JS-Heavy Sites | `requests-html` (when needed) | Python-native, simpler than Playwright             |
| HTML Cleaning  | `html2text`                   | Fast, minimal dependencies                         |
| Embeddings     | `sentence-transformers`       | Free, runs locally, good performance               |
| Vector Store   | `FAISS`                       | Facebook's library, fast, no API costs             |
| LLM Generator  | `OpenAI GPT-3.5-turbo`        | Best cost/performance ratio                        |
| Frontend       | `Streamlit`                   | **Your choice** - Easy to learn, rapid prototyping |
| Backend        | `FastAPI` (optional)          | Modern, fast, great for APIs if needed             |

## 🏁 Milestone Timeline

| Week | Milestone                                     |
| ---- | --------------------------------------------- |
| 1    | Basic scraper + cleaner + CLI prototype       |
| 2    | Embedding + RAG integration + working answers |
| 3    | Streamlit UI or FastAPI API                   |
| 4    | Caching, metadata, better UX, testing         |
| 5    | Optional: Local LLM support                   |
| 6    | Optional: Deploy backend + frontend           |

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ask-website-anything

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install core dependencies
pip install streamlit requests beautifulsoup4 sentence-transformers faiss-cpu openai html2text requests-html
```

### Quick Start

```bash
# Run the CLI interface
python interface/cli.py

# Or run the Streamlit web app (Phase 2+)
streamlit run interface/streamlit_app.py
```

## 🔥 Advanced RAG Enhancements

### RAGEnhancer Module

The project now includes a powerful `RAGEnhancer` module that improves RAG accuracy and performance:

- **Hybrid Reranking**: Combines semantic, TF-IDF, and keyword-based ranking methods
- **Context Selection**: Intelligently selects diverse and relevant context chunks
- **Query Analysis**: Analyzes query complexity to adjust search strategy
- **Context Quality Suggestions**: Identifies potential improvements to retrieved context

### Enhanced Prompt Engineering

- **Dynamic Prompting**: Adjusts prompts based on query type and complexity
- **Structured Markdown Responses**: Formats answers with proper markdown structure
- **Source Attribution**: Properly cites sources in generated answers
- **Context-Aware Instructions**: Modifies instructions based on detected question type

### Advanced Output Formatting

- **Multiple Format Support**: Save results in both Markdown and HTML formats
- **Detailed Metadata**: Includes query analysis, context quality, and answer quality metrics
- **Professional Styling**: HTML output with responsive design and proper styling
- **Quality Assessment**: Evaluates answer quality with specific improvement suggestions

### Usage Example

```python
from api.rag_pipeline import RAGPipeline
from api.result_formatter import ResultFormatter

# Initialize pipeline and formatter
rag_pipeline = RAGPipeline()
formatter = ResultFormatter()

# Process a website
url = "https://tailwindcss.com/docs/adding-custom-styles"
query = "How do I create custom utility classes in Tailwind CSS?"
process_result = rag_pipeline.process_website(url=url, query=query)

# Get the enhanced answer
answer = rag_pipeline.answer_question(query=query, session_id=process_result['session_id'])

# Save the result in multiple formats
saved_files = formatter.save_both_formats(answer, "tailwind_custom_utilities")
```

Run the enhanced RAG pipeline test:

```bash
python test_enhanced_rag.py
```

## 🔥 Bonus Features (Future Ideas)

- **Chat History**: Maintain conversation history per website
- **Multi-site Comparison**: Compare information across multiple sites
- **Hybrid Support**: Upload files + URL combination
- **Browser Plugin**: RAG assistant extension for any webpage
- **GitHub Integration**: Analyze repositories (README + docs + code)
- **Real-time Updates**: Monitor website changes and update embeddings
- **Advanced Filtering**: Content type filtering (docs, blogs, code, etc.)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern RAG (Retrieval-Augmented Generation) techniques
- Inspired by the need for dynamic, context-aware web content analysis
- Thanks to the open-source community for the amazing tools and libraries

---

**Created on:** July 10, 2025  
**Status:** In Development  
**Version:** 1.0.0-alpha
