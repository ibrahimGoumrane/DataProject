# RAG Pipeline - Intelligent Web Content Processing System

## 🚀 Project Overview

The RAG Pipeline is an advanced **Retrieval-Augmented Generation (RAG)** system that intelligently processes web content to answer questions with contextually relevant information. It combines modern web scraping, vector search, and large language models to create a powerful knowledge extraction and question-answering system.

### Key Features

- **🌐 Intelligent Web Scraping**: Async parallel scraping with JavaScript rendering support
- **🔍 Vector-Based Search**: FAISS-powered semantic search for relevant content retrieval
- **🧠 AI-Powered Answers**: OpenAI integration for contextual question answering
- **📊 Multi-Website Context**: Combine information from multiple sources under one session
- **⚡ Performance Optimized**: Async processing, smart caching, and efficient data handling
- **🔧 MCP Server Integration**: Model Context Protocol server for external tool integration
- **📈 Advanced Analytics**: Query analysis, relevance scoring, and quality metrics

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE SYSTEM                      │
├─────────────────────────────────────────────────────────────────┤
│  Web Scraping Layer                                            │
│  ├── Async Scraper (Playwright + aiohttp)                     │
│  ├── Content Extraction (HTML2Text)                           │
│  └── Multi-strategy Fallback System                           │
│                                                                 │
│  Data Processing Layer                                          │
│  ├── Text Chunking & Normalization                            │
│  ├── Embedding Generation (Sentence Transformers)             │
│  └── Deduplication & Filtering                                │
│                                                                 │
│  Storage Layer                                                  │
│  ├── FAISS Vector Database                                     │
│  ├── Session Management                                        │
│  └── Metadata Storage                                          │
│                                                                 │
│  AI Enhancement Layer                                           │
│  ├── Query Analysis & Complexity Scoring                      │
│  ├── Hybrid Re-ranking (Semantic + TF-IDF + Keywords)         │
│  └── Context Selection & Optimization                         │
│                                                                 │
│  LLM Integration Layer                                          │
│  ├── OpenAI API Integration                                    │
│  ├── Prompt Engineering                                        │
│  └── Response Quality Assessment                               │
│                                                                 │
│  Interface Layer                                                │
│  ├── Interactive CLI                                           │
│  ├── MCP Server (Model Context Protocol)                      │
│  └── Result Formatting & Export                               │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Core Components

### 1. **RAG Pipeline** (`rag_pipeline.py`)

The central orchestrator that coordinates all components:

- **Website Processing**: Scrapes and processes web content
- **Question Answering**: Retrieves relevant context and generates answers
- **Session Management**: Handles multi-website contexts
- **Quality Control**: Ensures high-quality results

### 2. **Async Web Scraper** (`scraper.py`)

Advanced web scraping with multiple strategies:

- **Parallel Processing**: Concurrent scraping for speed
- **JavaScript Support**: Playwright for dynamic content
- **Fallback Strategies**: Multiple approaches for reliability
- **Rate Limiting**: Respectful scraping practices

### 3. **Data Handler** (`dataHandler.py`)

Intelligent content processing:

- **Text Extraction**: HTML to readable text conversion
- **Smart Chunking**: Sentence-boundary aware splitting
- **Embedding Generation**: Semantic vector creation
- **Deduplication**: Efficient duplicate removal

### 4. **Storage Manager** (`storage_manager.py`)

Efficient data storage and retrieval:

- **FAISS Integration**: High-performance vector search
- **Session Management**: Multi-website context handling
- **Metadata Storage**: Rich content information
- **Persistence**: Disk-based data storage

### 5. **RAG Enhancer** (`rag_enhancer.py`)

Advanced query processing and result enhancement:

- **Query Analysis**: Complexity scoring and type detection
- **Hybrid Re-ranking**: Multi-factor relevance scoring
- **Context Selection**: Optimal content selection
- **Quality Metrics**: Answer quality assessment

### 6. **LLM Integration** (`llm.py`)

OpenAI API integration with intelligent prompting:

- **Contextual Prompting**: Optimized prompt construction
- **Response Processing**: Quality assessment and formatting
- **Error Handling**: Robust API interaction
- **Token Management**: Efficient token usage

### 7. **MCP Server** (`server.py`)

Model Context Protocol server for external integrations:

- **URL Registration**: Process websites for context
- **Query Processing**: Answer questions with context
- **Session Management**: Handle multi-website contexts
- **API Endpoints**: Tools for external systems

## 🔄 System Flow

Let me explain the detailed flow of execution when running both the `process_website` and `answer_question` methods in the RAG pipeline, including how data is processed and how the LLM receives its context.

### 1. Website Processing Flow (`process_website`)

The `process_website` method handles fetching, processing, and storing website content:

```
User calls process_website with URL and query
├── 1. Scraper scrapes the website (using Playwright)
│   ├── Extracts HTML content
│   ├── Renders JavaScript if needed
│   ├── Follows links up to the configured depth
│   └── Returns raw content
│
├── 2. DataHandler processes the raw content
│   ├── Converts HTML to readable text (via HTML2Text)
│   ├── Performs minimal text cleaning (whitespace normalization)
│   ├── Chunks the text based on size and sentence boundaries
│   ├── Filters out empty chunks and exact duplicates
│   └── Creates embeddings for each chunk
│
├── 3. StorageManager stores the data
│   ├── Creates a unique session ID
│   ├── Normalizes embeddings for cosine similarity
│   ├── Stores embeddings in FAISS vector index
│   ├── Stores original chunks and metadata
│   └── Persists data to disk
│
└── 4. Returns session info to the user
    ├── Session ID for future reference
    ├── Number of chunks stored
    └── Sample of immediate relevant context
```

### 2. Question Answering Flow (`answer_question`)

The `answer_question` method retrieves relevant content and generates an answer:

```
User calls answer_question with query and session ID
├── 1. RAGEnhancer analyzes query complexity
│   ├── Calculates complexity score based on length, entities, etc.
│   ├── Determines question type (procedural, definition, etc.)
│   └── Suggests optimal number of results to retrieve (k)
│
├── 2. DataHandler processes the query
│   ├── Performs minimal normalization (just whitespace)
│   └── Creates an embedding for the query
│
├── 3. StorageManager searches for relevant content
│   ├── Uses FAISS to find vectors similar to query embedding
│   ├── Retrieves k chunks with highest similarity
│   ├── Includes original text and metadata for each chunk
│   └── Filters by session ID if specified
│
├── 4. RAGEnhancer re-ranks search results
│   ├── Performs hybrid re-ranking:
│   │   ├── Semantic similarity (50%)
│   │   ├── TF-IDF matching (30%)
│   │   └── Keyword overlap (20%)
│   └── Returns results in new ranked order
│
├── 5. RAGEnhancer selects context
│   ├── Takes re-ranked results
│   ├── Combines chunks up to max_context_length (10,000 chars)
│   ├── Adds relevance scores to each chunk
│   └── Creates a single context string
│
├── 6. LLM generates answer with context
│   ├── Creates enhanced prompt with:
│   │   ├── Original query
│   │   ├── Selected context (up to 10,000 chars)
│   │   ├── Source information
│   │   └── Type-specific instructions
│   ├── Sends prompt to OpenAI with system instructions
│   └── Retrieves and processes response
│
└── 7. Returns comprehensive answer to user
    ├── LLM-generated answer
    ├── Original and re-ranked search results
    ├── Query analysis information
    ├── Source metadata
    └── Answer quality metrics
```

### 3. LLM Context Processing

The key part of this process is how the LLM receives the context data:

1. **Context Selection**: The RAG enhancer takes the re-ranked search results and combines them into a single context string up to the maximum context length (10,000 characters), preserving as much of the original content as possible.

2. **Prompt Construction**: The LLM module creates an enhanced prompt with the following structure:

   ```
   QUESTION: {user_query}

   RELEVANT CONTEXT:
   [Relevance: 0.923] {first_chunk_content}

   [Relevance: 0.887] {second_chunk_content}

   ...more context chunks...

   SOURCES:
   - {source_url_1}
   - {source_url_2}
   - etc.

   Please provide a comprehensive and accurate answer to the question based on the context provided.

   Instructions:
   1. Format your response in clean Markdown...
   2. Answer directly and concisely...
   ...more instructions...
   ```

3. **System Prompt**: A system prompt is also provided to the LLM to set its behavior:

   ```
   You are a helpful AI assistant that answers questions based on provided context from web content.

   Your role:
   - Answer questions accurately using only the provided context
   - Be concise but comprehensive
   ...more role description...
   ```

4. **API Call**: The complete prompt (system prompt + enhanced prompt with context) is sent to the OpenAI API with the configured parameters (temperature, max tokens, etc.).

5. **Response Handling**: The LLM's response is processed, quality metrics are calculated, and the full result is returned to the user.

This approach ensures that the LLM has access to the most relevant content from the website while staying within token limits, and that the answer is based specifically on the retrieved content rather than the model's general knowledge.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- Modern web browser (for Playwright)

### Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd DataProject
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Playwright browsers**:

   ```bash
   playwright install
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   RAG_MAX_CONTEXT_LENGTH=10000
   RAG_MODEL_NAME=o4-mini-2025-04-16
   SCRAPER_MAX_CONCURRENT=10
   ```

### Quick Start

1. **Run the interactive CLI**:

   ```bash
   python api/main.py
   ```

2. **Start the MCP server**:

   ```bash
   python api/server.py
   ```

3. **Basic usage example**:

   ```python
   from api.rag_pipeline import RAGPipeline

   # Initialize pipeline
   pipeline = RAGPipeline()

   # Process a website
   result = pipeline.process_website(
       url="https://example.com",
       query="What is this about?"
   )

   # Ask a question
   answer = pipeline.answer_question(
       query="How does this work?",
       session_id=result['session_id']
   )

   print(answer['answer'])
   ```

## 📁 Project Structure

```
DataProject/
├── api/                          # Core API components
│   ├── rag_pipeline.py          # Main RAG pipeline orchestrator
│   ├── scraper.py               # Async web scraping system
│   ├── dataHandler.py           # Content processing and embedding
│   ├── storage_manager.py       # Vector storage and retrieval
│   ├── rag_enhancer.py          # Query analysis and re-ranking
│   ├── llm.py                   # OpenAI integration
│   ├── result_formatter.py      # Output formatting
│   ├── config.py                # Configuration management
│   ├── main.py                  # Interactive CLI interface
│   └── server.py                # MCP server implementation
├── storage/                      # Data storage directory
│   ├── chunks.pkl               # Processed text chunks
│   ├── faiss_index.bin          # FAISS vector index
│   ├── metadata.pkl             # Content metadata
│   └── sessions.json            # Session management
├── results/                      # Output directory
│   ├── result_*.html            # HTML formatted results
│   └── result_*.md              # Markdown formatted results
├── requirements.txt              # Python dependencies
├── README.md                    # This file
└── ASYNC_SCRAPING_README.md     # Async scraping documentation
```

## 🎯 Usage Examples

### Basic Website Processing

```python
from api.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

# Process a single website
result = pipeline.process_website(
    url="https://docs.python.org/3/",
    query="Python documentation"
)

print(f"Processed {result['chunks_stored']} chunks")
print(f"Session ID: {result['session_id']}")
```

### Multi-Website Context

```python
# Process first website
result1 = pipeline.process_website(
    url="https://fastapi.tiangolo.com/",
    query="FastAPI framework"
)
session_id = result1['session_id']

# Add second website to the same session
result2 = pipeline.process_website(
    url="https://docs.pydantic.dev/",
    query="Pydantic models",
    session_id=session_id
)

# Ask questions with combined context
answer = pipeline.answer_question(
    query="How do FastAPI and Pydantic work together?",
    session_id=session_id
)
```

### Advanced Query Processing

```python
# Process website with async scraping
result = pipeline.process_website(
    url="https://example.com",
    query="comprehensive guide",
    use_async=True
)

# Ask complex questions
answer = pipeline.answer_question(
    query="What are the best practices and common pitfalls?",
    session_id=result['session_id'],
    use_session_only=True  # Focus on processed content only
)

# Access detailed results
print(f"Answer: {answer['answer']}")
print(f"Confidence: {answer['confidence']}")
print(f"Sources: {answer['sources']}")
```

## 🔧 Configuration

The system uses environment variables for configuration. Key settings:

### Core Settings

- `OPENAI_API_KEY`: Your OpenAI API key
- `RAG_MAX_CONTEXT_LENGTH`: Maximum context length (default: 10000)
- `RAG_MODEL_NAME`: OpenAI model to use (default: o4-mini-2025-04-16)
- `RAG_TEMPERATURE`: Response creativity (default: 0.7)

### Scraping Settings

- `SCRAPER_MAX_CONCURRENT`: Concurrent scraping limit (default: 10)
- `SCRAPER_TIMEOUT`: Request timeout in seconds (default: 60)
- `SCRAPER_MAX_DEPTH`: Maximum scraping depth (default: 4)

### Data Processing

- `DATA_CHUNK_SIZE`: Text chunk size (default: 1000)
- `DATA_CHUNK_OVERLAP`: Chunk overlap (default: 100)

See `api/config.py` for all available settings.

## 🎭 MCP Server Integration

The project includes a Model Context Protocol (MCP) server for external tool integration:

### Available Tools

1. **`register_url`**: Register a URL for processing

   ```python
   register_url(
       url="https://example.com",
       contentToLookFor="specific content",
       sessionId="session-123"
   )
   ```

2. **`search_url`**: Search registered content
   ```python
   search_url(
       query="What is this about?",
       sessionId="session-123",
       useOwnContextOnly=True
   )
   ```

### Starting the MCP Server

```bash
python api/server.py
```

The server provides a FastMCP interface for external systems to interact with the RAG pipeline.

## 📊 Performance & Scalability

### Async Processing

- **Parallel scraping**: Up to 10 concurrent requests
- **Batch processing**: Efficient memory usage
- **Smart throttling**: Respects rate limits

### Vector Search Performance

- **FAISS indexing**: Sub-second similarity search
- **Efficient storage**: Normalized embeddings
- **Scalable design**: Handles large document collections

### Memory Management

- **Lazy loading**: Components loaded on demand
- **Efficient chunking**: Optimized text processing
- **Smart caching**: Reduced redundant processing

## 🔍 Advanced Features

### Query Analysis

- **Complexity scoring**: Automatic difficulty assessment
- **Type detection**: Question categorization
- **Optimal retrieval**: Dynamic result count adjustment

### Hybrid Re-ranking

- **Semantic similarity**: 50% weight
- **TF-IDF matching**: 30% weight
- **Keyword overlap**: 20% weight

### Quality Metrics

- **Answer confidence**: Relevance scoring
- **Source attribution**: Content traceability
- **Quality assessment**: Automatic evaluation

## 🧪 Testing & Development

### Running Tests

```bash
# Test multi-website context
python test_multi_website_context.py

# Interactive testing
python api/main.py
```

### Development Setup

1. Install development dependencies
2. Set up pre-commit hooks
3. Configure IDE with project structure

## 🔮 Future Enhancements

### Planned Features

- **Real-time processing**: Stream processing capabilities
- **Multi-language support**: International content processing
- **Advanced analytics**: Usage metrics and insights
- **API endpoints**: RESTful API interface
- **Web interface**: Browser-based UI

### Technical Improvements

- **Caching layers**: Redis integration
- **Distributed processing**: Multi-node scaling
- **Advanced embeddings**: Custom model training
- **Quality optimization**: Enhanced ranking algorithms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- FAISS for vector search
- Playwright for web scraping
- Sentence Transformers for embeddings
- The open-source community for excellent tools

---

_Built with ❤️ for intelligent web content processing_

## 🌐 Multi-Website Context Support

The RAG pipeline supports combining content from multiple websites under a single session ID, enabling comprehensive analysis across multiple sources.

### How Multi-Website Context Works

1. **Process First Website**: Process a website normally to get a session ID
2. **Process Additional Websites**: Use the same session ID when processing additional websites
3. **Ask Questions**: When asking questions, use the shared session ID to get context from all processed websites

### Multi-Website Example

```python
# Process first website
result1 = pipeline.process_website(
    url="https://example.com/page1",
    query="What is X?"
)
session_id = result1['session_id']

# Process second website using the same session ID
result2 = pipeline.process_website(
    url="https://example.com/page2",
    query="How does X relate to Y?",
    session_id=session_id  # Reuse the session ID
)

# Ask a question that requires context from both websites
answer = pipeline.answer_question(
    "How does X work with Y?",
    session_id=session_id
)
```

### Benefits of Multi-Website Context

- **Broader Context**: Get answers that incorporate information from multiple related sources
- **Cross-Reference Information**: Answer questions that require comparing or combining information
- **Topic Exploration**: Build a comprehensive knowledge base on a specific topic across multiple pages

### Testing Multi-Website Context

Use the test script `test_multi_website_context.py` to see this feature in action, or run `main.py` and select option 2 to test multi-website context processing.
