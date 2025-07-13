# Execution Flow of process_website and answer_question

Let me explain the detailed flow of execution when running both the `process_website` and `answer_question` methods in the RAG pipeline, including how data is processed and how the LLM receives its context.

## 1. process_website Flow

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

## 2. answer_question Flow

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
│   ├── Sends prompt to Ollama locally running model (deepseek-r1:1.5b) with system instructions
│   └── Retrieves and processes response
│
└── 7. Returns comprehensive answer to user
    ├── LLM-generated answer
    ├── Original and re-ranked search results
    ├── Query analysis information
    ├── Source metadata
    └── Answer quality metrics
```

## How the LLM Gets Fed Data

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

4. **API Call**: The complete prompt (system prompt + enhanced prompt with context) is sent to the Ollama API with the configured parameters (temperature, max tokens, etc.).

5. **Response Handling**: The LLM's response is processed, quality metrics are calculated, and the full result is returned to the user.

This approach ensures that the LLM has access to the most relevant content from the website while staying within token limits, and that the answer is based specifically on the retrieved content rather than the model's general knowledge.

# Multi-Website Context Support

The RAG pipeline now supports combining content from multiple websites under a single session ID. This feature allows for broader context when answering questions that span multiple sources.

## How It Works

1. **Process First Website**: Process a website normally to get a session ID
2. **Process Additional Websites**: Use the same session ID when processing additional websites
3. **Ask Questions**: When asking questions, use the shared session ID to get context from all processed websites

## Example Usage

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

## Benefits

- **Broader Context**: Get answers that incorporate information from multiple related sources
- **Cross-Reference Information**: Answer questions that require comparing or combining information
- **Topic Exploration**: Build a comprehensive knowledge base on a specific topic across multiple pages

## Testing

Use the test script `test_multi_website_context.py` to see this feature in action, or run `main.py` and select option 2 to test multi-website context processing.
