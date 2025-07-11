# Async Web Scraping System - Technical Documentation

## Overview

This documentation explains the async web scraping system implemented in the DataProject. The system uses **asynchronous parallel processing** to efficiently scrape websites while respecting rate limits and providing comprehensive URL tracking for all scraped content.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ASYNC SCRAPING PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│  Entry Point: scrape_parallel()                                │
│       ↓                                                         │
│  Async Controller: smart_scrape_parallel()                     │
│       ↓                                                         │
│  Event Loop Manager: smart_scrape_async()                      │
│       ↓                                                         │
│  Page Processor: _scrape_single_page_async()                   │
│       ↓                                                         │
│  Content Fetcher: _fetch_content_async()                       │
│       ↓                                                         │
│  Strategy Chain: Playwright → Advanced → Basic                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Async Functions

### 1. `scrape_parallel(url, query)` - Main Entry Point

**Purpose**: Primary interface for async scraping with URL tracking

**Input**:

- `url` (str): Starting URL to scrape
- `query` (str): Search query for relevance filtering

**Output**:

```python
{
    'query': 'user search query',
    'query_embedding': np.array([...]),
    'processed_chunks': ['chunk1', 'chunk2', ...],
    'chunk_embeddings': np.array([[...], [...]]),
    'chunk_sources': ['url1', 'url1', 'url2', ...],  # NEW: Source URL for each chunk
    'source_url': 'starting_url',
    'total_chunks': 42
}
```

**Example**:

```python
scraper = Scraper()
result = scraper.scrape_parallel(
    url="https://example.com",
    query="machine learning tutorials"
)

# Access chunk sources
for i, chunk in enumerate(result['processed_chunks']):
    source_url = result['chunk_sources'][i]
    print(f"Chunk {i}: {chunk[:100]}...")
    print(f"Source: {source_url}")
```

### 2. `smart_scrape_parallel(url, query)` - Async Controller

**Purpose**: Manages async event loop and provides sync interface

**Features**:

- Handles existing event loops in Jupyter/async environments
- Creates new event loops in threads when needed
- Provides performance metrics
- Thread-safe execution

**Example**:

```python
# Returns list of (content, source_url) tuples
content_tuples = scraper.smart_scrape_parallel(
    url="https://docs.python.org",
    query="async programming"
)

for content, source_url in content_tuples:
    print(f"Content from {source_url}: {len(content)} characters")
```

### 3. `smart_scrape_async(url, query, depth)` - Core Async Engine

**Purpose**: Recursive async scraping with batched concurrent processing

**Key Features**:

- **Thread-safe quota management**: Prevents exceeding `max_anchor_links`
- **Batched processing**: Processes links in configurable batches
- **Semaphore limiting**: Controls concurrent requests
- **Relevance filtering**: Skips irrelevant content to save quota
- **URL tracking**: Returns (content, source_url) tuples

**Configuration** (from `config.py`):

```python
SCRAPER_MAX_CONCURRENT = 10      # Max concurrent requests
SCRAPER_BATCH_SIZE = 5           # Links per batch
SCRAPER_ASYNC_TIMEOUT = 30       # Request timeout (seconds)
SCRAPER_SEMAPHORE_LIMIT = 15     # Semaphore limit
```

**Example**:

```python
# Direct async usage (in async context)
async def scrape_example():
    scraper = Scraper()
    content_tuples = await scraper.smart_scrape_async(
        url="https://example.com",
        query="python tutorials",
        current_depth=0
    )

    for content, source_url in content_tuples:
        print(f"Scraped {len(content)} chars from {source_url}")
```

### 4. `_scrape_single_page_async(url, query, depth)` - Page Processor

**Purpose**: Processes individual pages with content validation

**Returns**: `((text_content, source_url), anchor_links)`

**Logic Flow**:

1. Fetch content using multi-strategy approach
2. Parse HTML and extract text
3. Check content relevance against query
4. Extract internal anchor links
5. Return content with source URL

**Example Return**:

```python
# Success case
(("This is the page content...", "https://example.com/page1"),
 ["https://example.com/page2", "https://example.com/page3"])

# Failed/irrelevant case
(("", ""), [])
```

### 5. `_fetch_content_async(url, use_js)` - Content Fetcher

**Purpose**: Multi-strategy async content fetching with anti-detection

**Strategy Chain**:

1. **Playwright Stealth** (if `use_js=True`)

   - Full browser automation
   - JavaScript execution
   - Human behavior simulation
   - Anti-detection measures

2. **Advanced aiohttp** (primary)

   - Async HTTP requests
   - Random headers/user agents
   - SSL handling
   - Rate limit detection

3. **Basic aiohttp** (fallback)
   - Simple HTTP requests
   - Minimal headers
   - Last resort option

**Example**:

```python
# In async context
content = await scraper._fetch_content_async(
    url="https://spa-website.com",
    use_js=True  # Use Playwright for JavaScript-heavy sites
)
```

## Thread Safety & Quota Management

### Thread-Safe Counter Operations

The system uses `threading.Lock()` to ensure safe concurrent access to the anchor link counter:

```python
def _increment_anchor_links(self) -> bool:
    """Thread-safe increment with quota check"""
    with self._anchor_links_lock:
        if self.current_anchor_links < self.max_anchor_links:
            self.current_anchor_links += 1
            return True
        return False

def _decrement_anchor_links(self):
    """Thread-safe decrement (for error rollback)"""
    with self._anchor_links_lock:
        if self.current_anchor_links > 0:
            self.current_anchor_links -= 1

def _get_anchor_links_status(self) -> tuple:
    """Thread-safe status check"""
    with self._anchor_links_lock:
        current = self.current_anchor_links
        max_count = self.max_anchor_links
        remaining = max_count - current
        return current, max_count, remaining
```
## URL Tracking System

### How URLs are Tracked

Each piece of content is paired with its source URL throughout the pipeline:

```python
# 1. Single page returns content with source
async def _scrape_single_page_async(self, url, query, depth):
    # ... scraping logic ...
    return (text_content, url), anchor_links  # Content paired with source URL

# 2. Recursive scraping maintains URL associations
async def smart_scrape_async(self, url, query, depth):
    content_tuple, anchor_links = await self._scrape_single_page_async(url, query, depth)
    all_contents = [content_tuple] if content_tuple[0] else []

    # Child pages also return (content, url) tuples
    for result in batch_results:
        if isinstance(result, list):
            all_contents.extend(result)  # Each item is (content, source_url)

    return all_contents

# 3. Processing maintains URL tracking
def scrape_parallel(self, url, query):
    raw_content_list = self.smart_scrape_parallel(url, query)

    all_chunk_sources = []
    for content_tuple in raw_content_list:
        raw_content, source_url = content_tuple
        processed_chunks, embeddings = self.process_html(raw_content)

        # Each chunk gets the same source URL
        all_chunk_sources.extend([source_url] * len(processed_chunks))
```

### Using URL Tracking

```python
# Example: Find all chunks from a specific domain
result = scraper.scrape_parallel("https://example.com", "tutorials")

domain_chunks = []
for i, chunk in enumerate(result['processed_chunks']):
    source_url = result['chunk_sources'][i]
    if "docs.python.org" in source_url:
        domain_chunks.append({
            'content': chunk,
            'source': source_url,
            'embedding': result['chunk_embeddings'][i]
        })

print(f"Found {len(domain_chunks)} chunks from Python docs")
```

## Performance Features

### Batched Concurrent Processing

```python
# Process links in batches to avoid overwhelming servers
batch_size = self.config.SCRAPER_BATCH_SIZE  # Default: 5
semaphore = asyncio.Semaphore(self.config.SCRAPER_SEMAPHORE_LIMIT)  # Default: 15

for i in range(0, len(links), batch_size):
    batch = links[i:i + batch_size]

    # Create tasks with semaphore limiting
    tasks = [process_with_semaphore(link) for link in batch]

    # Wait for batch completion
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and add delay between batches
    await asyncio.sleep(random.uniform(1, 2))
```

### Anti-Detection Measures

```python
# 1. Random User Agents
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36...',
    # ... 9 different user agents
]

# 2. Random Headers
headers_pool = [
    {'Accept': 'text/html,application/xhtml+xml...', 'Accept-Language': 'en-US,en;q=0.5'},
    {'Accept': 'text/html,application/xhtml+xml...', 'Accept-Language': 'en-US,en;q=0.9'},
    # ... 3 different header sets
]

# 3. Random Delays
await asyncio.sleep(random.uniform(0.5, 1.5))  # Between requests
await asyncio.sleep(random.uniform(1, 2))      # Between batches

# 4. Playwright Stealth Mode
stealth_js = """
// Override navigator.webdriver
Object.defineProperty(navigator, 'webdriver', { get: () => false });
// Mock screen properties, plugins, etc.
"""
```

## Configuration Options

### Key Settings in `config.py`

```python
# Async Scraping Configuration
SCRAPER_MAX_CONCURRENT = 10        # Maximum concurrent requests
SCRAPER_BATCH_SIZE = 5             # Links processed per batch
SCRAPER_ASYNC_TIMEOUT = 30         # Request timeout in seconds
SCRAPER_SEMAPHORE_LIMIT = 15       # Semaphore limit for connection pooling

# Quota Management
SCRAPER_MAX_ANCHOR_LINKS = 50      # Maximum links to process
SCRAPER_MAX_DEPTH = 2              # Maximum recursion depth

# Content Filtering
SCRAPER_SPARSE_CONTENT_THRESHOLD = 500       # Min content length
SCRAPER_MEAN_SIMILARITY_THRESHOLD = 0.3     # Relevance threshold
```

### Tuning Performance

```python
# For faster scraping (more aggressive)
SCRAPER_MAX_CONCURRENT = 20
SCRAPER_BATCH_SIZE = 10
SCRAPER_SEMAPHORE_LIMIT = 25

# For more respectful scraping (slower)
SCRAPER_MAX_CONCURRENT = 5
SCRAPER_BATCH_SIZE = 3
SCRAPER_SEMAPHORE_LIMIT = 8
```

## Error Handling & Resilience

### Automatic Fallback Strategy

```python
async def _fetch_content_async(self, url, use_js=False):
    # Strategy 1: Playwright (JavaScript-heavy sites)
    if use_js:
        content = await self._fetch_with_playwright_stealth_async(url)
        if content: return content

    # Strategy 2: Advanced aiohttp (most cases)
    content = await self._fetch_with_advanced_requests_async(url)
    if content: return content

    # Strategy 3: Basic aiohttp (fallback)
    content = await self._fetch_with_basic_requests_async(url)
    return content
```

### Error Recovery

```python
async def process_link_with_error_handling(link_url):
    try:
        # Increment quota counter
        if not self._increment_anchor_links():
            return []  # Quota reached

        # Process the link
        result = await self.smart_scrape_async(link_url, query, depth + 1)
        return result

    except Exception as e:
        print(f"Error processing {link_url}: {e}")
        # Rollback quota counter on failure
        self._decrement_anchor_links()
        return []
```

## Usage Examples

### Basic Usage

```python
from scraper import Scraper

# Initialize scraper
scraper = Scraper()

# Scrape with URL tracking
result = scraper.scrape_parallel(
    url="https://python.org/docs",
    query="async programming"
)

# Access results
print(f"Scraped {result['total_chunks']} chunks")
print(f"From {len(set(result['chunk_sources']))} unique URLs")
```

### Advanced Usage with Custom Configuration

```python
from scraper import Scraper
from config import RAGConfig

# Custom configuration
config = RAGConfig()
config.SCRAPER_MAX_CONCURRENT = 15
config.SCRAPER_BATCH_SIZE = 8
config.SCRAPER_MAX_ANCHOR_LINKS = 100

# Initialize with custom config
scraper = Scraper()
scraper.config = config

# Scrape large site
result = scraper.scrape_parallel(
    url="https://docs.python.org",
    query="machine learning"
)

# Analyze source distribution
source_counts = {}
for source in result['chunk_sources']:
    source_counts[source] = source_counts.get(source, 0) + 1

print("Content distribution by source:")
for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {source}: {count} chunks")
```

### Integration with RAG Pipeline

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline (uses async by default)
pipeline = RAGPipeline()

# Process website with async scraping
result = pipeline.process_website(
    url="https://fastapi.tiangolo.com",
    query="REST API development",
    session_id="my_session"
)

# The result automatically includes URL tracking
print(f"Stored {result['chunks_stored']} chunks")
print(f"From sources: {set(result.get('chunk_sources', []))}")
```

## Benefits of Async Approach

### Performance Improvements

1. **Parallel Processing**: Multiple pages scraped simultaneously
2. **Non-blocking I/O**: No waiting for individual requests
3. **Efficient Resource Usage**: Better CPU and network utilization
4. **Batch Optimization**: Reduced overhead through batching

### Quality Improvements

1. **URL Tracking**: Know exactly where each piece of content came from
2. **Relevance Filtering**: Skip irrelevant content early to save quota
3. **Thread Safety**: No race conditions in concurrent operations
4. **Anti-Detection**: Better success rates with stealthy requests

### Reliability Features

1. **Multiple Fallback Strategies**: If one method fails, try others
2. **Quota Management**: Never exceed configured limits
3. **Error Recovery**: Graceful handling of network issues
4. **Configurable Timeouts**: Prevent hanging requests

This async scraping system provides a robust, efficient, and feature-rich solution for web scraping with comprehensive URL tracking and performance optimization.
