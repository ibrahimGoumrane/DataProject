# RAGEnhancer Documentation

## Overview

The `RAGEnhancer` module provides advanced features to improve the accuracy and performance of Retrieval-Augmented Generation (RAG) pipelines. It offers sophisticated methods for re-ranking search results, enhancing context selection, analyzing query complexity, and suggesting context improvements.

## Class: RAGEnhancer

The core class that implements all enhancement features.

### Constructor

```python
def __init__(self)
```

**Description:** Initializes the RAG enhancer with default settings.

**Key Components:**

- `tfidf_vectorizer`: A TF-IDF vectorizer configured with:
  - 1000 max features
  - English stop words removal
  - Unigram and bigram support (1-2 word phrases)
- `query_cache`: Storage for previously processed queries to improve performance

## Re-ranking Methods

### rerank_search_results

```python
def rerank_search_results(self, query: str, search_results: List[Dict], method: str = 'hybrid') -> List[Dict]
```

**Description:** Re-ranks search results using different scoring methods to improve relevance.

**Parameters:**

- `query`: The original user query (string)
- `search_results`: List of initial search results, each as a dictionary containing at least a 'chunk' of text and 'similarity_score'
- `method`: Re-ranking algorithm to use (string):
  - 'hybrid': Combines semantic, TF-IDF, and keyword matching (default)
  - 'tfidf': Uses only TF-IDF scoring
  - 'bm25': Reserved for future BM25 implementation

**Returns:** List of re-ranked search results with added scores:

- Each result includes the original data plus:
  - `rerank_score`: The new relevance score
  - `original_rank`: The position in the original results list

**Usage Example:**

```python
enhanced_results = enhancer.rerank_search_results(
    "How do Python decorators work?",
    initial_results,
    method='hybrid'
)
```

### \_hybrid_rerank (Internal)

```python
def _hybrid_rerank(self, query: str, search_results: List[Dict]) -> List[Dict]
```

**Description:** Internal method that implements hybrid re-ranking by combining multiple scoring approaches.

**Scoring Components:**

1. **Semantic scores**: The original vector similarity scores (weight: 0.5)
2. **TF-IDF scores**: Term frequency-inverse document frequency scores (weight: 0.3)
3. **Keyword overlap scores**: Jaccard similarity between query and chunk words (weight: 0.2)

**Process:**

1. Extracts chunks and original scores from search results
2. Calculates TF-IDF scores by comparing query against chunks
3. Calculates keyword overlap scores
4. Combines all scores with their respective weights
5. Re-ranks results based on combined scores

### \_tfidf_rerank (Internal)

```python
def _tfidf_rerank(self, query: str, search_results: List[Dict]) -> List[Dict]
```

**Description:** Internal method that implements TF-IDF based re-ranking.

**Process:**

1. Transforms query and chunks into TF-IDF vectors
2. Calculates cosine similarity between query vector and each chunk vector
3. Re-ranks results based on TF-IDF similarity scores

### \_calculate_keyword_overlap (Internal)

```python
def _calculate_keyword_overlap(self, query: str, chunks: List[str]) -> List[float]
```

**Description:** Calculates Jaccard similarity scores between query terms and each chunk.

**Process:**

1. Extracts all words from the query
2. For each chunk:
   - Extracts all words
   - Calculates Jaccard similarity: |intersection| / |union|
   - Returns a score between 0 and 1

## Context Selection Methods

### enhance_context_selection

```python
def enhance_context_selection(self, query: str, search_results: List[Dict], max_context_length: int = 10000) -> str
```

**Description:** Intelligently selects and combines context from search results to create an optimal context for the LLM.

**Parameters:**

- `query`: The user query (string)
- `search_results`: List of search results to select from
- `max_context_length`: Maximum character length for the combined context (default: 10000)

**Returns:** A string containing the combined context with relevance indicators.

**Process:**

1. Re-ranks the search results using hybrid re-ranking
2. Selects diverse chunks to avoid redundancy
3. Combines chunks with relevance scores as prefixes
4. Limits the total length to the specified maximum
5. Adds relevance indicators like `[Relevance: 0.85]` before each chunk

**Format Example:**

```
[Relevance: 0.92] This is the most relevant chunk of information that directly answers the query...

[Relevance: 0.78] This is the second most relevant chunk that provides additional context...
```

### \_select_diverse_chunks (Internal)

```python
def _select_diverse_chunks(self, query: str, search_results: List[Dict], max_length: int) -> List[Dict]
```

**Description:** Selects a diverse set of chunks from search results, avoiding redundancy.

**Process:**

1. Iterates through re-ranked search results
2. For each result:
   - Checks if the chunk is redundant compared to already selected chunks
   - If not redundant and fits within the max length, adds to selected chunks
   - Stops when max length is reached

**Returns:** A filtered list of search result dictionaries that provide diverse information.

### \_is_redundant (Internal)

```python
def _is_redundant(self, new_chunk: str, existing_chunks: List[str], threshold: float = 0.8) -> bool
```

**Description:** Determines if a new chunk is too similar to any existing chunks.

**Parameters:**

- `new_chunk`: The chunk to evaluate
- `existing_chunks`: List of already selected chunks
- `threshold`: Similarity threshold above which chunks are considered redundant (default: 0.8)

**Returns:** Boolean indicating if the chunk is redundant (True) or unique enough (False).

**Process:**

1. Extracts words from the new chunk
2. For each existing chunk:
   - Extracts words from the existing chunk
   - Calculates Jaccard similarity between word sets
   - If similarity exceeds threshold, considers it redundant

## Query Analysis Methods

### analyze_query_complexity

```python
def analyze_query_complexity(self, query: str) -> Dict
```

**Description:** Analyzes query complexity to adjust search strategy and parameters.

**Parameters:**

- `query`: The user query (string)

**Returns:** Dictionary containing analysis results:

- `word_count`: Number of words in the query
- `character_count`: Number of characters in the query
- `question_type`: Detected question type (definition, procedural, causal, etc.)
- `complexity_score`: Calculated complexity score (0.0-1.0)
- `entities`: List of detected entities (proper nouns, numbers, quoted terms)
- `suggested_k`: Recommended number of search results to retrieve based on complexity

**Complexity Calculation Factors:**

1. Query length (word count / 10)
2. Number of entities (entities / 5)
3. Question mark presence (1 if present, 0.5 if not)

**Adaptive k Selection:**

- High complexity (>0.7): k=10
- Medium complexity (>0.4): k=7
- Low complexity (≤0.4): k=5

**Usage Example:**

```python
analysis = enhancer.analyze_query_complexity(
    "How do I implement authentication in a React application using JWT tokens?"
)
# Might return:
# {
#   'word_count': 12,
#   'character_count': 65,
#   'question_type': 'procedural',
#   'complexity_score': 0.73,
#   'entities': ['React', 'JWT'],
#   'suggested_k': 10
# }
```

### \_detect_question_type (Internal)

```python
def _detect_question_type(self, query: str) -> str
```

**Description:** Detects the type of question being asked in the query.

**Returns:** String indicating the question type:

- `definition`: Questions about what something is or means
- `procedural`: Questions about how to do something
- `causal`: Questions about why something happens
- `temporal`: Questions about when something occurs
- `spatial`: Questions about where something is located
- `comparative`: Questions comparing two or more things
- `general`: General questions that don't fit other categories

**Detection Method:**

- Uses keyword matching against common question words and phrases

### \_extract_simple_entities (Internal)

```python
def _extract_simple_entities(self, query: str) -> List[str]
```

**Description:** Extracts potential entities from the query text.

**Extracted Entity Types:**

1. Capitalized words (potential proper nouns)
2. Numbers
3. Terms in quotes (both single and double quotes)

**Returns:** List of unique entities found in the query.

## Context Improvement Methods

### suggest_context_improvements

```python
def suggest_context_improvements(self, query: str, context: str, answer_quality: float = 0.0) -> Dict
```

**Description:** Analyzes context quality and suggests improvements for better answer generation.

**Parameters:**

- `query`: The user query (string)
- `context`: The current context provided to the LLM (string)
- `answer_quality`: Optional estimated quality of the current answer (0.0-1.0)

**Returns:** Dictionary containing improvement suggestions:

- `needs_more_context`: Boolean indicating if more context is needed
- `context_too_long`: Boolean indicating if context is excessively long
- `missing_keywords`: List of query keywords missing from the context
- `redundant_content`: Boolean indicating if context contains redundant information
- `suggestions`: List of specific improvement suggestions as strings

**Analysis Performed:**

1. **Context Length Check**:
   - Too short (<500 chars): Suggests retrieving more chunks
   - Too long (>3000 chars): Suggests filtering content
2. **Missing Keywords Check**:
   - Identifies query terms not present in context
   - Suggests addressing specific missing keywords
3. **Redundancy Check**:
   - Analyzes sentence-level uniqueness
   - Identifies if context has significant repeated information
4. **Quality-Based Suggestions**:
   - If answer quality is low (<0.5), suggests query expansion

**Usage Example:**

```python
suggestions = enhancer.suggest_context_improvements(
    "How do I create a virtual environment in Python?",
    context_text,
    answer_quality=0.42
)
# Might return:
# {
#   'needs_more_context': True,
#   'context_too_long': False,
#   'missing_keywords': ['virtual', 'environment'],
#   'redundant_content': False,
#   'suggestions': [
#     'Consider retrieving more context chunks',
#     'Missing keywords: virtual, environment',
#     'Consider expanding search criteria or adjusting query'
#   ]
# }
```

## Integration with RAG Pipeline

The RAGEnhancer is designed to integrate seamlessly with a standard RAG pipeline:

1. **Pre-retrieval**: Use `analyze_query_complexity` to adjust search parameters
2. **Post-retrieval**: Use `rerank_search_results` to improve result ranking
3. **Context building**: Use `enhance_context_selection` to create optimal context
4. **Quality check**: Use `suggest_context_improvements` to evaluate context quality

## Best Practices

1. **Hybrid Re-ranking**: Always use the hybrid method for best results as it combines semantic, statistical, and keyword approaches
2. **Adaptive Parameters**: Use the query complexity analysis to adjust retrieval parameters
3. **Diverse Context**: Apply the context selection with redundancy checking for better LLM responses
4. **Context Quality Analysis**: Check for context quality issues before sending to the LLM
5. **Iterative Improvement**: Use context improvement suggestions to refine results

## Performance Considerations

- TF-IDF vectorization can be resource-intensive for very large result sets
- The module caches query results to improve performance for repeated queries
- For very large documents, consider increasing the redundancy threshold to ensure coverage

## Example Workflow

```python
# Initialize the enhancer
enhancer = RAGEnhancer()

# Analyze query to determine optimal search parameters
query = "How does React's virtual DOM differ from the real DOM?"
analysis = enhancer.analyze_query_complexity(query)

# Use analysis to guide retrieval (k parameter)
search_results = vector_store.search(query_embedding, k=analysis['suggested_k'])

# Re-rank results for better relevance
reranked_results = enhancer.rerank_search_results(query, search_results)

# Select optimal context with diversity
enhanced_context = enhancer.enhance_context_selection(
    query,
    reranked_results,
    max_context_length=10000
)

# Check context quality and get suggestions
quality_check = enhancer.suggest_context_improvements(query, enhanced_context)

# Use suggestions to improve context if needed
if quality_check['needs_more_context'] or quality_check['missing_keywords']:
    # Implement improvement strategy based on suggestions
    pass

# Send enhanced context to LLM
answer = llm.generate_answer(query, enhanced_context)
```
