from bs4 import BeautifulSoup as bs
from urllib.parse import urlparse, urljoin
from playwright.async_api import async_playwright
from dataHandler import DataHandler
import numpy as np
from config import RAGConfig
import random
import asyncio
import aiohttp
import time
from typing import List, Tuple
import threading
class Scraper:
    def __init__(self, url=None):
        """
        Initialize the Enhanced Scraper with anti-detection measures.
        """
        self.config = RAGConfig()
        self.url = url
        self.base_url = self._get_base_url(url) if url else None
        self.data_handler = DataHandler()
        self.max_depth = self.config.SCRAPER_MAX_DEPTH
        self.max_retries = self.config.SCRAPER_MAX_RETRIES
        self.sparse_content_threshold = self.config.SCRAPER_SPARSE_CONTENT_THRESHOLD
        self.mean_similarity_threshold = self.config.SCRAPER_MEAN_SIMILARITY_THRESHOLD
        self.max_anchor_links = self.config.SCRAPER_MAX_ANCHOR_LINKS
        self.current_anchor_links = 0
        
        # Enhanced anti-detection measures
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
        ]
        
        self.headers_pool = [
            {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0',
            },
            {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
            },
            {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'same-origin',
                'DNT': '1',
            }
        ]
        
        # Thread-safe counter for async operations
        self._anchor_links_lock = threading.Lock()
        
        # Stealth JavaScript for Playwright
        self.stealth_js = """
        // Override the navigator.webdriver property
        Object.defineProperty(navigator, 'webdriver', {
            get: () => false,
        });
        
        // Override the navigator.plugins property
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });
        
        // Override the navigator.languages property
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
        });
        
        // Override the screen properties
        Object.defineProperty(screen, 'width', {
            get: () => 1920,
        });
        Object.defineProperty(screen, 'height', {
            get: () => 1080,
        });
        
        // Remove automation indicators
        window.chrome = {
            runtime: {},
        };
        
        // Mock permissions
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
        );
        """
    
    def _get_base_url(self, url):
        """
        Extract the base URL from the given URL.
        Args:
            url (str): The URL to extract the base from.
        Returns:
            str: The base URL.
        """
        if not url:  # Handle None or empty string
            return None
            
        # First let's check if there is an http or https in the url
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url  # Default to https instead of http
        # After that get the base url which is the domain name
        parsed_url = urlparse(url)
        return f"{parsed_url.scheme}://{parsed_url.netloc}"

    def __get_anchor_tags(self, soup: bs, current_url: str):
        """
        Extract and resolve all internal anchor tag URLs from the given BeautifulSoup object.
        
        Args:
            soup (BeautifulSoup): The parsed HTML content.
            current_url (str): The current URL being processed.
        
        Returns:
            list: A list of absolute URLs (within same domain).
        """
        all_anchor_tags = soup.find_all('a')

        if not all_anchor_tags:
            return []

        # Filter anchor tags with href, excluding non-content files
        excluded_extensions = {'.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.pdf', '.zip', '.tar', '.gz'}
        excluded_fragments = {'#', 'css', 'asp', 'javascript:', 'mailto:', 'tel:'}
        
        valid_anchors = []
        for tag in all_anchor_tags:
            href = tag.get('href')
            if not href:
                continue
                
            # Skip fragments and excluded patterns
            if any(frag in href.lower() for frag in excluded_fragments):
                continue
                
            # Skip file extensions
            if any(href.lower().endswith(ext) for ext in excluded_extensions):
                continue
                
            valid_anchors.append(href)
        
        # Remove duplicates
        valid_anchors = list(set(valid_anchors))

        # Determine base domain
        if not self.base_url:
            self.base_url = self._get_base_url(current_url)
        
        if not self.base_url:
            return []
        
        base_domain = urlparse(self.base_url).netloc

        filtered_urls = []
        for href in valid_anchors:
            full_url = urljoin(self.base_url, href)
            parsed_url = urlparse(full_url)

            # Keep only URLs that share the same domain
            if parsed_url.netloc == base_domain and parsed_url.scheme in {"http", "https"}:
                filtered_urls.append(full_url)

        return filtered_urls

    def __check_content_relevance(self, text_content, query):
        """
        Check if the text content is relevant to the query.
        This method uses the DataHandler to compute similarity.
        
        Args:
            text_content (str): The text content to check.
            query (str): The query to compare against.
        
        Returns:
            bool: True if relevant, False otherwise.
        """
        similarities = self.data_handler.compute_similarity(text_content, query)
        mean_similarity = np.mean([sim[1] for sim in similarities]) if similarities else 0.0
        return mean_similarity , mean_similarity >= self.mean_similarity_threshold

    def __reset_anchor_links(self):
        """
        Reset the current anchor links count.
        This is useful when starting a new scrape session.
        """
        self.current_anchor_links = 0

    def get_scraping_status(self):
        """
        Get current scraping status and configuration.
        
        Returns:
            dict: Status information including async scraping capabilities.
        """
        return {
            'current_anchor_links': self.current_anchor_links,
            'max_anchor_links': self.max_anchor_links,
            'max_depth': self.max_depth,
            'sparse_content_threshold': self.sparse_content_threshold,
            'mean_similarity_threshold': self.mean_similarity_threshold,
            'async_scraping_enabled': True,
            'user_agents_available': len(self.user_agents),
            'header_pools_available': len(self.headers_pool),
            'stealth_mode_enabled': True,
            'async_delays_enabled': True,
            'multi_strategy_async_enabled': True,
            'thread_safe_quota_management': True,
            'url_tracking_enabled': True
        }
    
    def reset_scraping_session(self):
        """
        Reset scraping session for async operations.
        """
        self.__reset_anchor_links()
        print("[RESET] Async scraping session reset")
    
    # ============================================================================
    # ASYNC SCRAPING METHODS - Parallel Processing for Better Performance
    # ============================================================================
    
    async def _fetch_content_async(self, url: str, use_js: bool = False) -> str:
        """
        Async version of content fetching with multiple strategies.
        
        Args:
            url (str): The URL to fetch.
            use_js (bool): Whether to render JavaScript or not.
        
        Returns:
            str: The HTML content of the page, or empty string if failed.
        """
        # Strategy 1: Playwright with stealth (for JS-heavy sites)
        if use_js:
            content = await self._fetch_with_playwright_stealth_async(url)
            if content:
                return content
        
        # Strategy 2: Advanced requests with anti-detection
        content = await self._fetch_with_advanced_requests_async(url)
        if content:
            return content
        
        # Strategy 3: Basic requests (fallback)
        return await self._fetch_with_basic_requests_async(url)

    async def _fetch_with_playwright_stealth_async(self, url: str) -> str:
        """Async version of Playwright stealth fetching."""
        try:
            async with async_playwright() as p:
                # Use a random browser
                browser_types = [p.chromium, p.firefox]
                browser_type = random.choice(browser_types)
                
                browser = await browser_type.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-accelerated-2d-canvas',
                        '--disable-gpu',
                        '--window-size=1920,1080',
                        '--disable-web-security',
                        '--disable-features=VizDisplayCompositor'
                    ]
                )
                
                # Create context with random user agent
                context = await browser.new_context(
                    user_agent=random.choice(self.user_agents),
                    viewport={'width': 1920, 'height': 1080},
                    ignore_https_errors=True,
                    java_script_enabled=True
                )
                
                # Add stealth scripts
                await context.add_init_script(self.stealth_js)
                
                page = await context.new_page()
                
                # Random delay before navigation
                await asyncio.sleep(random.uniform(1, 3))
                
                # Navigate with timeout
                await page.goto(url, wait_until='networkidle', timeout=self.config.SCRAPER_ASYNC_TIMEOUT * 1000)
                
                # Wait for content to load
                await page.wait_for_selector("body", timeout=10000)
                
                # Simulate human behavior
                await page.mouse.move(random.randint(100, 800), random.randint(100, 600))
                await page.wait_for_timeout(random.randint(1000, 3000))
                
                # Scroll to trigger lazy loading
                for i in range(3):
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(random.randint(1000, 2000))
                
                # Get final content
                content = await page.content()
                await browser.close()
                
                return content
                
        except Exception as e:
            # Only log critical errors, not common timeouts
            if "timeout" not in str(e).lower() and "connection" not in str(e).lower():
                print(f"Playwright error: {e}")
            return ""

    async def _fetch_with_advanced_requests_async(self, url: str) -> str:
        """Async version of advanced requests fetching."""
        try:
            # Create random headers
            headers = random.choice(self.headers_pool).copy()
            headers['User-Agent'] = random.choice(self.user_agents)
            
            # Add random delay
            await asyncio.sleep(random.uniform(1, 2))
            
            # Use aiohttp for async requests
            timeout = aiohttp.ClientTimeout(total=self.config.SCRAPER_ASYNC_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url, ssl=False) as response:
                    if response.status == 403:
                        return ""
                    elif response.status == 429:
                        # Wait longer and retry once
                        await asyncio.sleep(random.uniform(5, 10))
                        async with session.get(url, ssl=False) as retry_response:
                            if retry_response.status == 200:
                                return await retry_response.text()
                        return ""
                    
                    if response.status == 200:
                        return await response.text()
                    else:
                        return ""
                        
        except Exception:
            return ""

    async def _fetch_with_basic_requests_async(self, url: str) -> str:
        """Async version of basic requests fallback."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url, ssl=False) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        return ""
                        
        except Exception:
            return ""

    async def _scrape_single_page_async(self, url: str, query: str, current_depth: int) -> Tuple[Tuple[str, str], List[str]]:
        """
        Async method to scrape a single page and return content + anchor links.
        
        Args:
            url (str): URL to scrape
            query (str): Query for relevance checking
            current_depth (int): Current depth level
            
        Returns:
            Tuple[Tuple[str, str], List[str]]: ((text_content, source_url), anchor_links)
        """
        try:
            # Fetch content
            content_html = await self._fetch_content_async(url, use_js=False)
            if not content_html:
                return ("", ""), []

            # Parse content
            soup = bs(content_html, 'html.parser')
            text_content = self.data_handler.clean_html_text(str(soup))

            # Check if content is sparse (likely JS-heavy site)
            if len(text_content) < self.sparse_content_threshold:
                content_html = await self._fetch_content_async(url, use_js=True)
                if content_html:
                    soup = bs(content_html, 'html.parser')
                    text_content = self.data_handler.clean_html_text(str(soup))

            # Check content relevance
            mean_similarity, is_relevant = self.__check_content_relevance(text_content, query)

            # Skip irrelevant deeper pages to save quota
            if not is_relevant and current_depth > 0:
                return ("", ""), []
            
            # Extract anchor links if we haven't reached max depth
            anchor_links = []
            if current_depth < self.max_depth:
                anchor_links = self.__get_anchor_tags(soup, url)

            return (text_content, url), anchor_links

        except Exception as e:
            if "timeout" not in str(e).lower():
                print(f"Error scraping {url}: {e}")
            return ("", ""), []

    async def smart_scrape_async(self, url: str = None, query: str = None, current_depth: int = 0) -> List[Tuple[str, str]]:
        """
        Async version of smart scraping with batched concurrent processing.
        
        Args:
            url (str): URL to start from
            query (str): Query for relevance checking
            current_depth (int): Current recursion depth
            
        Returns:
            List[Tuple[str, str]]: List of (text_content, source_url) tuples from all scraped pages
        """
        if url is None:
            url = self.url

        current, max_count, _ = self._get_anchor_links_status()
        
        # Check if we've reached the limit before processing
        if current >= max_count:
            return []
        
        # Scrape the current page
        content_tuple, links_to_process = await self._scrape_single_page_async(url, query, current_depth)
        
        # Start with current page content (if relevant)
        all_contents = [content_tuple] if content_tuple[0] else []
        
        # If we have anchor links and haven't reached max depth, process them with proper quota management
        if links_to_process and current_depth < self.max_depth:
            # Get current remaining quota
            current, max_count, remaining = self._get_anchor_links_status()
            
            if links_to_process:
                # Process links in batches to avoid overwhelming the server
                batch_size = self.config.SCRAPER_BATCH_SIZE
                semaphore = asyncio.Semaphore(self.config.SCRAPER_SEMAPHORE_LIMIT)
                
                async def process_link_with_semaphore(link_url):
                    async with semaphore:
                        # Thread-safe increment with quota check
                        if not self._increment_anchor_links():
                            return []
                        
                        try:
                            # Add random delay
                            await asyncio.sleep(random.uniform(0.5, 1.5))
                            
                            # Recursively scrape the link
                            result = await self.smart_scrape_async(link_url, query, current_depth + 1)
                            return result
                        except Exception as e:
                            # If processing failed, decrement the counter since we didn't actually process it
                            self._decrement_anchor_links()
                            return []
                
                # Process links in batches
                for i in range(0, len(links_to_process), batch_size):
                    # Check quota before each batch
                    current, max_count, remaining = self._get_anchor_links_status()
                    if current >= max_count:
                        break
                    
                    batch = links_to_process[i:i + batch_size]
                    # Limit batch size based on remaining quota
                    batch = batch[:remaining]
                    
                    if not batch:  # No more links to process
                        break
                    
                    # Create tasks for this batch
                    batch_tasks = [process_link_with_semaphore(link) for link in batch]
                    
                    # Wait for batch to complete
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Process batch results
                    for result in batch_results:
                        if isinstance(result, Exception):
                            continue
                        elif isinstance(result, list):
                            all_contents.extend(result)
                    
                    # Small delay between batches to be respectful
                    current, max_count, _ = self._get_anchor_links_status()
                    if i + batch_size < len(links_to_process) and current < max_count:
                        await asyncio.sleep(random.uniform(1, 2))
        
        return all_contents

    def smart_scrape_parallel(self, url: str = None, query: str = None) -> List[Tuple[str, str]]:
        """
        Wrapper method to run async scraping in sync context.
        
        Args:
            url (str): URL to start from
            query (str): Query for relevance checking
            
        Returns:
            List[Tuple[str, str]]: List of (text_content, source_url) tuples from all scraped pages
        """
        if url is None:
            url = self.url
        
        # Reset anchor links counter
        self.__reset_anchor_links()
        
        max_concurrent = self.config.SCRAPER_MAX_CONCURRENT
        start_time = time.time()
        
        # Run the async scraping
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, create a new one in a thread
                    
                    result = [None]
                    exception = [None]
                    
                    def run_in_thread():
                        try:
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            result[0] = new_loop.run_until_complete(
                                self.smart_scrape_async(url, query, 0)
                            )
                            new_loop.close()
                        except Exception as e:
                            exception[0] = e
                    
                    thread = threading.Thread(target=run_in_thread)
                    thread.start()
                    thread.join()
                    
                    if exception[0]:
                        raise exception[0]
                    
                    results = result[0]
                else:
                    results = loop.run_until_complete(
                        self.smart_scrape_async(url, query, 0)
                    )
            except RuntimeError:
                # No event loop exists, create a new one
                results = asyncio.run(self.smart_scrape_async(url, query, 0))
            
            return results
            
        except Exception as e:
            print(f"Error in parallel scraping: {e}")
            return []

    # Update the main scrape method to use parallel scraping optionally
    def scrape_parallel(self, url: str = None, query: str = None):
        """
        Parallel version of the main scrape method.
        
        Args:
            url (str): URL to scrape
            query (str): Query for relevance checking
            
        Returns:
            dict: Essential data package for vector storage
        """
        print(f"[START] Starting parallel scrape for query: '{query}' on URL: {url}")
        
        # 1. Scrape the content in parallel
        raw_content_list = self.smart_scrape_parallel(url, query)
        
        if not raw_content_list:
            print("[ERROR] No content scraped")
            return None
        
        # 2. Process query
        _, query_embedding = self.data_handler.process_query(query)
        
        # 3. Process all scraped content into chunks
        all_processed_chunks = []
        all_chunk_embeddings = []
        all_chunk_sources = []  # Track source URLs for each chunk
        
        print(f"[PROCESS] Processing {len(raw_content_list)} scraped pages...")
        
        for content_tuple in raw_content_list:
            if isinstance(content_tuple, tuple) and len(content_tuple) == 2:
                raw_content, source_url = content_tuple
                if raw_content.strip():  # Only process non-empty content
                    # Process HTML content into chunks
                    processed_chunks, chunk_embeddings = self.data_handler.process_html(raw_content, True)
                    
                    if len(processed_chunks) > 0:
                        all_processed_chunks.extend(processed_chunks)
                        all_chunk_embeddings.extend(chunk_embeddings)
                        # Each chunk from this page gets the same source URL
                        all_chunk_sources.extend([source_url] * len(processed_chunks))
        
        if not all_processed_chunks:
            print("[ERROR] No valid chunks after processing")
            return None
        
        # 4. Return only essential data
        scrape_result = {
            'query': query,
            'query_embedding': query_embedding,
            'processed_chunks': all_processed_chunks,
            'chunk_embeddings': np.array(all_chunk_embeddings),
            'chunk_sources': all_chunk_sources,  # Source URL for each chunk
            'source_url': url,  # Main starting URL
            'total_chunks': len(all_processed_chunks)
        }
        
        print(f"[SUCCESS] Parallel scrape complete: {len(all_processed_chunks)} chunks ready for storage")
        return scrape_result

    def _increment_anchor_links(self) -> bool:
        """
        Thread-safe increment of anchor links counter.
        
        Returns:
            bool: True if increment was successful (within limit), False if limit reached
        """
        with self._anchor_links_lock:
            if self.current_anchor_links < self.max_anchor_links:
                self.current_anchor_links += 1
                return True
            return False
    
    def _decrement_anchor_links(self):
        """
        Thread-safe decrement of anchor links counter.
        """
        with self._anchor_links_lock:
            if self.current_anchor_links > 0:
                self.current_anchor_links -= 1
    
    def _get_anchor_links_status(self) -> tuple:
        """
        Thread-safe getter for anchor links status.
        
        Returns:
            tuple: (current_count, max_count, remaining)
        """
        with self._anchor_links_lock:
            current = self.current_anchor_links
            max_count = self.max_anchor_links
            remaining = max_count - current
            return current, max_count, remaining

