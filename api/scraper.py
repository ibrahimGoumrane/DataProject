import requests
from bs4 import BeautifulSoup as bs
from urllib.parse import urlparse, urljoin
from time import sleep
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from dataHandler import DataHandler
import numpy as np
from config import RAGConfig
import random
import json
from pathlib import Path

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
        
        # Initialize session with random settings
        self.session = requests.Session()
        self._randomize_session()
        
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
    
    def _randomize_session(self):
        """Randomize session settings for better anti-detection."""
        # Random user agent
        user_agent = random.choice(self.user_agents)
        headers = random.choice(self.headers_pool).copy()
        headers['User-Agent'] = user_agent
        
        self.session.headers.update(headers)
        
        # Random timeout
        self.session.timeout = random.uniform(10, 30)
        
    def _random_delay(self, min_delay=1, max_delay=3):
        """Add random delay between requests."""
        delay = random.uniform(min_delay, max_delay)
        sleep(delay)
    
    def _enhanced_fetch_content(self, url, use_js=False):
        """
        Enhanced content fetching with multiple strategies and anti-detection.
        
        Args:
            url (str): The URL to fetch.
            use_js (bool): Whether to render JavaScript or not.
        
        Returns:
            str: The HTML content of the page, or empty string if failed.
        """
        print(f"🔄 Fetching content from: {url}")
        
        # Strategy 1: Playwright with stealth (for JS-heavy sites)
        if use_js:
            print("🎭 Using Playwright with stealth mode...")
            content = self._fetch_with_playwright_stealth(url)
            if content:
                print(f"✅ Playwright stealth successful for {url}")
                return content
            print("❌ Playwright stealth failed, trying advanced requests...")
        
        # Strategy 2: Advanced requests with anti-detection
        print("🔧 Using advanced requests with anti-detection...")
        content = self._fetch_with_advanced_requests(url)
        if content:
            print(f"✅ Advanced requests successful for {url}")
            return content
        print("❌ Advanced requests failed, trying basic requests...")
        
        # Strategy 3: Basic requests (fallback)
        print("🔄 Using basic requests as fallback...")
        content = self._fetch_with_basic_requests(url)
        if content:
            print(f"✅ Basic requests successful for {url}")
            return content
        
        print(f"❌ All strategies failed for {url}")
        return ""
    
    def _fetch_with_playwright_stealth(self, url):
        """Fetch content using Playwright with stealth mode."""
        try:
            with sync_playwright() as p:
                # Use a random browser
                browser_type = random.choice([p.chromium, p.firefox])
                browser = browser_type.launch(
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
                context = browser.new_context(
                    user_agent=random.choice(self.user_agents),
                    viewport={'width': 1920, 'height': 1080},
                    ignore_https_errors=True,
                    java_script_enabled=True
                )
                
                # Add stealth scripts
                context.add_init_script(self.stealth_js)
                
                page = context.new_page()
                
                # Random delay before navigation
                self._random_delay(1, 3)
                
                # Navigate with timeout
                page.goto(url, wait_until='networkidle', timeout=30000)
                
                # Wait for content to load
                page.wait_for_selector("body", timeout=10000)
                
                # Simulate human behavior
                page.mouse.move(random.randint(100, 800), random.randint(100, 600))
                page.wait_for_timeout(random.randint(1000, 3000))
                
                # Scroll to trigger lazy loading
                for i in range(3):
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(random.randint(1000, 2000))
                
                # Get final content
                content = page.content()
                browser.close()
                
                return content
                
        except PlaywrightTimeoutError:
            print(f"Playwright timeout for {url}")
            return ""
        except Exception as e:
            print(f"Playwright error for {url}: {e}")
            return ""
    
    def _fetch_with_advanced_requests(self, url):
        """Fetch content using advanced requests with anti-detection."""
        try:
            # Randomize session settings
            self._randomize_session()
            
            # Add random delay
            self._random_delay(1, 2)
            
            # Make request with timeout
            response = self.session.get(url, timeout=30)
            
            # Check for common anti-bot responses
            if response.status_code == 403:
                print(f"403 Forbidden - possible anti-bot protection on {url}")
                return ""
            elif response.status_code == 429:
                print(f"429 Too Many Requests - rate limited on {url}")
                # Wait longer and retry once
                self._random_delay(5, 10)
                response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.text
            else:
                print(f"HTTP {response.status_code} for {url}")
                return ""
                
        except requests.exceptions.RequestException as e:
            print(f"Advanced requests error for {url}: {e}")
            return ""
    
    def _fetch_with_basic_requests(self, url):
        """Basic requests fallback."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                return response.text
            else:
                print(f"Basic requests HTTP {response.status_code} for {url}")
                return ""
                
        except requests.exceptions.RequestException as e:
            print(f"Basic requests error for {url}: {e}")
            return ""

    def _fetch_content(self, url, use_js=False):
        """
        Fetch the content of the given URL using enhanced anti-detection methods.
        Args:
            url (str): The URL to fetch.
            use_js (bool): Whether to render JavaScript or not.
        
        Returns:
            str: The HTML content of the page, or empty string if failed.
        """
        return self._enhanced_fetch_content(url, use_js)

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
            print("No anchor tags found.")
            return []

        # Filter only anchor tags with href
        anchor_tags = [tag for tag in all_anchor_tags if tag.get('href') and len(tag.get('href').split('#')) == 1 and 'css' not in tag.get('href') and 'asp' not in tag.get('href')]
        # Remove duplicate URLs
        anchor_tags = list(set(anchor_tags))

        # All the anchor tags that have the same path but change after the # Should be merged

        # Determine base domain
        if not self.base_url:
            self.base_url = self._get_base_url(current_url)  # Use the current URL parameter
        
        if not self.base_url:  # If still None, return empty list
            print("Could not determine base URL.")
            return []
        
        base_domain = urlparse(self.base_url).netloc

        filtered_urls = []
        for tag in anchor_tags:
            href = tag.get('href')
            full_url = urljoin(self.base_url, href)
            parsed_url = urlparse(full_url)

            # Keep only URLs that share the same domain
            if parsed_url.netloc == base_domain and parsed_url.scheme in {"http", "https"}:
                filtered_urls.append(full_url)

        return filtered_urls
    
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

    def smart_scrape(self, url=None, query=None, current_depth=0 , is_root=False):
        """
        Intelligently scrape content - tries simple scraping first, falls back to JS rendering.
        
        Args:
            url (str): URL to start from.
            current_depth (int): Current recursion depth.
        
        Returns:
            list[str]: Flattened list of text content from the crawled pages.
        """
        print(f"Anchor links so far: {self.current_anchor_links}/{self.max_anchor_links}")
        if self.current_anchor_links >= self.max_anchor_links:
            print(f"Reached maximum anchor links limit ({self.max_anchor_links}). Stopping further scraping.")
            return []
        if url is None:
            url = self.url

        # Try simple scraping first
        content_html = self._fetch_content(url, use_js=False)
        if not content_html:
            return []

        # Get the content with the html tags
        soup = bs(content_html, 'html.parser')
        # Use the html2text library to convert HTML to text
        text_content = self.data_handler.clean_html_text(str(soup))

        # Check if content is sparse (likely JS-heavy site)
        if len(text_content) < self.sparse_content_threshold:  # Threshold for sparse content
            print(f"Sparse content detected for {url}, trying JavaScript rendering...")
            content_html = self._fetch_content(url, use_js=True)
            if content_html:
                soup = bs(content_html, 'html.parser')
                text_content = self.data_handler.clean_html_text(str(soup))

        # Check content relevance to the query
        mean_similarity, is_relevant = self.__check_content_relevance(text_content, query)
        print(f"Mean similarity for {url} is {mean_similarity:.2f} (threshold: {self.mean_similarity_threshold})")
        # If depth exceeded, just return this page's text
        if current_depth >= self.max_depth:
                return [text_content] if is_relevant else []
        # Extract internal anchor tags
        anchor_tags = self.__get_anchor_tags(soup, url)
        if not anchor_tags:
            print("No internal links found.")


        # Collect content from this page
        contents = [text_content]

        for i, anchor in enumerate(anchor_tags):
            if self.current_anchor_links >= self.max_anchor_links:
                print(f"Reached maximum anchor links limit ({self.max_anchor_links}). Stopping further scraping.")
                break
            self.current_anchor_links += 1
            # Add enhanced delay with randomization
            self._random_delay(0.5, 1.5)
            child_content = self.smart_scrape(url=anchor, query=query, current_depth=current_depth + 1)
            if child_content:
                contents.extend(child_content)


        return contents
    

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

    # Keep the original scrape method for backward compatibility
    def scrape(self, url=None, query=None, current_depth=0):
        """
        Simplified scraping pipeline - only stores what's actually needed.
        
        Returns:
            dict: Essential data package for vector storage
        """
        print(f"🚀 Starting scrape for query: '{query}' on URL: {url}")
        
        # 1. Scrape the content
        raw_content_list = self.smart_scrape(url, query, current_depth)
        self.__reset_anchor_links()
        
        if not raw_content_list:
            print("❌ No content scraped")
            return None
        
        # 2. Process query
        _, query_embedding = self.data_handler.process_query(query)
        
        # 3. Process all scraped content into chunks
        all_processed_chunks = []
        all_chunk_embeddings = []
        
        print(f"📄 Processing {len(raw_content_list)} scraped pages...")
        
        for raw_content in raw_content_list:
            # Process HTML content into chunks
            processed_chunks, chunk_embeddings = self.data_handler.process_html(raw_content, True)
            
            if len(processed_chunks) > 0:
                all_processed_chunks.extend(processed_chunks)
                all_chunk_embeddings.extend(chunk_embeddings)
        
        if not all_processed_chunks:
            print("❌ No valid chunks after processing")
            return None
        
        # 4. Return only essential data
        scrape_result = {
            'query': query,
            'query_embedding': query_embedding,
            'processed_chunks': all_processed_chunks,
            'chunk_embeddings': np.array(all_chunk_embeddings),
            'source_url': url,
            'total_chunks': len(all_processed_chunks)
        }
        
        print(f"✅ Scrape complete: {len(all_processed_chunks)} chunks ready for storage")
        return scrape_result

    def get_scraping_status(self):
        """
        Get current scraping status and configuration.
        
        Returns:
            dict: Status information including anti-detection measures.
        """
        return {
            'current_anchor_links': self.current_anchor_links,
            'max_anchor_links': self.max_anchor_links,
            'max_depth': self.max_depth,
            'sparse_content_threshold': self.sparse_content_threshold,
            'mean_similarity_threshold': self.mean_similarity_threshold,
            'anti_detection_enabled': True,
            'user_agents_available': len(self.user_agents),
            'header_pools_available': len(self.headers_pool),
            'stealth_mode_enabled': True,
            'random_delays_enabled': True,
            'multi_strategy_enabled': True
        }
    
    def reset_scraping_session(self):
        """
        Reset scraping session and reinitialize anti-detection measures.
        """
        self.__reset_anchor_links()
        self._randomize_session()
        print("🔄 Scraping session reset with new anti-detection settings")
    
    # Enhanced scraping method with comprehensive error handling
    def enhanced_scrape(self, url=None, query=None):
        """
        Enhanced scraping method with retry logic and comprehensive error handling.
        
        Args:
            url (str): URL to scrape.
            query (str): Query for relevance checking.
            max_retries (int): Maximum number of retry attempts.
            
        Returns:
            dict: Scraping results or None if failed.
        """
        if url is None:
            url = self.url

        for attempt in range(self.max_retries):
            try:
                print(f"🚀 Enhanced scrape attempt {attempt + 1}/{self.max_retries} for: {url}")

                # Reset session for each retry
                if attempt > 0:
                    self._randomize_session()
                    print(f"🔄 Retry {attempt + 1}: Using new anti-detection settings")
                
                # Perform the scrape
                result = self.scrape(url, query)
                
                if result and result.get('total_chunks', 0) > 0:
                    print(f"✅ Enhanced scrape successful on attempt {attempt + 1}")
                    return result
                else:
                    print(f"⚠️ Attempt {attempt + 1} returned no valid chunks")
                    
            except Exception as e:
                print(f"❌ Enhanced scrape attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Wait progressively longer between retries
                    self._random_delay(2 * (attempt + 1), 4 * (attempt + 1))
                else:
                    print(f"❌ All {self.max_retries} attempts failed for {url}")
                    
        return None

