import requests
from bs4 import BeautifulSoup as bs
from urllib.parse import urlparse, urljoin
from time import sleep
from playwright.sync_api import sync_playwright
from dataHandler import DataHandler
import os
import numpy as np


class Scraper:
    def __init__(self, url=None, max_retries=3, max_depth=1, sparse_content_threshold=750, mean_similarity_threshold=0.15, max_anchor_links=10):
        """
        Initialize the Scraper with a URL.
        """
        self.url = url
        self.base_url = self._get_base_url(url) if url else None
        self.max_retries = max_retries
        self.max_depth = max_depth
        self.data_handler = DataHandler()
        self.sparse_content_threshold = sparse_content_threshold
        self.mean_similarity_threshold = mean_similarity_threshold
        self.max_anchor_links = max_anchor_links  # Maximum number of anchor links to scrape per page
        self.current_anchor_links = 0  # Track current anchor links to avoid exceeding max_anchor_links
        # Headers to mimic a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

    def _fetch_content(self, url, use_js=False):
        """
        Fetch the content of the given URL.
        Args:
            url (str): The URL to fetch.
            use_js (bool): Whether to render JavaScript or not.
        Returns:
            str: The HTML content of the page.
        """
        try:
            if use_js:
                return self._fetch_js_content_playwright(url)
            else:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                return response.content
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            self.current_anchor_links -= 1  # Decrement anchor links count
            return ""

    def _fetch_js_content_playwright(self, url):
        """
        Fetch content from JavaScript-heavy pages using Playwright.
        This is much more reliable than requests-html.
        """
        
        try:
            print(f"Rendering JavaScript with Playwright for {url}...")
            
            with sync_playwright() as p:
                # Launch browser (headless by default)
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
                page = context.new_page()
                
                # Navigate to the page
                page.goto(url, wait_until='networkidle', timeout=30000)
                
                # Wait for content to load
                page.wait_for_selector("body", timeout=10000)
                
                # Scroll down to trigger lazy loading
                for i in range(3):
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(1000)
                
                # Get the final HTML
                html_content = page.content()
                
                browser.close()
                return html_content
                
        except Exception as e:
            print(f"Error rendering JavaScript with Playwright for {url}: {e}")
            print("Falling back to regular requests...")
            return self._fetch_content(url, use_js=False)
        
    def __get_anchor_tags(self, soup: bs):
        """
        Extract and resolve all internal anchor tag URLs from the given BeautifulSoup object.
        
        Args:
            soup (BeautifulSoup): The parsed HTML content.
        
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
            self.base_url = self._get_base_url(self.url)
        
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
        # First let's check if there is an http or https in the url
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
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
        if not is_relevant:
            self.current_anchor_links -= 1  # Decrement anchor links count
            print(f"Content at {url} is not relevant to the query '{query}'. Skipping.")
            return [] if not is_root else [text_content]
        # If depth exceeded, just return this page's text
        if current_depth >= self.max_depth:
                return [text_content]
        # Extract internal anchor tags
        anchor_tags = self.__get_anchor_tags(soup)
        if not anchor_tags:
            print("No internal links found.")


        # Collect content from this page
        contents = [text_content]

        for i, anchor in enumerate(anchor_tags):
            if self.current_anchor_links >= self.max_anchor_links:
                print(f"Reached maximum anchor links limit ({self.max_anchor_links}). Stopping further scraping.")
                break
            self.current_anchor_links += 1
            sleep(0.5)  # Slightly longer delay for JS rendering
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
    def scrape_js_heavy(self, url=None):
        """
        Scrape a JavaScript-heavy page using Playwright.
        This method specifically handles sites that rely on JavaScript for content rendering.
        
        Args:
            url (str): URL to scrape. If None, uses self.url.
            
        Returns:
            str: The content of the scraped page after JavaScript execution.
        """
        if url is None:
            url = self.url
            
        content_html = self._fetch_js_content_playwright(url)
        if content_html:
            soup = bs(content_html, 'html.parser')
            return soup.get_text()
        return ""

    # Keep the original scrape method for backward compatibility
    def scrape(self, url=None , query=None, current_depth=0):
        """
        Original scrape method - now calls smart_scrape for better handling.
        """
        data = self.smart_scrape(url, query , current_depth) 
        self.current_anchor_links = 0  # Reset anchor links count after scraping
        return data
    
# Example usage
if __name__ == "__main__":
    
    # Create a directory to store scraped content
    output_dir = "scraped_content"
    
    URL = "https://www.w3schools.com/css/css_align.asp"
    scraper = Scraper(url=URL, max_retries=3, max_depth=3, sparse_content_threshold=750, mean_similarity_threshold=0.2, max_anchor_links=6)
    
    # Use smart scraping (automatically detects JS-heavy sites)
    print("=== Smart Scraping ===")
    contents = scraper.smart_scrape(query="How do I center a div in Tailwind CSS?")
    print(f"Scraped {contents} pieces of content.")