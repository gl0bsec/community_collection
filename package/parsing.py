import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from threading import Lock
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
import json
import io

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

from urlextract import URLExtract

warnings.filterwarnings('ignore')

##Text and URL extraction from content columns 
def parse_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Parse Discord message, extracting content, URLs, and embed metadata.
    
    Args:
        message: Discord message dictionary
        
    Returns:
        Dict with id, timestamp, content (URLs stripped), urls list, and embeds
    """
    content = message.get("content", "") or ""
    urls = re.findall(r"https?://\S+", content)
    content_without_urls = re.sub(r"https?://\S+", "", content).strip()

    embeds_metadata = []
    for embed in message.get("embeds", []):
        metadata = {
            "url": embed.get("url"),
            "title": embed.get("title"),
            "description": embed.get("description"),
            "thumbnail_url": embed.get("thumbnail", {}).get("url") if embed.get("thumbnail") else None,
            "author_name": embed.get("author", {}).get("name") if embed.get("author") else None,
        }
        embeds_metadata.append(metadata)

    return {
        "id": message.get("id"),
        "timestamp": message.get("timestamp"),
        "content": content_without_urls,
        "urls": urls,
        "embeds": embeds_metadata,
    }


def create_combined_content(df: pd.DataFrame) -> pd.DataFrame:
    """Add combined_content column merging embeds and content.
    
    Args:
        df: DataFrame with 'embeds' and 'content' columns
        
    Returns:
        DataFrame with added 'combined_content' column
    """
    df = df.copy()
    df["combined_content"] = df["embeds"].astype(str) + " " + df["content"].astype(str)
    return df


def parse_message_content(df: pd.DataFrame, 
                         url_column: str, 
                         date_column: str,
                         delay: float = 0.1,
                         timeout: int = 10,
                         max_retries: int = 3,
                         max_workers: int = 10,
                         cache_duplicates: bool = True) -> pd.DataFrame:
    """Extract URLs from DataFrame and fetch their metadata using concurrent processing.
    
    Args:
        df: DataFrame with URL and date columns
        url_column: Column name containing URLs
        date_column: Column name containing dates
        delay: Seconds between requests per domain (default: 0.1)
        timeout: Request timeout seconds (default: 10)
        max_retries: Max retry attempts (default: 3)
        max_workers: Number of concurrent workers (default: 10)
        cache_duplicates: Whether to cache duplicate URL results (default: True)
        
    Returns:
        DataFrame with columns: url, date, title, description, domain, status
    """
    
    # Shared session for connection pooling
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    # URL cache and domain rate limiting
    url_cache = {} if cache_duplicates else None
    domain_locks = {}
    domain_last_request = {}
    cache_lock = Lock()
    
    def extract_urls_from_text(text: Union[str, Any]) -> List[str]:
        """Extract all URLs from text string using improved methods."""
        if pd.isna(text) or not text:
            return []
        
        text_str = str(text)
        
        # Use URLExtract library for better URL detection
        extractor = URLExtract()
        urls = extractor.find_urls(text_str)
        # Filter to only http/https URLs
        urls = [url for url in urls if url.startswith(('http://', 'https://'))]
        
        # Clean and validate URLs
        cleaned_urls = []
        for url in urls:
            # Remove trailing punctuation that's not part of URL
            url = url.rstrip('.,;!?)}]')
            # Basic validation
            if len(url) > 10 and '.' in url:
                cleaned_urls.append(url)
        
        return cleaned_urls
    
    def get_domain(url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return ""
    
    def get_page_metadata_cached(url: str, timeout: int = 10, max_retries: int = 3) -> Tuple[str, str, str]:
        """Fetch page title and description from URL with caching and rate limiting."""
        
        # Check cache first
        if url_cache is not None:
            with cache_lock:
                if url in url_cache:
                    cached_result = url_cache[url]
                    return cached_result['title'], cached_result['description'], cached_result['status']
        
        domain = get_domain(url)
        
        # Domain-specific rate limiting
        if domain:
            if domain not in domain_locks:
                domain_locks[domain] = Lock()
            
            with domain_locks[domain]:
                # Check if we need to wait for this domain
                if domain in domain_last_request:
                    time_since_last = time.time() - domain_last_request[domain]
                    if time_since_last < delay:
                        time.sleep(delay - time_since_last)
                
                # Fetch metadata
                title, description, status = get_page_metadata(url, timeout, max_retries)
                
                # Update last request time for this domain
                domain_last_request[domain] = time.time()
        else:
            title, description, status = get_page_metadata(url, timeout, max_retries)
        
        # Cache result
        if url_cache is not None:
            with cache_lock:
                url_cache[url] = {
                    'title': title,
                    'description': description,
                    'status': status
                }
        
        return title, description, status
    
    def get_page_metadata(url: str, timeout: int = 10, max_retries: int = 3) -> Tuple[str, str, str]:
        """Fetch page title and description from URL with enhanced error handling."""
        
        for attempt in range(max_retries):
            try:
                response = session.get(url, timeout=timeout)
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    if attempt < max_retries - 1:
                        time.sleep(min(retry_after, 60))  # Cap at 60 seconds
                        continue
                    return "", "", "rate_limited"
                
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                title = ""
                if soup.title and hasattr(soup.title, 'string') and soup.title.string:
                    title = str(soup.title.string).strip()
                
                description = ""
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if not meta_desc:
                    meta_desc = soup.find('meta', attrs={'property': 'og:description'})
                if not meta_desc:
                    meta_desc = soup.find('meta', attrs={'name': 'twitter:description'})
                
                if meta_desc and isinstance(meta_desc, Tag):
                    content = meta_desc.get('content', '')
                    description = str(content).strip() if content else ''
                
                return title, description, "success"
                
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    return "", "", "timeout"
                time.sleep(min(2 ** attempt, 10))  # Exponential backoff, capped at 10s
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return "", "", f"error: {type(e).__name__}"
                time.sleep(min(2 ** attempt, 10))  # Exponential backoff
        
        return "", "", "max_retries_exceeded"
    
    def process_url_batch(url_batch: List[Tuple[int, str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of URLs concurrently."""
        results = []
        
        for idx, url, date_value in url_batch:
            domain = get_domain(url)
            title, description, status = get_page_metadata_cached(url, timeout, max_retries)
            
            results.append({
                'url': url,
                'date': date_value,
                'title': title,
                'description': description,
                'domain': domain,
                'status': status,
                'original_row_index': idx
            })
        
        return results
    
    # Validate input columns
    if url_column not in df.columns:
        raise ValueError(f"Column '{url_column}' not found in dataframe")
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in dataframe")
    
    # First pass: collect all URLs
    all_urls = []
    for idx, row in df.iterrows():
        text_content = row[url_column]
        urls = extract_urls_from_text(text_content)
        date_value = row[date_column]
        
        for url in urls:
            url = url.rstrip('.,;!?)')
            all_urls.append((idx, url, date_value))
    
    if not all_urls:
        print("No URLs found in the dataset.")
        return pd.DataFrame()
    
    # Remove duplicates if caching is enabled
    if cache_duplicates:
        unique_urls = []
        seen_urls = set()
        for item in all_urls:
            if item[1] not in seen_urls:
                unique_urls.append(item)
                seen_urls.add(item[1])
        print(f"Found {len(all_urls)} URLs ({len(unique_urls)} unique) from {len(df)} rows.")
        all_urls = unique_urls
    else:
        print(f"Found {len(all_urls)} URLs from {len(df)} rows.")
    
    # Process URLs concurrently with progress bar
    results = []
    failed_urls = []
    
    # Split URLs into batches for concurrent processing
    batch_size = max(1, len(all_urls) // max_workers)
    url_batches = [all_urls[i:i + batch_size] for i in range(0, len(all_urls), batch_size)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        futures = []
        for batch in url_batches:
            future = executor.submit(process_url_batch, batch)
            futures.append(future)
        
        # Process results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing URL batches"):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                print(f"Batch processing error: {e}")
    
    # Close session
    session.close()
    
    # Create DataFrame
    result_df = pd.DataFrame(results)
    
    if not result_df.empty:
        try:
            result_df['date'] = pd.to_datetime(result_df['date'])
        except Exception:
            pass
        
        # Collect failed URLs
        failed_urls = result_df[result_df['status'] != 'success']
    
    # Print statistics
    if not result_df.empty:
        success_count = len(result_df[result_df['status'] == 'success'])
        print(f"\nExtraction complete! Success rate: {success_count}/{len(result_df)} URLs ({success_count/len(result_df)*100:.1f}%)")
        
        if len(failed_urls) > 0:
            print(f"Failed URLs by reason:")
            failure_counts = failed_urls['status'].value_counts()
            for reason, count in failure_counts.items():
                print(f"  - {reason}: {count}")
    else:
        print("No URLs processed.")
    
    return result_df


def combine_multiple_columns(df, column_names, new_col_name='combined_text',
                           include_labels=True, separator='\n\n'):
    """Combine multiple text columns into a new formatted column.

    Args:
        df: Input DataFrame
        column_names: List of column names to combine
        new_col_name: Name for new combined column
        include_labels: Whether to include column names as labels
        separator: Separator between text sections

    Returns:
        DataFrame with new combined column added
    """
    result_df = df.copy()

    # Validate column names
    for col_name in column_names:
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in dataframe")

    def format_content(row):
        """Format the content for a single row"""
        parts = []
        for col_name in column_names:
            val = row[col_name]
            if pd.notna(val) and str(val).strip():
                if include_labels:
                    parts.append(f"{col_name}: {val}")
                else:
                    parts.append(str(val))
        return separator.join(parts) if parts else ""

    result_df[new_col_name] = df.apply(format_content, axis=1)
    return result_df


def extract_text_content(df: pd.DataFrame, 
                        text_column: str, 
                        date_column: str,
                        min_text_length: int = 5,
                        clean_whitespace: bool = True) -> pd.DataFrame:
    """Extract text content from column while excluding all URLs.
    
    Args:
        df: Input DataFrame containing text and dates
        text_column: Column name containing text (potentially with URLs)
        date_column: Column name containing dates
        min_text_length: Minimum length of text to include
        clean_whitespace: Whether to clean up extra whitespace
    
    Returns:
        DataFrame with columns: cleaned_text, original_text, date
    """
    
    def remove_urls_from_text(text: str) -> Tuple[str, int]:
        """Remove all URLs from text and return cleaned text with count."""
        if pd.isna(text):
            return "", 0
        
        text_str = str(text)
        
        # Use URLExtract for more robust URL detection
        extractor = URLExtract()
        urls_found = extractor.find_urls(text_str)
        
        # Filter to common URL schemes and clean URLs
        cleaned_urls = []
        for url in urls_found:
            # Remove trailing punctuation that's not part of URL
            original_url = url
            url = url.rstrip('.,;!?)}]"\'')
            
            # Only include http/https/ftp URLs and common www patterns
            if (url.startswith(('http://', 'https://', 'ftp://', 'www.')) or 
                ('.com' in url or '.org' in url or '.net' in url or '.edu' in url or 
                 '.gov' in url or '.io' in url or '.co' in url)):
                cleaned_urls.append((original_url, url))
        
        # Remove URLs from text while preserving sentence structure
        cleaned_text = text_str
        for original_url, _ in cleaned_urls:
            # Check if URL is at start/end of sentence or standalone
            url_pattern = re.escape(original_url)
            
            # Handle URLs at sentence boundaries more gracefully
            # Replace URL with appropriate spacing based on context
            if re.search(r'^\s*' + url_pattern + r'\s*$', cleaned_text, re.MULTILINE):
                # URL is on its own line - remove entirely
                cleaned_text = re.sub(r'^\s*' + url_pattern + r'\s*$', '', cleaned_text, flags=re.MULTILINE)
            elif re.search(r'^\s*' + url_pattern + r'\s+', cleaned_text):
                # URL at start of sentence - remove but keep following content
                cleaned_text = re.sub(r'^\s*' + url_pattern + r'\s+', '', cleaned_text)
            elif re.search(r'\s+' + url_pattern + r'\s*$', cleaned_text):
                # URL at end of sentence - remove but preserve preceding content
                cleaned_text = re.sub(r'\s+' + url_pattern + r'\s*$', '', cleaned_text)
            elif re.search(r'\.\s*' + url_pattern + r'\s', cleaned_text):
                # URL after sentence end - remove cleanly
                cleaned_text = re.sub(r'\.\s*' + url_pattern + r'\s+', '. ', cleaned_text)
            else:
                # URL in middle of text - replace with single space
                cleaned_text = re.sub(url_pattern, ' ', cleaned_text)
        
        return cleaned_text, len(cleaned_urls)
    
    def clean_text(text: str) -> str:
        """Clean up text by removing extra whitespace and normalizing."""
        if not text:
            return ""
        
        # Remove empty lines and normalize line breaks
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned = ' '.join(lines)
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Remove multiple punctuation marks
        cleaned = re.sub(r'[.]{2,}', '.', cleaned)
        cleaned = re.sub(r'[!]{2,}', '!', cleaned)
        cleaned = re.sub(r'[?]{2,}', '?', cleaned)
        cleaned = re.sub(r'[-]{2,}', '-', cleaned)
        
        # Clean up common artifacts from URL removal
        cleaned = re.sub(r'\s+[.,;!?]\s+', ' ', cleaned)
        cleaned = re.sub(r'\s+[.,;!?]$', '', cleaned)  # Remove trailing punctuation with spaces
        cleaned = re.sub(r'^[.,;!?]\s+', '', cleaned)  # Remove leading punctuation with spaces
        
        # Fix spacing around punctuation
        cleaned = re.sub(r'\s+([.,;!?])', r'\1', cleaned)  # Remove spaces before punctuation
        cleaned = re.sub(r'([.,;!?])([^\s])', r'\1 \2', cleaned)  # Add space after punctuation if missing
        
        # Handle common text artifacts
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single space
        cleaned = re.sub(r'\s*\.\s*\.', '.', cleaned)  # Fix broken ellipsis
        cleaned = re.sub(r'\s*-\s*-', '-', cleaned)  # Fix broken dashes
        
        # Remove isolated punctuation
        cleaned = re.sub(r'\s+[.,;!?]\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    # Validate input columns
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataframe")
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in dataframe")
    
    results = []
    print(f"Processing {len(df)} rows for text extraction...")
    
    for _, row in df.iterrows():
        original_text = row[text_column]
        date_value = row[date_column]
        
        try:
            is_na = bool(pd.isna(original_text))
        except (ValueError, TypeError):
            is_na = False
        
        if original_text is None or is_na or str(original_text).strip() == "":
            continue
        
        original_text_str = str(original_text)
        cleaned_text, _ = remove_urls_from_text(original_text_str)
        
        if clean_whitespace:
            cleaned_text = clean_text(cleaned_text)
        
        cleaned_length = len(cleaned_text)
        
        if cleaned_length >= min_text_length:
            results.append({
                'cleaned_text': cleaned_text,
                'original_text': original_text_str,
                'date': date_value
            })
    
    result_df = pd.DataFrame(results)
    
    if not result_df.empty:
        try:
            result_df['date'] = pd.to_datetime(result_df['date'])
        except Exception:
            pass
    
    print(f"Text extraction complete! Processed {len(result_df)} rows with sufficient text content.")
    
    return result_df


def analyze_text_results(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze results from extract_text_content function.
    
    Args:
        df: DataFrame from extract_text_content
        
    Returns:
        Same DataFrame (for chaining)
    """
    if df.empty:
        return pd.DataFrame()
    
    analysis = {
        'total_text_entries': len(df),
        'avg_original_length': df['original_text'].str.len().mean(),
        'avg_cleaned_length': df['cleaned_text'].str.len().mean(),
        'date_range': f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else 'N/A'
    }
    
    if 'date' in df.columns:
        try:
            entries_by_date = df.groupby(df['date'].dt.date).size() if pd.api.types.is_datetime64_any_dtype(df['date']) else df.groupby('date').size()
        except Exception:
            entries_by_date = df.groupby('date').size()
    else:
        entries_by_date = pd.Series()
    
    print("=== Text Extraction Analysis ===")
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    if not entries_by_date.empty:
        print(f"\nText entries by date:")
        print(entries_by_date.head(10))
    
    return df


#GDELT data parsing 
def extract_url_metadata(url: str, timeout: int = 0) -> Dict[str, Optional[str]]:
    """Extract comprehensive metadata from a webpage URL.
    
    Args:
        url: URL to extract metadata from
        timeout: Request timeout in seconds
        
    Returns:
        Dict with metadata fields (title, description, keywords, etc.)
    """
    metadata = {
        'url': url,
        'title': None,
        'description': None,
        'keywords': None,
        'author': None,
        'site_name': None,
        'image': None,
        'favicon': None,
        'canonical_url': None,
        'language': None,
        'content_type': None,
        'status_code': None,
        'error': None
    }

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        metadata['status_code'] = response.status_code
        metadata['content_type'] = response.headers.get('content-type', '')
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()

        # Extract meta tags
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            if not isinstance(tag, Tag):
                continue
            name = str(tag.get('name', '')).lower()
            prop = str(tag.get('property', '')).lower()
            content_attr = tag.get('content', '')
            content = str(content_attr).strip() if content_attr else ''
            
            if name == 'description' and not metadata['description']:
                metadata['description'] = content
            elif name == 'keywords':
                metadata['keywords'] = content
            elif name == 'author':
                metadata['author'] = content
            elif name == 'language':
                metadata['language'] = content
            elif prop == 'og:title' and not metadata['title']:
                metadata['title'] = content
            elif prop == 'og:description' and not metadata['description']:
                metadata['description'] = content
            elif prop == 'og:site_name':
                metadata['site_name'] = content
            elif prop == 'og:image':
                metadata['image'] = content
            elif name in ['twitter:title'] and not metadata['title']:
                metadata['title'] = content
            elif name in ['twitter:description'] and not metadata['description']:
                metadata['description'] = content
            elif name in ['twitter:image'] and not metadata['image']:
                metadata['image'] = content

        # Extract canonical URL
        canonical_link = soup.find('link', {'rel': 'canonical'})
        if canonical_link and isinstance(canonical_link, Tag):
            href = canonical_link.get('href', '')
            metadata['canonical_url'] = str(href).strip() if href else ''

        # Extract favicon
        favicon_link = soup.find('link', {'rel': 'icon'}) or soup.find('link', {'rel': 'shortcut icon'})
        if favicon_link and isinstance(favicon_link, Tag):
            favicon_href = favicon_link.get('href', '')
            if favicon_href:
                metadata['favicon'] = urljoin(url, str(favicon_href))

        # Extract language from html tag if not found in meta
        if not metadata['language']:
            html_tag = soup.find('html')
            if html_tag and isinstance(html_tag, Tag):
                lang = html_tag.get('lang', '')
                metadata['language'] = str(lang).strip() if lang else ''

        # Convert relative URLs to absolute
        if metadata['image'] and not metadata['image'].startswith(('http://', 'https://')):
            metadata['image'] = urljoin(url, metadata['image'])

        if metadata['canonical_url'] and not metadata['canonical_url'].startswith(('http://', 'https://')):
            metadata['canonical_url'] = urljoin(url, metadata['canonical_url'])

    except requests.exceptions.RequestException as e:
        metadata['error'] = f"Request error: {str(e)}"
    except Exception as e:
        metadata['error'] = f"Parsing error: {str(e)}"

    # Clean up empty values
    metadata = {k: v if v else None for k, v in metadata.items()}
    return metadata


def get_source_urls_with_metadata(df, actor1_code=None, actor2_code=None, geo_code=None,
                                 match_type='any', limit=None, show_events=True,
                                 extract_metadata=False, max_workers=5, delay=1.0,
                                 timeout=10, dataF=False):
    """Get source URLs with optional metadata extraction for GDELT events.

    Args:
        df: GDELT DataFrame
        actor1_code: Country code for Actor1 (None = any)
        actor2_code: Country code for Actor2 (None = any)
        geo_code: Country code for event location (None = any)
        match_type: 'any' (OR) or 'all' (AND) for multiple conditions
        limit: Maximum number of rows/URLs to return
        show_events: Whether to include event descriptions
        extract_metadata: Whether to extract webpage metadata
        max_workers: Number of concurrent threads for metadata extraction
        delay: Delay between requests in seconds
        timeout: Request timeout in seconds
        dataF: Return DataFrame with full event details and metadata

    Returns:
        DataFrame, list of tuples, or list of URLs depending on parameters
    """
    # Build conditions
    conditions = []
    if actor1_code:
        conditions.append(df['Actor1CountryCode'] == actor1_code)
    if actor2_code:
        conditions.append(df['Actor2CountryCode'] == actor2_code)
    if geo_code:
        conditions.append(df['ActionGeo_CountryCode'] == geo_code)

    # Apply conditions
    if not conditions:
        print("No filters specified. Processing entire dataset...")
        filtered_df = df.copy()
    else:
        if match_type == 'any':
            mask = conditions[0]
            for condition in conditions[1:]:
                mask = mask | condition
        else:
            mask = conditions[0]
            for condition in conditions[1:]:
                mask = mask & condition
        filtered_df = df[mask].copy()

    if len(filtered_df) == 0:
        print("No events found matching the specified criteria.")
        return pd.DataFrame() if dataF else []

    if dataF:
        # Aggregate GDELT variables by URL
        print("Aggregating GDELT events by URL...")
        filtered_df['SQLDATE_dt'] = pd.to_datetime(filtered_df['SQLDATE'], format='%Y%m%d')

        agg_dict = {
            'GoldsteinScale': ['mean', 'min', 'max', 'std', 'count'],
            'Actor1Name': lambda x: ' | '.join(x.dropna().unique()[:5]),
            'Actor2Name': lambda x: ' | '.join(x.dropna().unique()[:5]),
            'Actor1CountryCode': lambda x: ' | '.join(x.dropna().unique()),
            'Actor2CountryCode': lambda x: ' | '.join(x.dropna().unique()),
            'ActionGeo_CountryCode': lambda x: ' | '.join(x.dropna().unique()),
            'SQLDATE_dt': ['min', 'max'],
            'EventDescription': lambda x: list(x.dropna().unique()) if 'EventDescription' in filtered_df.columns else []
        }

        result_df = filtered_df.groupby('SOURCEURL').agg(agg_dict).reset_index()
        result_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in result_df.columns.values]

        # Rename columns for clarity
        column_mapping = {
            'GoldsteinScale_mean': 'avg_goldstein_score',
            'GoldsteinScale_min': 'min_goldstein_score',
            'GoldsteinScale_max': 'max_goldstein_score',
            'GoldsteinScale_std': 'goldstein_score_std',
            'GoldsteinScale_count': 'event_count',
            'Actor1Name_<lambda>': 'actor1_names',
            'Actor2Name_<lambda>': 'actor2_names',
            'Actor1CountryCode_<lambda>': 'actor1_countries',
            'Actor2CountryCode_<lambda>': 'actor2_countries',
            'ActionGeo_CountryCode_<lambda>': 'event_locations',
            'SQLDATE_dt_min': 'first_event_date',
            'SQLDATE_dt_max': 'last_event_date',
            'EventDescription_<lambda>': 'event_descriptions'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in result_df.columns:
                result_df = result_df.rename(columns={old_col: new_col})

        # Convert dates and add derived metrics
        if 'first_event_date' in result_df.columns:
            result_df['first_event_date'] = result_df['first_event_date'].dt.strftime('%Y-%m-%d')
        if 'last_event_date' in result_df.columns:
            result_df['last_event_date'] = result_df['last_event_date'].dt.strftime('%Y-%m-%d')

        if 'goldstein_score_std' in result_df.columns:
            result_df['goldstein_score_std'] = result_df['goldstein_score_std'].fillna(0)

        if 'event_count' in result_df.columns:
            result_df = result_df.sort_values('event_count', ascending=False)

        if limit:
            result_df = result_df.head(limit)

        # Extract metadata if requested
        if extract_metadata:
            print(f"Extracting metadata for {len(result_df)} URLs...")
            metadata_results = []

            def extract_metadata_delayed(url):
                time.sleep(delay)
                return extract_url_metadata(url, timeout)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_url = {executor.submit(extract_metadata_delayed, url): url
                               for url in result_df['SOURCEURL'].unique()}

                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        metadata = future.result()
                        metadata_results.append(metadata)
                    except Exception as exc:
                        print(f'URL {url} generated an exception: {exc}')
                        metadata_results.append({'url': url, 'error': str(exc)})

            metadata_df = pd.DataFrame(metadata_results)
            result_df = result_df.merge(metadata_df, left_on='SOURCEURL', right_on='url', how='left')

            if 'url' in result_df.columns:
                result_df = result_df.drop('url', axis=1)

        print(f"Found {len(result_df)} unique URLs from {len(filtered_df)} events")
        return result_df

    elif show_events:
        url_events = filtered_df.groupby('SOURCEURL').agg({
            'EventDescription': lambda x: list(x.dropna().unique()) if 'EventDescription' in filtered_df.columns else [],
            'SQLDATE': 'first'
        }).reset_index()

        url_events['SQLDATE'] = pd.to_datetime(url_events['SQLDATE'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        url_events['event_count'] = url_events['EventDescription'].apply(len)
        url_events = url_events.sort_values('event_count', ascending=False)

        if limit:
            url_events = url_events.head(limit)

        print(f"Found {len(filtered_df)} events with {len(url_events)} unique URLs")

        if extract_metadata:
            print(f"Extracting metadata for {len(url_events)} URLs...")

            def extract_metadata_with_context(row):
                time.sleep(delay)
                url, events, date = row['SOURCEURL'], row['EventDescription'], row['SQLDATE']
                metadata = extract_url_metadata(url, timeout)
                return (url, events, date, metadata)

            results_with_metadata = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(extract_metadata_with_context, row)
                          for _, row in url_events.iterrows()]

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results_with_metadata.append(result)
                    except Exception as exc:
                        print(f'Metadata extraction failed: {exc}')

            return results_with_metadata
        else:
            return list(zip(url_events['SOURCEURL'], url_events['EventDescription'], url_events['SQLDATE']))

    else:
        urls = filtered_df['SOURCEURL'].dropna().unique()
        if limit:
            urls = urls[:limit]

        print(f"Found {len(filtered_df)} events with {len(urls)} unique URLs")

        if extract_metadata:
            print(f"Extracting metadata for {len(urls)} URLs...")

            def extract_url_delayed(url):
                time.sleep(delay)
                return (url, extract_url_metadata(url, timeout))

            results_with_metadata = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(extract_url_delayed, url) for url in urls]

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results_with_metadata.append(result)
                    except Exception as exc:
                        print(f'Metadata extraction failed: {exc}')

            return results_with_metadata
        else:
            return urls.tolist()


def analyze_url_results(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze and print statistics from parse_message_content results.
    
    Args:
        df: DataFrame from parse_message_content
        
    Returns:
        Same DataFrame (for chaining)
    """
    if df.empty:
        return pd.DataFrame()
    
    analysis = {
        'total_urls': len(df),
        'successful_requests': len(df[df['status'] == 'success']),
        'failed_requests': len(df[df['status'] != 'success']),
        'unique_domains': df['domain'].nunique(),
        'date_range': f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else 'N/A'
    }
    
    top_domains = df['domain'].value_counts().head(10)
    
    if 'date' in df.columns:
        try:
            urls_by_date = df.groupby(df['date'].dt.date).size() if pd.api.types.is_datetime64_any_dtype(df['date']) else df.groupby('date').size()
        except Exception:
            urls_by_date = df.groupby('date').size()
    else:
        urls_by_date = pd.Series()
    
    print("=== URL Extraction Analysis ===")
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    print(f"\nTop domains:")
    print(top_domains)
    
    if not urls_by_date.empty:
        print(f"\nURLs by date:")
        print(urls_by_date.head(10))
    
    return df


def analyze_source_metadata(df_with_metadata):
    """Analyze extracted metadata to provide insights.
    
    Args:
        df_with_metadata: DataFrame with metadata columns
    """
    if 'site_name' not in df_with_metadata.columns:
        print("No metadata found in DataFrame")
        return

    print("=== Metadata Analysis ===")

    # Most common news sources
    print("\nTop news sources:")
    site_counts = df_with_metadata['site_name'].value_counts().head(10)
    for site, count in site_counts.items():
        if site:
            print(f"  {site}: {count} articles")

    # Language distribution
    if 'language' in df_with_metadata.columns:
        print("\nLanguage distribution:")
        lang_counts = df_with_metadata['language'].value_counts().head(5)
        for lang, count in lang_counts.items():
            if lang:
                print(f"  {lang}: {count} articles")

    # Success rate
    total_urls = len(df_with_metadata)
    successful = len(df_with_metadata[df_with_metadata['title'].notna()])
    print(f"\nMetadata extraction success rate: {successful}/{total_urls} ({successful/total_urls*100:.1f}%)")

#RSS feed parsing 
def parse_rss_feed(rss_url: str, timeout: int = 10) -> pd.DataFrame:
    """Parse an RSS feed into a pandas DataFrame with individual fields for metadata and content.
    
    Args:
        rss_url: URL of the RSS feed to parse
        timeout: Request timeout in seconds (default: 10)
        
    Returns:
        DataFrame with columns: title, description, link, published_date, guid, author, 
                              categories, content, feed_title, feed_description, feed_link
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(rss_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        
        # Initialize feed-level metadata
        feed_info: Dict[str, Optional[str]] = {
            'feed_title': None,
            'feed_description': None,
            'feed_link': None,
            'feed_language': None,
            'feed_last_build_date': None
        }
        
        # Extract feed-level information
        channel = root.find('channel')
        if channel is not None:
            feed_info['feed_title'] = _get_text(channel.find('title'))
            feed_info['feed_description'] = _get_text(channel.find('description'))
            feed_info['feed_link'] = _get_text(channel.find('link'))
            feed_info['feed_language'] = _get_text(channel.find('language'))
            feed_info['feed_last_build_date'] = _get_text(channel.find('lastBuildDate'))
        
        # Extract items
        items = []
        item_elements = root.findall('.//item')
        
        for item in item_elements:
            item_data = {
                'title': _get_text(item.find('title')),
                'description': _get_text(item.find('description')),
                'link': _get_text(item.find('link')),
                'published_date': _parse_date(_get_text(item.find('pubDate'))),
                'guid': _get_text(item.find('guid')),
                'author': _get_text(item.find('author')),
                'categories': _get_categories(item),
                'content': _get_content(item),
                'comments': _get_text(item.find('comments')),
                'enclosure_url': _get_enclosure_url(item),
                'enclosure_type': _get_enclosure_type(item),
                'enclosure_length': _get_enclosure_length(item)
            }
            
            # Add feed-level metadata to each item
            item_data.update(feed_info)
            items.append(item_data)
        
        df = pd.DataFrame(items)
        
        # Convert published_date to datetime if possible
        if not df.empty and 'published_date' in df.columns:
            df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching RSS feed: {e}")
        return pd.DataFrame()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error: {e}")
        return pd.DataFrame()


def _get_text(element) -> Optional[str]:
    """Extract text content from XML element."""
    if element is not None and element.text:
        return element.text.strip()
    return None


def _get_categories(item) -> List[str]:
    """Extract all categories from an RSS item."""
    categories = []
    for category in item.findall('category'):
        if category.text:
            categories.append(category.text.strip())
    return categories


def _get_content(item) -> Optional[str]:
    """Extract content from RSS item, checking multiple possible tags."""
    # Try content:encoded first (common in WordPress feeds)
    content_encoded = item.find('{http://purl.org/rss/1.0/modules/content/}encoded')
    if content_encoded is not None and content_encoded.text:
        return content_encoded.text.strip()
    
    # Try description as fallback
    description = item.find('description')
    if description is not None and description.text:
        return description.text.strip()
    
    return None


def _get_enclosure_url(item) -> Optional[str]:
    """Extract enclosure URL from RSS item."""
    enclosure = item.find('enclosure')
    if enclosure is not None:
        return enclosure.get('url')
    return None


def _get_enclosure_type(item) -> Optional[str]:
    """Extract enclosure type from RSS item."""
    enclosure = item.find('enclosure')
    if enclosure is not None:
        return enclosure.get('type')
    return None


def _get_enclosure_length(item) -> Optional[str]:
    """Extract enclosure length from RSS item."""
    enclosure = item.find('enclosure')
    if enclosure is not None:
        return enclosure.get('length')
    return None


def _parse_date(date_string: Optional[str]) -> Optional[str]:
    """Parse RSS date string to ISO format."""
    if not date_string:
        return None
    
    # Common RSS date formats
    formats = [
        '%a, %d %b %Y %H:%M:%S %z',  # RFC 2822
        '%a, %d %b %Y %H:%M:%S GMT',
        '%a, %d %b %Y %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S%z',       # ISO 8601
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S'
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_string.strip(), fmt)
            return dt.isoformat()
        except ValueError:
            continue
    
    # If no format matches, return original string
    return date_string


#Google sheets parsing 
def parse_google_sheets(sheets_url: str, 
                       credentials_path: Optional[str] = None,
                       sheet_name: Optional[str] = None,
                       timeout: int = 10) -> pd.DataFrame:
    """Parse a Google Sheets URL into a pandas DataFrame.
    
    This function supports both public and private Google Sheets:
    - Public sheets: Uses CSV export method (no authentication needed)
    - Private sheets: Uses Google Sheets API with service account credentials
    
    Args:
        sheets_url: Google Sheets URL (various formats supported)
        credentials_path: Path to service account JSON file (for private sheets)
        sheet_name: Name of specific sheet tab (optional, defaults to first sheet)
        timeout: Request timeout in seconds (default: 10)
        
    Returns:
        DataFrame with the Google Sheets data
        
    Examples:
        # Public sheet (no auth needed)
        df = parse_google_sheets('https://docs.google.com/spreadsheets/d/ABC123/edit')
        
        # Private sheet with service account
        df = parse_google_sheets('https://docs.google.com/spreadsheets/d/ABC123/edit', 
                               credentials_path='path/to/service-account.json')
    """
    
    # Extract sheet ID and gid from URL
    sheet_id, gid = _extract_sheet_info(sheets_url)
    
    if not sheet_id:
        raise ValueError("Could not extract sheet ID from URL. Please check the URL format.")
    
    # First try CSV export method (works for public sheets)
    try:
        df = _parse_sheets_csv_export(sheet_id, gid, timeout)
        if not df.empty:
            return df
    except Exception as e:
        print(f"CSV export method failed: {e}")
    
    # If CSV export fails, try Google Sheets API
    if credentials_path:
        try:
            df = _parse_sheets_api(sheet_id, credentials_path, sheet_name, timeout)
            return df
        except Exception as e:
            print(f"Google Sheets API method failed: {e}")
            raise
    else:
        raise ValueError(
            "Sheet appears to be private. Please provide credentials_path for service account authentication, "
            "or make the sheet public (Share > Anyone with the link > Viewer)."
        )


def _extract_sheet_info(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract sheet ID and gid from Google Sheets URL."""
    
    # Store original URL for gid extraction
    original_url = url
    
    # Remove fragments from URL for sheet ID extraction
    url = url.split('#')[0]
    
    # Pattern to match sheet ID
    sheet_id_pattern = r'/spreadsheets/d/([a-zA-Z0-9-_]+)'
    sheet_id_match = re.search(sheet_id_pattern, url)
    
    if not sheet_id_match:
        return None, None
    
    sheet_id = sheet_id_match.group(1)
    
    # Extract gid from original URL (including fragments)
    gid = None
    if '#gid=' in original_url:
        gid_match = re.search(r'#gid=(\d+)', original_url)
        if gid_match:
            gid = gid_match.group(1)
    elif 'gid=' in original_url:
        parsed_url = urlparse(original_url)
        query_params = parse_qs(parsed_url.query)
        if 'gid' in query_params:
            gid = query_params['gid'][0]
    
    return sheet_id, gid


def _parse_sheets_csv_export(sheet_id: str, gid: Optional[str], timeout: int) -> pd.DataFrame:
    """Parse Google Sheets using CSV export method (public sheets only)."""
    
    # Construct CSV export URL
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    if gid:
        csv_url += f"&gid={gid}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(csv_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    
    # Check if response is actually CSV data
    if response.headers.get('content-type', '').startswith('text/csv') or len(response.text) > 0:
        # Parse CSV data
        df = pd.read_csv(io.StringIO(response.text))
        return df
    else:
        raise ValueError("Failed to retrieve CSV data. Sheet may be private or not accessible.")


def _parse_sheets_api(sheet_id: str, 
                     credentials_path: str, 
                     sheet_name: Optional[str], 
                     timeout: int) -> pd.DataFrame:
    """Parse Google Sheets using Google Sheets API with service account credentials."""
    
    try:
        # Try to import google-auth and google-api-python-client
        from google.auth import exceptions as auth_exceptions
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except ImportError:
        raise ImportError(
            "Google API client libraries not found. Install with: "
            "pip install google-auth google-auth-oauthlib google-api-python-client"
        )
    
    # Load service account credentials
    try:
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
    except Exception as e:
        raise ValueError(f"Failed to load credentials from {credentials_path}: {e}")
    
    # Build the service
    service = build('sheets', 'v4', credentials=credentials)
    
    # Get sheet metadata to find available sheets
    try:
        sheet_metadata = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
        sheets = sheet_metadata.get('sheets', [])
        
        if not sheets:
            raise ValueError("No sheets found in the spreadsheet")
        
        # Determine which sheet to read
        if sheet_name:
            target_sheet = None
            for sheet in sheets:
                if sheet['properties']['title'] == sheet_name:
                    target_sheet = sheet
                    break
            if not target_sheet:
                available_sheets = [s['properties']['title'] for s in sheets]
                raise ValueError(f"Sheet '{sheet_name}' not found. Available sheets: {available_sheets}")
        else:
            # Use first sheet if no name specified
            target_sheet = sheets[0]
        
        sheet_title = target_sheet['properties']['title']
        
        # Read the sheet data
        range_name = f"'{sheet_title}'"
        result = service.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=range_name
        ).execute()
        
        values = result.get('values', [])
        
        if not values:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(values[1:], columns=values[0]) if len(values) > 1 else pd.DataFrame()
        
        return df
        
    except HttpError as e:
        if e.resp.status == 403:
            raise ValueError(
                "Access denied. Make sure the service account has access to the sheet, "
                "or the sheet is shared with the service account email."
            )
        else:
            raise ValueError(f"Google Sheets API error: {e}")
    except Exception as e:
        raise ValueError(f"Error accessing Google Sheets: {e}")


def example_usage():
    """Example usage of parse_message_content function."""
    sample_data = {
        'messages': [
            'Check out this article: https://example.com/article1 and also https://news.com/story',
            'Visit https://github.com/user/repo for the code',
            'No URLs in this message',
            'Multiple links: https://stackoverflow.com/questions/123 and https://docs.python.org/3/'
        ],
        'timestamps': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
    }
    
    df = pd.DataFrame(sample_data)
    
    result = parse_message_content(
        df=df,
        url_column='messages',
        date_column='timestamps',
        delay=0.5,
        timeout=5
    )
    
    print("\nSample result:")
    print(result.head())
    
    return result