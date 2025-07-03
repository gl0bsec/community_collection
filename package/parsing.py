import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag, NavigableString
from tqdm import tqdm

from urlextract import URLExtract

warnings.filterwarnings('ignore')


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
                         delay: float = 1.0,
                         timeout: int = 10,
                         max_retries: int = 3) -> pd.DataFrame:
    """Extract URLs from DataFrame and fetch their metadata.
    
    Args:
        df: DataFrame with URL and date columns
        url_column: Column name containing URLs
        date_column: Column name containing dates
        delay: Seconds between requests (default: 1.0)
        timeout: Request timeout seconds (default: 10)
        max_retries: Max retry attempts (default: 3)
        
    Returns:
        DataFrame with columns: url, date, title, description, domain, status
    """
    
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
    
    def get_page_metadata(url: str, timeout: int = 10, max_retries: int = 3) -> Tuple[str, str, str]:
        """Fetch page title and description from URL.
        
        Returns:
            Tuple of (title, description, status)
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
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
                
                if meta_desc and hasattr(meta_desc, 'get'):
                    content = meta_desc.get('content', '')
                    description = str(content).strip() if content else ''
                
                return title, description, "success"
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return "", "", f"Request failed: {type(e).__name__}"
                time.sleep(delay * (attempt + 1))
        
        return "", "", "error: max retries exceeded"
    
    def get_domain(url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return ""
    
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
    
    print(f"Found {len(all_urls)} URLs from {len(df)} rows. Fetching metadata...")
    
    # Second pass: fetch metadata with progress bar
    results = []
    failed_urls = []
    
    for idx, url, date_value in tqdm(all_urls, desc="Fetching metadata"):
        domain = get_domain(url)
        title, description, status = get_page_metadata(url, timeout, max_retries)
        
        if status != "success":
            failed_urls.append((url, status))
        
        results.append({
            'url': url,
            'date': date_value,
            'title': title,
            'description': description,
            'domain': domain,
            'status': status,
            'original_row_index': idx
        })
        
        time.sleep(delay)
    
    result_df = pd.DataFrame(results)
    
    if not result_df.empty:
        try:
            result_df['date'] = pd.to_datetime(result_df['date'])
        except Exception:
            pass
    
    # Print warnings for failed URLs
    if failed_urls:
        print(f"\nWarning: {len(failed_urls)} URLs failed to fetch metadata:")
        for url, error in failed_urls[:5]:  # Show first 5 failures
            print(f"  - {url}: {error}")
        if len(failed_urls) > 5:
            print(f"  ... and {len(failed_urls) - 5} more")
    
    success_count = len(result_df[result_df['status'] == 'success']) if not result_df.empty else 0
    print(f"\nExtraction complete! Success rate: {success_count}/{len(result_df)} URLs ({success_count/len(result_df)*100:.1f}%)" if not result_df.empty else "No URLs processed.")
    
    return result_df


def extract_url_metadata(url: str, timeout: int = 10) -> Dict[str, Optional[str]]:
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
        DataFrame with columns: cleaned_text, date, original_text_length, 
        cleaned_text_length, urls_removed_count, original_row_index
    """
    
    def remove_urls_from_text(text: str) -> Tuple[str, int]:
        """Remove all URLs from text and return cleaned text with count."""
        if pd.isna(text):
            return "", 0
        
        text_str = str(text)
        url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
        
        urls_found = re.findall(url_pattern, text_str)
        urls_count = len(urls_found)
        cleaned_text = re.sub(url_pattern, ' ', text_str)
        
        return cleaned_text, urls_count
    
    def clean_text(text: str) -> str:
        """Clean up text by removing extra whitespace and normalizing."""
        if not text:
            return ""
        
        # Remove extra whitespace
        cleaned = ' '.join(text.split())
        
        # Remove multiple punctuation marks
        cleaned = re.sub(r'[.]{2,}', '.', cleaned)
        cleaned = re.sub(r'[!]{2,}', '!', cleaned)
        cleaned = re.sub(r'[?]{2,}', '?', cleaned)
        
        # Clean up common artifacts from URL removal
        cleaned = re.sub(r'\s+[.,;!?]\s+', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    # Validate input columns
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataframe")
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in dataframe")
    
    results = []
    print(f"Processing {len(df)} rows for text extraction...")
    
    for idx, row in df.iterrows():
        original_text = row[text_column]
        date_value = row[date_column]
        
        if pd.isna(original_text) or not str(original_text).strip():
            continue
        
        original_text_str = str(original_text)
        original_length = len(original_text_str)
        cleaned_text, urls_removed = remove_urls_from_text(original_text_str)
        
        if clean_whitespace:
            cleaned_text = clean_text(cleaned_text)
        
        cleaned_length = len(cleaned_text)
        
        if cleaned_length >= min_text_length:
            results.append({
                'cleaned_text': cleaned_text,
                'date': date_value,
                'original_text_length': original_length,
                'cleaned_text_length': cleaned_length,
                'urls_removed_count': urls_removed,
                'original_row_index': idx
            })
    
    result_df = pd.DataFrame(results)
    
    if not result_df.empty:
        try:
            result_df['date'] = pd.to_datetime(result_df['date'])
        except Exception:
            pass
    
    print(f"Text extraction complete! Processed {len(result_df)} rows with sufficient text content.")
    if not result_df.empty:
        total_urls_removed = result_df['urls_removed_count'].sum()
        avg_text_reduction = ((result_df['original_text_length'] - result_df['cleaned_text_length']) / result_df['original_text_length'] * 100).mean()
        print(f"Total URLs removed: {total_urls_removed}")
        print(f"Average text length reduction: {avg_text_reduction:.1f}%")
    
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
        'total_urls_removed': df['urls_removed_count'].sum(),
        'avg_original_length': df['original_text_length'].mean(),
        'avg_cleaned_length': df['cleaned_text_length'].mean(),
        'avg_length_reduction': ((df['original_text_length'] - df['cleaned_text_length']) / df['original_text_length'] * 100).mean(),
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