import re
import json
from typing import Any, Dict
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
from typing import Tuple

def parse_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a Discord style message dictionary.

    Extracts the message id, timestamp, content with any urls stripped, a
    list of urls and simplified embed metadata.
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
    """Add a ``combined_content`` column containing ``embeds`` and ``content``."""
    df = df.copy()
    df["combined_content"] = df["embeds"].astype(str) + " " + df["content"].astype(str)
    return df



def parse_message_content(df: pd.DataFrame, 
                         url_column: str, 
                         date_column: str,
                         delay: float = 1.0,
                         timeout: int = 10,
                         max_retries: int = 3) -> pd.DataFrame:
    """
    Extract URLs from a specified column in a dataframe and retrieve their metadata.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing URLs and dates
    url_column : str
        Name of the column containing URLs
    date_column : str
        Name of the column containing dates
    delay : float, default=1.0
        Delay between requests in seconds (to be respectful to servers)
    timeout : int, default=10
        Request timeout in seconds
    max_retries : int, default=3
        Maximum number of retry attempts for failed requests
    
    Returns:
    --------
    pd.DataFrame
        New dataframe with columns: ['url', 'date', 'title', 'description', 'domain', 'status']
    """
    
    def extract_urls_from_text(text: str) -> list:
        """Extract all URLs from a text string."""
        if pd.isna(text):
            return []
        
        # Comprehensive URL regex pattern
        url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
        urls = re.findall(url_pattern, str(text))
        return urls
    
    def get_page_metadata(url: str, timeout: int = 10, max_retries: int = 3) -> Tuple[str, str, str]:
        """
        Retrieve title and description from a URL.
        
        Returns:
        --------
        tuple: (title, description, status)
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title = ""
                if soup.title:
                    title = soup.title.string.strip() if soup.title.string else ""
                
                # Extract description (try multiple meta tags)
                description = ""
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if not meta_desc:
                    meta_desc = soup.find('meta', attrs={'property': 'og:description'})
                if not meta_desc:
                    meta_desc = soup.find('meta', attrs={'name': 'twitter:description'})
                
                if meta_desc:
                    description = meta_desc.get('content', '').strip()
                
                return title, description, "success"
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return "", "", f"error: {str(e)}"
                time.sleep(delay * (attempt + 1))  # Exponential backoff
        
        return "", "", "error: max retries exceeded"
    
    def get_domain(url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return ""
    
    # Validate input columns
    if url_column not in df.columns:
        raise ValueError(f"Column '{url_column}' not found in dataframe")
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in dataframe")
    
    # Extract URLs and create new dataframe
    results = []
    
    print(f"Processing {len(df)} rows...")
    
    for idx, row in df.iterrows():
        urls = extract_urls_from_text(row[url_column])
        date_value = row[date_column]
        
        if not urls:
            continue
            
        for url in urls:
            # Clean URL (remove common trailing characters)
            url = url.rstrip('.,;!?)')
            
            # Get domain
            domain = get_domain(url)
            
            # Get metadata
            print(f"Fetching metadata for: {url}")
            title, description, status = get_page_metadata(url, timeout, max_retries)
            
            results.append({
                'url': url,
                'date': date_value,
                'title': title,
                'description': description,
                'domain': domain,
                'status': status,
                'original_row_index': idx
            })
            
            # Be respectful to servers
            time.sleep(delay)
    
    # Create result dataframe
    result_df = pd.DataFrame(results)
    
    # Convert date column to datetime if possible
    if not result_df.empty:
        try:
            result_df['date'] = pd.to_datetime(result_df['date'])
        except:
            pass  # Keep original format if conversion fails
    
    print(f"Extraction complete! Found {len(result_df)} URLs from {len(df)} rows.")
    print(f"Success rate: {len(result_df[result_df['status'] == 'success'])} / {len(result_df)} URLs")
    
    return result_df

# Example usage function
def example_usage_parse_message_content():
    """
    Example of how to use the parse_message_content function
    """
    # Create sample data
    sample_data = {
        'messages': [
            'Check out this article: https://example.com/article1 and also https://news.com/story',
            'Visit https://github.com/user/repo for the code',
            'No URLs in this message',
            'Multiple links: https://stackoverflow.com/questions/123 and https://docs.python.org/3/'
        ],
        'timestamps': [
            '2024-01-01',
            '2024-01-02', 
            '2024-01-03',
            '2024-01-04'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Parse the messages
    result = parse_message_content(
        df=df,
        url_column='messages',
        date_column='timestamps',
        delay=0.5,  # Shorter delay for example
        timeout=5
    )
    
    print("\nSample result:")
    print(result.head())
    
    return result

# Helper function to analyze results
def analyze_url_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the results from parse_message_content function
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
    
    # Top domains
    top_domains = df['domain'].value_counts().head(10)
    
    # URLs by date
    if 'date' in df.columns:
        urls_by_date = df.groupby(df['date'].dt.date).size() if df['date'].dtype == 'datetime64[ns]' else df.groupby('date').size()
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

