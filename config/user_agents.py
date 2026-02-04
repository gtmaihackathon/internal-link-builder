"""
User Agent Strings for Web Crawling
Use Googlebot to bypass certain website blocks
"""

USER_AGENTS = {
    # Google crawlers
    'googlebot': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'googlebot_mobile': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 5X Build/MMB29P) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/W.X.Y.Z Mobile Safari/537.36 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'googlebot_image': 'Googlebot-Image/1.0',
    'googlebot_news': 'Googlebot-News',
    
    # Other search engines
    'bingbot': 'Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)',
    'yandexbot': 'Mozilla/5.0 (compatible; YandexBot/3.0; +http://yandex.com/bots)',
    'duckduckbot': 'DuckDuckBot/1.0; (+http://duckduckgo.com/duckduckbot.html)',
    
    # Browsers
    'chrome': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'chrome_mac': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'firefox': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'safari': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'edge': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
    
    # Mobile browsers
    'chrome_mobile': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
    'safari_mobile': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1',
}

# Default headers for crawling
DEFAULT_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}


def get_user_agent(name: str) -> str:
    """
    Get user agent string by name.
    
    Args:
        name: User agent name (e.g., 'googlebot', 'chrome')
        
    Returns:
        User agent string, defaults to googlebot if name not found
    """
    return USER_AGENTS.get(name.lower(), USER_AGENTS['googlebot'])


def get_headers(user_agent_name: str = 'googlebot') -> dict:
    """
    Get complete headers dict for requests.
    
    Args:
        user_agent_name: Name of user agent to use
        
    Returns:
        Headers dictionary
    """
    headers = DEFAULT_HEADERS.copy()
    headers['User-Agent'] = get_user_agent(user_agent_name)
    return headers


def list_user_agents() -> list:
    """Return list of available user agent names"""
    return list(USER_AGENTS.keys())
