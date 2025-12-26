"""
Display Full Marketaux API JSON Response

This script fetches news from Marketaux API and displays the complete JSON response.
"""

import requests
import json
from datetime import datetime, timedelta
import pytz

def show_full_json_response():
    """Fetch and display the complete JSON response from Marketaux API."""
    
    # API key
    api_key = "6hV3UI51G9IwfgEZ1Yg7oQBQ6QP5zwCcMF5k3lnR"
    
    # Test parameters
    ticker = "AAPL"
    # Use a recent date
    test_date = (datetime.now(pytz.UTC) - timedelta(days=1)).strftime('%Y-%m-%d')
    
    base_url = "https://api.marketaux.com/v1/news/all"
    
    print("=" * 80)
    print("Full Marketaux API JSON Response")
    print("=" * 80)
    print(f"\nTicker: {ticker}")
    print(f"Date: {test_date}")
    print(f"API Endpoint: {base_url}")
    print("\n" + "-" * 80)
    
    # Request parameters
    params = {
        'api_token': api_key,
        'symbols': ticker,
        'published_on': test_date,
        'limit': 10,
        'page': 1
    }
    
    print(f"\nRequest Parameters:")
    for key, value in params.items():
        if key == 'api_token':
            print(f"  {key}: {'*' * 20}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("Making API Request...")
    print("=" * 80)
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        
        print(f"\nResponse Status Code: {response.status_code}")
        print(f"Response Headers:")
        for key, value in response.headers.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 80)
        print("COMPLETE JSON RESPONSE")
        print("=" * 80)
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                # Display the full JSON with proper formatting
                print("\n" + json.dumps(data, indent=2, ensure_ascii=False))
                
                print("\n" + "=" * 80)
                print("JSON Structure Summary")
                print("=" * 80)
                
                if 'meta' in data:
                    print(f"\nMeta Information:")
                    meta = data['meta']
                    for key, value in meta.items():
                        print(f"  {key}: {value}")
                
                if 'data' in data:
                    articles = data['data']
                    print(f"\nArticles Array:")
                    print(f"  Number of articles: {len(articles)}")
                    
                    if len(articles) > 0:
                        print(f"\n  First Article Structure:")
                        first = articles[0]
                        print(f"    Keys: {list(first.keys())}")
                        print(f"    Entities count: {len(first.get('entities', []))}")
                        
                        if 'entities' in first and len(first['entities']) > 0:
                            print(f"\n    First Entity Structure:")
                            first_entity = first['entities'][0]
                            print(f"      Keys: {list(first_entity.keys())}")
                
                # Also save to file for easier viewing
                output_file = f"marketaux_response_{ticker}_{test_date}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"\n" + "=" * 80)
                print(f"JSON also saved to: {output_file}")
                print("=" * 80)
                
            except json.JSONDecodeError as e:
                print(f"\n❌ Failed to parse JSON: {e}")
                print(f"\nRaw Response Text:")
                print(response.text)
        else:
            print(f"\n❌ Non-200 status code: {response.status_code}")
            print(f"\nResponse Text:")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")


if __name__ == "__main__":
    show_full_json_response()

