try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS
import argparse
import urllib.parse
import requests
import json
import sys

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='duckduckgo_search')
warnings.filterwarnings('ignore', category=FutureWarning)


def parse_polymarket(url):
    parsed = urllib.parse.urlparse(url)
    slug = parsed.path.strip('/').split('/')[-1]

    api_url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        if data and len(data) > 0:
            event = data[0]
            title = event.get('title')
            markets = event.get('markets', [])

            # Case 1: Multiple choice (e.g., Presidential Election Winner)
            if len(markets) > 1:
                top_prob = 0
                top_outcome = None
                for market in markets:
                    try:
                        outcomes = json.loads(market['outcomes']) if isinstance(market['outcomes'], str) else market['outcomes']
                        prices = json.loads(market['outcomePrices']) if isinstance(market['outcomePrices'], str) else market['outcomePrices']

                        if 'Yes' in outcomes:
                            yes_idx = outcomes.index('Yes')
                            price = float(prices[yes_idx]) * 100
                            if price > top_prob:
                                top_prob = price
                                top_outcome = market.get('groupItemTitle') or title
                    except Exception:
                        continue
                if top_outcome:
                    return f"{title} - {top_outcome}", top_prob

            # Case 2: Single binary market
            for market in markets:
                try:
                    outcomes = json.loads(market['outcomes']) if isinstance(market['outcomes'], str) else market['outcomes']
                    prices = json.loads(market['outcomePrices']) if isinstance(market['outcomePrices'], str) else market['outcomePrices']

                    if 'Yes' in outcomes:
                        yes_idx = outcomes.index('Yes')
                        price = float(prices[yes_idx]) * 100
                        return title, price
                except Exception:
                    continue
    return None, None

def parse_kalshi(url):
    parsed = urllib.parse.urlparse(url)
    path_parts = parsed.path.strip('/').split('/')
    if 'markets' in path_parts:
        ticker = path_parts[-1]
        api_url = f"https://api.elections.kalshi.com/trade-api/v2/events/{ticker}"
        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()
            if 'event' in data:
                title = data['event'].get('title')
                markets = data.get('markets', [])
                top_prob = 0
                top_outcome = None

                for market in markets:
                    price = market.get('yes_ask', 0)
                    if price == 0:
                        price = market.get('last_price', 0)
                    if price > top_prob:
                        top_prob = price
                        top_outcome = market.get('subtitle', market.get('title'))

                if len(markets) == 1:
                    return title, top_prob
                elif len(markets) > 1 and top_outcome:
                    return f"{title} - {top_outcome}", top_prob
    return None, None



def get_news_articles(query):
    print(f"\nSearching for news articles related to: '{query}'")
    articles = []
    try:
        with DDGS() as ddgs:
            results = ddgs.news(query, max_results=5)
            for r in results:
                articles.append({
                    "title": r.get('title'),
                    "body": r.get('body'),
                    "url": r.get('url'),
                    "source": r.get('source'),
                    "date": r.get('date')
                })
    except Exception as e:
        print(f"Error fetching news: {e}")
    return articles


import os
from openai import OpenAI
import anthropic

try:
    from google import genai
except ImportError:
    import google.generativeai as genai


def call_llm_for_probability(llm_name, prompt):
    print(f"Calling {llm_name}...")
    try:
        if llm_name == "GPT-4o":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Missing OPENAI_API_KEY. Simulating GPT-4o output.")
                return 55.0 # default simulation
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return float(response.choices[0].message.content.strip().replace('%', ''))

        elif llm_name == "Claude 3.5":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("Missing ANTHROPIC_API_KEY. Simulating Claude 3.5 output.")
                return 52.0 # default simulation
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            return float(response.content[0].text.strip().replace('%', ''))

        elif llm_name == "Gemini 1.5 Pro":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("Missing GEMINI_API_KEY. Simulating Gemini 1.5 Pro output.")
                return 58.0 # default simulation
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            return float(response.text.strip().replace('%', ''))
    except Exception as e:
        print(f"Error calling {llm_name}: {e}")
        return None

def calculate_consensus(event_title, articles):
    with open("model_performance.json", "r") as f:
        weights = json.load(f)

    context = "\n".join([f"Title: {a['title']}\nSummary: {a['body']}" for a in articles])
    prompt = f"Given the following news context:\n{context}\n\nEstimate the winning probability (0-100) for the event '{event_title}'. Respond ONLY with a number, no other text."

    probabilities = {}
    total_weight = 0
    weighted_sum = 0

    for llm_name, weight in weights.items():
        prob = call_llm_for_probability(llm_name, prompt)
        if prob is not None:
            probabilities[llm_name] = prob
            weighted_sum += prob * weight
            total_weight += weight

    if total_weight > 0:
        consensus_prob = weighted_sum / total_weight
        return consensus_prob, probabilities
    return None, None


def generate_trade_thesis(event_title, market_prob, consensus_prob, articles):
    print("\nGenerating Trade Thesis...")
    context = "\n".join([f"Title: {a['title']}\nSummary: {a['body']}" for a in articles])
    prompt = f"Event: {event_title}\nMarket Probability: {market_prob}%\nConsensus Probability: {consensus_prob:.2f}%\n\nNews Context:\n{context}\n\nWrite a detailed trade thesis. Should we buy or sell? Explain the reasoning based on the discrepancy between the market and consensus probabilities and the recent news."

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY. Simulating Trade Thesis.")
        return "Simulated Trade Thesis:\nBased on the discrepancy, there appears to be a significant mispricing in the market. The consensus probability is markedly different from the market price. The recent news articles suggest varying perspectives that the market may not have fully priced in. Recommendation: Act according to the consensus edge."

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating thesis: {e}"

def main():
    parser = argparse.ArgumentParser(description="Consensus Agent for Market Probabilities")
    parser.add_argument("url", help="Polymarket or Kalshi market URL")
    args = parser.parse_args()

    url = args.url
    print(f"Analyzing Market URL: {url}")

    event_title = None
    market_prob = None

    if "polymarket.com" in url:
        event_title, market_prob = parse_polymarket(url)
    elif "kalshi.com" in url:
        event_title, market_prob = parse_kalshi(url)
    else:
        print("Unsupported URL. Please provide a Polymarket or Kalshi URL.")
        sys.exit(1)

    if not event_title or market_prob is None:
        print("Failed to fetch market data from the provided URL.")
        sys.exit(1)

    print(f"Event: {event_title}")
    print(f"Market Probability: {market_prob:.2f}%")

    # Get news articles
    articles = get_news_articles(event_title)
    for idx, article in enumerate(articles, 1):
        print(f"  {idx}. [{article['source']}] {article['title']}")
        print(f"     {article['body'][:100]}...")

    consensus_prob, probabilities = calculate_consensus(event_title, articles)
    if consensus_prob is not None:
        print("\n=== Probabilities ===")
        for llm, prob in probabilities.items():
            print(f"{llm}: {prob:.2f}%")
        print(f"\nConsensus Probability: {consensus_prob:.2f}%")
        print(f"Market Probability:    {market_prob:.2f}%")

        difference = abs(consensus_prob - market_prob)
        if difference > 10.0:
            print(f"\nDiscrepancy is {difference:.2f}% (> 10%). Initiating Trade Thesis generation.")
            thesis = generate_trade_thesis(event_title, market_prob, consensus_prob, articles)
            print("\n==================================")
            print("         TRADE THESIS             ")
            print("==================================")
            print(thesis)
        else:
            print(f"\nDiscrepancy is {difference:.2f}% (<= 10%). No trade thesis generated.")

if __name__ == "__main__":
    main()
