import streamlit as st
import os
import json
import requests
import urllib.parse

from duckduckgo_search import DDGS
from openai import OpenAI
import anthropic

try:
    from google import genai
except ImportError:
    import google.generativeai as genai

# --- Backend Logic (adapted from consensus_agent.py) ---

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
        st.error(f"Error fetching news: {e}")
    return articles

def call_llm_for_probability(llm_name, prompt):
    try:
        if llm_name == "GPT-4o":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return 55.0 # simulated fallback
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
                return 52.0
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
                return 58.0
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(prompt)
            return float(response.text.strip().replace('%', ''))
    except Exception as e:
        st.error(f"Error calling {llm_name}: {e}")
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
    context = "\n".join([f"Title: {a['title']}\nSummary: {a['body']}" for a in articles])
    prompt = f"Event: {event_title}\nMarket Probability: {market_prob}%\nConsensus Probability: {consensus_prob:.2f}%\n\nNews Context:\n{context}\n\nWrite a detailed trade thesis. Should we buy or sell? Explain the reasoning based on the discrepancy between the market and consensus probabilities and the recent news."

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
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

# --- Streamlit UI ---

st.set_page_config(page_title="Consensus Agent", page_icon="📈", layout="wide")

st.title("📈 Consensus Agent")
st.markdown("Enter a Polymarket or Kalshi event URL to analyze the market probability against a multi-LLM consensus.")

url_input = st.text_input("Market URL (Polymarket / Kalshi):", placeholder="https://polymarket.com/event/...")

if st.button("Run Analysis", type="primary"):
    if not url_input:
        st.warning("Please enter a valid URL.")
    else:
        with st.status("Analyzing Market...", expanded=True) as status:
            st.write("Fetching market data...")
            event_title = None
            market_prob = None

            if "polymarket.com" in url_input:
                event_title, market_prob = parse_polymarket(url_input)
            elif "kalshi.com" in url_input:
                event_title, market_prob = parse_kalshi(url_input)
            else:
                st.error("Unsupported URL format.")
                status.update(label="Analysis failed.", state="error", expanded=False)
                st.stop()

            if not event_title or market_prob is None:
                st.error("Failed to fetch data from the provided URL.")
                status.update(label="Analysis failed.", state="error", expanded=False)
                st.stop()

            st.write(f"**Event:** {event_title}")
            st.write(f"**Market Probability:** {market_prob:.2f}%")

            st.write("Scraping recent news...")
            articles = get_news_articles(event_title)

            st.write("Calculating multi-LLM consensus probability...")
            consensus_prob, probabilities = calculate_consensus(event_title, articles)

            status.update(label="Analysis complete!", state="complete", expanded=False)

        # Display Results
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Market vs Consensus")
            st.metric("Market Price", f"{market_prob:.2f}%")
            if consensus_prob is not None:
                st.metric("Consensus Probability", f"{consensus_prob:.2f}%", delta=f"{consensus_prob - market_prob:.2f}%")

                st.markdown("#### LLM Breakdown")
                for llm, prob in probabilities.items():
                    st.write(f"**{llm}:** {prob:.2f}%")

        with col2:
            st.subheader("Recent News")
            if not articles:
                st.info("No news articles found.")
            else:
                for idx, article in enumerate(articles, 1):
                    with st.expander(f"[{article['source']}] {article['title']}"):
                        st.write(article['body'])
                        if article['url']:
                            st.markdown(f"[Read Full Article]({article['url']})")

        st.divider()

        if consensus_prob is not None:
            difference = abs(consensus_prob - market_prob)
            if difference > 10.0:
                st.subheader("🔥 Trade Thesis Generated")
                with st.spinner("Generating detailed trade thesis..."):
                    thesis = generate_trade_thesis(event_title, market_prob, consensus_prob, articles)
                st.info(thesis)
            else:
                st.subheader("Trade Thesis")
                st.markdown(f"Discrepancy is **{difference:.2f}%** (<= 10%). No strong trade edge detected. Thesis not generated.")
