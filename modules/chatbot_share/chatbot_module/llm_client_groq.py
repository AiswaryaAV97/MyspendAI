import os, json, requests
from dotenv import load_dotenv

# load project .env
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
env_path = os.path.join(root, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")  # e.g. "llama3-8b-8192"
if not GROQ_API_KEY or not GROQ_MODEL:
    raise RuntimeError("GROQ_API_KEY and GROQ_MODEL must be set in environment")

BASE_URL = "https://api.groq.com/openai/v1"  # Groq uses OpenAI-compatible endpoint

def generate_answer(context: str, question: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """
    Generate answer using Groq's OpenAI-compatible chat completions endpoint.
    """
    system_prompt = """You are a friendly and professional financial advisor assistant. 

Your role is to help users understand their financial information in a clear, conversational manner.

Important guidelines:
- Answer questions based ONLY on the provided information
- NEVER mention "context", "provided context", "based on context" or similar phrases - respond naturally as if you have direct knowledge
- KEEP RESPONSES SHORT - maximum 3-4 sentences or one short paragraph
- For advice questions (strategies, recommendations, tips), focus on actionable guidance - DO NOT calculate totals unless specifically asked
- For calculation questions (totals, savings, how much spent), provide accurate math with clear breakdowns
- When the information includes pre-calculated totals, use those instead of doing your own math
- Use natural, user-friendly language - avoid technical jargon like 'username', 'field names', or database terms
- Never mention internal IDs, email addresses, or technical field names in your response
- Use proper currency formatting (e.g., $15,000.00)
- For savings calculations: Total Income - Total Expenses = Savings
- If asked for totals or summaries, provide the final answer prominently at the start
- If the information doesn't contain what's needed, politely say so in one sentence
- Be CONCISE and DIRECT - get straight to the point without lengthy explanations
- For strategy/advice questions, give 2-3 practical recommendations maximum

SPECIAL INSTRUCTIONS FOR RATE QUERIES:
- When presenting financial rates with markdown links (e.g., [Ratehub Mortgage Rates](https://...)), PRESERVE the exact markdown formatting in your response
- Include ALL rates, provider information, and source links from the information provided
- Present the information in a clear, organized format using the same structure as provided
- Do NOT summarize or paraphrase the rate information - copy it exactly as formatted"""

    user_prompt = f"INFORMATION:\n{context}\n\nQUESTION: {question}\n\nProvide a clear, helpful answer. Remember to be user-friendly and avoid mentioning technical details."
    
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    print(f"🤖 Calling Groq API: {url}")
    print(f"🤖 Model: {GROQ_MODEL}")
    
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    
    # Print response for debugging
    print(f"🤖 Response status: {resp.status_code}")
    if resp.status_code != 200:
        print(f"🤖 Response body: {resp.text}")
    
    resp.raise_for_status()
    j = resp.json()
    
    print(f"🤖 Response JSON keys: {list(j.keys())}")

    # Parse OpenAI-compatible response format
    if isinstance(j, dict) and "choices" in j and isinstance(j["choices"], list) and j["choices"]:
        choice = j["choices"][0]
        if "message" in choice and isinstance(choice["message"], dict):
            answer = choice["message"].get("content", "")
            print(f"🤖 Generated answer: {answer[:200]}...")
            return answer
        elif "text" in choice:
            print(f"🤖 Generated text: {choice['text'][:200]}...")
            return choice["text"]
    
    # Fallback
    print(f"🤖 Unexpected response format, returning JSON dump")
    return json.dumps(j)