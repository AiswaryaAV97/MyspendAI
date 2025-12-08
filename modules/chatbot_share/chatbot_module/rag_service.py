from modules.chatbot_share.chatbot_module.embedder import Embedder
from modules.chatbot_share.chatbot_module.vector_client_pinecone import upsert_vectors, query_vector
from modules.chatbot_share.chatbot_module.llm_client_groq import generate_answer
from modules.db import db
from bson import ObjectId
import datetime

rag_coll = db["rag_docs"]

_embedder = None

def get_embedder():
    """Get or create embedder instance (lazy initialization)"""
    global _embedder
    if _embedder is None:
        print("📥 Loading embedding model (first time)...")
        _embedder = Embedder()
        print("✅ Embedding model loaded!")
    return _embedder

def make_json_safe(obj, _seen=None):
    """
    Recursively convert non-JSON-safe objects (ObjectId, datetime, etc.) to JSON-safe types.
    Handles Pinecone Match objects and other edge cases.
    Prevents infinite loops with circular references.
    """
    # Track seen objects to prevent infinite recursion
    if _seen is None:
        _seen = set()
    
    # Check for circular reference
    obj_id = id(obj)
    if obj_id in _seen:
        return str(obj)
    
    if obj is None:
        return None
    elif isinstance(obj, dict):
        _seen.add(obj_id)
        result = {k: make_json_safe(v, _seen) for k, v in obj.items()}
        _seen.remove(obj_id)
        return result
    elif isinstance(obj, (list, tuple)):
        _seen.add(obj_id)
        result = [make_json_safe(v, _seen) for v in obj]
        _seen.remove(obj_id)
        return result
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # For any other non-serializable object (like Pinecone Match), just stringify it
        # Don't try to recurse into __dict__ as it can cause infinite loops
        return str(obj)

def ingest_chunks_for_user(user_id, chunks):
    """chunks: list of {'text','source','chunk_index', 'snippet_meta'(opt)}"""
    if not chunks:
        return []
    embedder = get_embedder()  # Get embedder instance
    texts = [c["text"] for c in chunks]
    embs = embedder.embed(texts)
    items = []
    inserted_ids = []
    for c, emb in zip(chunks, embs):
        doc = {
            "user_id": str(user_id),
            "text": c["text"],
            "source": c.get("source"),
            "chunk_index": c.get("chunk_index"),
            "metadata": c.get("snippet_meta", {}),
            "created_at": datetime.datetime.utcnow(),
            "_ingested": True
        }
        res = rag_coll.insert_one(doc)
        inserted_ids.append(str(res.inserted_id))
        items.append({"id": str(res.inserted_id), "vector": emb, "metadata": {"user_id": str(user_id), "source": doc["source"]}})
    if items:
        upsert_vectors(items)
    return inserted_ids

def query_user(user_id, query, top_k=5):
    embedder = get_embedder()  # Get embedder instance
    q_emb = embedder.embed([query])[0]
    matches = query_vector(q_emb, top_k=top_k, filter={"user_id":{"$eq": str(user_id)}})
    ids = [m.get("id") or m.get("match", {}).get("id") for m in matches]
    # filter valid object ids and fetch from Mongo
    object_ids = [ObjectId(i) for i in ids if i and ObjectId.is_valid(i)]
    docs = list(rag_coll.find({"_id": {"$in": object_ids}}))
    id_to_doc = {str(d["_id"]): d for d in docs}
    ordered = [id_to_doc.get(i) for i in ids if id_to_doc.get(i)]
    context = "\n\n".join([f"{idx+1}. {d.get('text')[:1000]}" for idx,d in enumerate(ordered) if d])
    return {"query": query, "results": ordered, "context": context, "raw_matches": matches}

def get_all_bank_transactions(user_id):
    """
    Retrieve ALL bank statement transactions directly from MongoDB for financial calculations.
    This bypasses vector search to ensure complete data.
    """
    expenses_coll = db["expenses"]
    transactions = list(expenses_coll.find({
        "username": user_id,
        "source": "bank_statement"
    }).sort("date", 1))  # Sort by date ascending
    
    if not transactions:
        return ""
    
    # Build comprehensive context with all transactions
    context_parts = []
    context_parts.append("=== COMPLETE BANK STATEMENT DATA ===")
    context_parts.append(f"Total transactions: {len(transactions)}")
    
    # Group by type
    income_txns = [t for t in transactions if t.get('type') == 'income']
    expense_txns = [t for t in transactions if t.get('type') == 'expense']
    
    context_parts.append(f"\n--- INCOME TRANSACTIONS ({len(income_txns)}) ---")
    for idx, txn in enumerate(income_txns, 1):
        context_parts.append(
            f"{idx}. Date: {txn.get('date')}, "
            f"Description: {txn.get('description')}, "
            f"Amount: ${txn.get('amount', 0):.2f}, "
            f"Category: {txn.get('category')}"
        )
    
    context_parts.append(f"\n--- EXPENSE TRANSACTIONS ({len(expense_txns)}) ---")
    for idx, txn in enumerate(expense_txns, 1):
        context_parts.append(
            f"{idx}. Date: {txn.get('date')}, "
            f"Description: {txn.get('description')}, "
            f"Amount: ${txn.get('amount', 0):.2f}, "
            f"Category: {txn.get('category')}"
        )
    
    # Add totals
    total_income = sum(t.get('amount', 0) for t in income_txns)
    total_expenses = sum(t.get('amount', 0) for t in expense_txns)
    
    context_parts.append(f"\n--- TOTALS ---")
    context_parts.append(f"Total Income: ${total_income:.2f}")
    context_parts.append(f"Total Expenses: ${total_expenses:.2f}")
    context_parts.append(f"Net Savings: ${total_income - total_expenses:.2f}")
    
    return "\n".join(context_parts)

def is_financial_calculation_query(question):
    """
    Detect if the question is asking for financial calculations that require ALL transactions.
    Only triggers for explicit calculation/total requests, not general advice questions.
    """
    question_lower = question.lower()
    
    # Strong calculation indicators - must have one of these
    strong_calc_keywords = [
        'how much did i save', 'how much did i spend', 'how much have i save', 'how much have i spend',
        'total income', 'total expense', 'total spending', 'total savings',
        'calculate my', 'sum of', 'add up',
        'what did i spend', 'what have i spent', 'what did i save',
        'all my expenses', 'all my spending', 'all my transactions',
        'savings in', 'spent in', 'expenses in'  # e.g., "savings in October"
    ]
    
    # Check if question contains strong calculation indicators
    if any(keyword in question_lower for keyword in strong_calc_keywords):
        return True
    
    # Also trigger for single-word calculation queries
    simple_calc_words = ['balance', 'breakdown']
    # Only if the question is short and focused on calculations
    if any(word in question_lower for word in simple_calc_words):
        # Additional check: question should be short and not asking for advice
        if len(question.split()) < 10 and 'should' not in question_lower and 'advice' not in question_lower:
            return True
    
    return False

def is_audit_query(question):
    """
    Detect if the question is asking for an audit or comparison between contract and actual spending.
    """
    keywords = ['audit', 'contract', 'agreement', 'policy', 'overcharged', 'charged too much', 'correct amount', 'supposed to pay']
    return any(k in question.lower() for k in keywords)

def perform_audit(user_id, question):
    """
    Perform an audit by comparing contract terms (from RAG) with actual spending (from MongoDB).
    """
    print(f"🕵️‍♀️ perform_audit: Starting audit for question: {question}")
    
    # 1. Retrieve contract terms using RAG
    # We specifically look for contracts/agreements
    audit_query = f"{question} contract agreement terms price cost"
    res = query_user(user_id, audit_query, top_k=5)
    contract_context = res.get("context", "")
    
    if not contract_context:
        return {
            "question": question,
            "answer": "I couldn't find any uploaded contracts or agreements to audit. Please upload a contract PDF first.",
            "context": "No contracts found.",
            "results": []
        }

    # 2. Use LLM to extract Merchant and Expected Amount from the contract context
    extraction_prompt = f"""
    Based on the following contract text, identify the Service Provider (Merchant) and the Expected Payment Amount/Frequency.
    
    Contract Text:
    {contract_context}
    
    Return ONLY a JSON object with keys: "merchant_name", "expected_amount" (number), "frequency" (e.g. monthly), "currency".
    If not found, return null for values.
    """
    
    # We use a separate LLM call for extraction to be precise
    extraction_response = generate_answer(contract_context, "Extract merchant and amount as JSON", max_tokens=100)
    print(f"🕵️‍♀️ Extraction response: {extraction_response}")
    
    merchant_name = ""
    try:
        # Simple heuristic to find merchant name if JSON parsing fails or is simple text
        # In a real app, we'd use structured output or better parsing
        import re
        # Try to find JSON-like structure
        json_match = re.search(r'\{.*\}', extraction_response, re.DOTALL)
        if json_match:
            import json
            data = json.loads(json_match.group(0))
            merchant_name = data.get("merchant_name")
    except:
        pass
        
    # If extraction failed, we'll just proceed with a general comparison using the context
    
    # 3. Retrieve actual transactions from MongoDB
    # We'll fetch all expenses and let the LLM filter, or we could try to filter by merchant_name if we extracted it
    expenses_coll = db["expenses"]
    # Get last 6 months of transactions to be safe
    start_date = (datetime.datetime.utcnow() - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
    
    query = {
        "username": user_id,
        "date": {"$gte": start_date},
        "type": "expense"
    }
    
    # If we have a merchant name, we could try to filter, but regex search in Mongo is better done by LLM on a smaller set
    # or we just fetch all and let LLM find the matches. Fetching all recent expenses is safer.
    transactions = list(expenses_coll.find(query).sort("date", -1).limit(100))
    
    transaction_text = "Recent Transactions:\n"
    for t in transactions:
        transaction_text += f"- {t.get('date')}: {t.get('description')} (${t.get('amount')})\n"
        
    # 4. Generate Audit Report
    final_context = f"""
    CONTRACT TERMS:
    {contract_context}
    
    ACTUAL SPENDING (Last 6 months):
    {transaction_text}
    """
    
    audit_prompt = f"""
    You are a Financial Auditor. Compare the Contract Terms with the Actual Spending.
    
    Question: {question}
    
    Task:
    1. Identify the specific cost mentioned in the contract.
    2. Identify the actual payments made for this service in the transaction list.
    3. Determine if there is a discrepancy (overcharge, undercharge, or correct).
    4. Highlight the difference.
    
    Response Format:
    - **Status**: [Overcharged / Correct / Undercharged / Unclear]
    - **Contract Rate**: [Amount from contract]
    - **Actual Charged**: [Amount from transactions]
    """
    expenses_coll = db["expenses"]
    transactions = list(expenses_coll.find({
        "username": user_id,
        "source": "bank_statement"
    }).sort("date", 1))  # Sort by date ascending
    
    if not transactions:
        return ""
    
    # Build comprehensive context with all transactions
    context_parts = []
    context_parts.append("=== COMPLETE BANK STATEMENT DATA ===")
    context_parts.append(f"Total transactions: {len(transactions)}")
    
    # Group by type
    income_txns = [t for t in transactions if t.get('type') == 'income']
    expense_txns = [t for t in transactions if t.get('type') == 'expense']
    
    context_parts.append(f"\n--- INCOME TRANSACTIONS ({len(income_txns)}) ---")
    for idx, txn in enumerate(income_txns, 1):
        context_parts.append(
            f"{idx}. Date: {txn.get('date')}, "
            f"Description: {txn.get('description')}, "
            f"Amount: ${txn.get('amount', 0):.2f}, "
            f"Category: {txn.get('category')}"
        )
    
    context_parts.append(f"\n--- EXPENSE TRANSACTIONS ({len(expense_txns)}) ---")
    for idx, txn in enumerate(expense_txns, 1):
        context_parts.append(
            f"{idx}. Date: {txn.get('date')}, "
            f"Description: {txn.get('description')}, "
            f"Amount: ${txn.get('amount', 0):.2f}, "
            f"Category: {txn.get('category')}"
        )
    
    # Add totals
    total_income = sum(t.get('amount', 0) for t in income_txns)
    total_expenses = sum(t.get('amount', 0) for t in expense_txns)
    
    context_parts.append(f"\n--- TOTALS ---")
    context_parts.append(f"Total Income: ${total_income:.2f}")
    context_parts.append(f"Total Expenses: ${total_expenses:.2f}")
    context_parts.append(f"Net Savings: ${total_income - total_expenses:.2f}")
    
    return "\n".join(context_parts)

def is_financial_calculation_query(question):
    """
    Detect if the question is asking for financial calculations that require ALL transactions.
    Only triggers for explicit calculation/total requests, not general advice questions.
    """
    question_lower = question.lower()
    
    # Strong calculation indicators - must have one of these
    strong_calc_keywords = [
        'how much did i save', 'how much did i spend', 'how much have i save', 'how much have i spend',
        'total income', 'total expense', 'total spending', 'total savings',
        'calculate my', 'sum of', 'add up',
        'what did i spend', 'what have i spent', 'what did i save',
        'all my expenses', 'all my spending', 'all my transactions',
        'savings in', 'spent in', 'expenses in'  # e.g., "savings in October"
    ]
    
    # Check if question contains strong calculation indicators
    if any(keyword in question_lower for keyword in strong_calc_keywords):
        return True
    
    # Also trigger for single-word calculation queries
    simple_calc_words = ['balance', 'breakdown']
    # Only if the question is short and focused on calculations
    if any(word in question_lower for word in simple_calc_words):
        # Additional check: question should be short and not asking for advice
        if len(question.split()) < 10 and 'should' not in question_lower and 'advice' not in question_lower:
            return True
    
    return False

def is_audit_query(question):
    """
    Detect if the question is asking for an audit or comparison between contract and actual spending.
    """
    keywords = ['audit', 'contract', 'agreement', 'policy', 'overcharged', 'charged too much', 'correct amount', 'supposed to pay']
    return any(k in question.lower() for k in keywords)

def perform_audit(user_id, question):
    """
    Perform an audit by comparing contract terms (from RAG) with actual spending (from MongoDB).
    """
    print(f"🕵️‍♀️ perform_audit: Starting audit for question: {question}")
    
    # 1. Retrieve contract terms using RAG
    # We specifically look for contracts/agreements
    audit_query = f"{question} contract agreement terms price cost"
    res = query_user(user_id, audit_query, top_k=5)
    contract_context = res.get("context", "")
    
    if not contract_context:
        return {
            "question": question,
            "answer": "I couldn't find any uploaded contracts or agreements to audit. Please upload a contract PDF first.",
            "context": "No contracts found.",
            "results": []
        }

    # 2. Use LLM to extract Merchant and Expected Amount from the contract context
    extraction_prompt = f"""
    Based on the following contract text, identify the Service Provider (Merchant) and the Expected Payment Amount/Frequency.
    
    Contract Text:
    {contract_context}
    
    Return ONLY a JSON object with keys: "merchant_name", "expected_amount" (number), "frequency" (e.g. monthly), "currency".
    If not found, return null for values.
    """
    
    # We use a separate LLM call for extraction to be precise
    extraction_response = generate_answer(contract_context, "Extract merchant and amount as JSON", max_tokens=100)
    print(f"🕵️‍♀️ Extraction response: {extraction_response}")
    
    merchant_name = ""
    try:
        # Simple heuristic to find merchant name if JSON parsing fails or is simple text
        # In a real app, we'd use structured output or better parsing
        import re
        # Try to find JSON-like structure
        json_match = re.search(r'\{.*\}', extraction_response, re.DOTALL)
        if json_match:
            import json
            data = json.loads(json_match.group(0))
            merchant_name = data.get("merchant_name")
    except:
        pass
        
    # If extraction failed, we'll just proceed with a general comparison using the context
    
    # 3. Retrieve actual transactions from MongoDB
    # We'll fetch all expenses and let the LLM filter, or we could try to filter by merchant_name if we extracted it
    expenses_coll = db["expenses"]
    # Get last 6 months of transactions to be safe
    start_date = (datetime.datetime.utcnow() - datetime.timedelta(days=180)).strftime("%Y-%m-%d")
    
    query = {
        "username": user_id,
        "date": {"$gte": start_date},
        "type": "expense"
    }
    
    # If we have a merchant name, we could try to filter, but regex search in Mongo is better done by LLM on a smaller set
    # or we just fetch all and let LLM find the matches. Fetching all recent expenses is safer.
    transactions = list(expenses_coll.find(query).sort("date", -1).limit(100))
    
    transaction_text = "Recent Transactions:\n"
    for t in transactions:
        transaction_text += f"- {t.get('date')}: {t.get('description')} (${t.get('amount')})\n"
        
    # 4. Generate Audit Report
    final_context = f"""
    CONTRACT TERMS:
    {contract_context}
    
    ACTUAL SPENDING (Last 6 months):
    {transaction_text}
    """
    
    audit_prompt = f"""
    You are a Financial Auditor. Compare the Contract Terms with the Actual Spending.
    
    Question: {question}
    
    Task:
    1. Identify the specific cost mentioned in the contract.
    2. Identify the actual payments made for this service in the transaction list.
    3. Determine if there is a discrepancy (overcharge, undercharge, or correct).
    4. Highlight the difference.
    
    Response Format:
    - **Status**: [Overcharged / Correct / Undercharged / Unclear]
    - **Contract Rate**: [Amount from contract]
    - **Actual Charged**: [Amount from transactions]
    - **Analysis**: [Brief explanation]
    """
    
    answer = generate_answer(final_context, audit_prompt, max_tokens=512)
    
    return {
        "question": question,
        "answer": answer,
        "context": final_context,
        "results": res.get("results", [])
    }

from modules.chatbot_share.financial_rates import get_consolidated_rates

def is_rate_query(question):
    """
    Detect if the question is asking for live financial rates.
    """
    keywords = ['rate', 'mortgage', 'gic', 'savings', 'interest', 'best deal', 'offer']
    return any(k in question.lower() for k in keywords)

def answer_user(user_id: str, question: str, top_k: int = 20) -> dict:
    """
    Retrieve relevant chunks, build context, call Groq to generate an answer.
    Returns JSON-safe response.
    For financial calculation queries, retrieves ALL bank statement data directly from MongoDB.
    """
    print(f"📊 answer_user: querying for user_id={user_id}, question={question[:50]}...", flush=True)
    
    # Check for Audit Query
    if is_audit_query(question):
        print("🕵️‍♀️ Detected AUDIT query, performing audit workflow...")
        response = perform_audit(user_id, question)
        return make_json_safe(response)

    # Detect if this is a financial calculation query
    if is_financial_calculation_query(question):
        print(f"💰 Detected financial calculation query, retrieving ALL bank transactions...", flush=True)
        
        # Get complete bank statement data directly from MongoDB
        bank_context = get_all_bank_transactions(user_id)
        
        # Also get some RAG context for additional info (profile, investments)
        res = query_user(user_id, question, top_k=5)
        rag_context = res.get("context", "")
        
        # Combine contexts - prioritize bank statement data
        if bank_context:
            context = f"{bank_context}\n\n=== ADDITIONAL CONTEXT ===\n{rag_context}"
        else:
            context = rag_context
        
        results = res.get("results", [])
        raw_matches = res.get("raw_matches", [])
        
        print(f"💰 Using complete bank statement data with {len(context)} chars", flush=True)
        
    elif is_rate_query(question):
        print("📈 Detected RATE query, fetching live rates...")
        live_rates = get_consolidated_rates()
        
        # Also get RAG context in case there's personal info needed
        res = query_user(user_id, question, top_k=5)
        rag_context = res.get("context", "")
        
        context = f"LIVE FINANCIAL RATES:\n{live_rates}\n\nUSER CONTEXT:\n{rag_context}"
        results = res.get("results", [])
        raw_matches = res.get("raw_matches", [])
        
    else:
        # Use normal RAG retrieval for other queries
        res = query_user(user_id, question, top_k=top_k)
        context = res.get("context", "")
        results = res.get("results", [])
        raw_matches = res.get("raw_matches", [])
        print(f"📊 answer_user: query_user returned, got {len(results)} results", flush=True)
    
    if not context:
        context = "No relevant documents found."
    
    print(f"📊 answer_user: calling generate_answer with context length={len(context)}", flush=True)
    answer = generate_answer(context, question, max_tokens=256, temperature=0.0)  # Short, concise responses
    print(f"📊 answer_user: generate_answer returned, answer length={len(answer)}", flush=True)
    
    # Build response and make it JSON-safe
    print(f"📊 answer_user: building response dict", flush=True)
    response = {
        "question": question,
        "answer": answer,
        "context": context,
        "raw_matches": raw_matches,
        "results": results
    }
    print(f"📊 answer_user: calling make_json_safe", flush=True)
    safe_response = make_json_safe(response)
    print(f"📊 answer_user: make_json_safe complete, returning", flush=True)
    return safe_response