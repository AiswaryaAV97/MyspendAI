"""
Bank Statement Parser - Extract transactions from PDF bank statements
"""
import pdfplumber
import re
from datetime import datetime
from typing import List, Dict

def parse_bank_statement(file_path: str) -> Dict:
    """
    Parse a bank statement PDF and extract transaction data.
    
    Returns a dict with:
    - account_holder: str
    - account_number: str (masked)
    - statement_period: str
    - transactions: List[Dict] with date, description, category, amount
    - opening_balance: float
    - closing_balance: float
    - total_deposits: float
    - total_withdrawals: float
    """
    
    result = {
        "account_holder": "",
        "account_number": "",
        "statement_period": "",
        "opening_balance": 0.0,
        "closing_balance": 0.0,
        "total_deposits": 0.0,
        "total_withdrawals": 0.0,
        "transactions": []
    }
    
    with pdfplumber.open(file_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    
    # Extract account holder
    holder_match = re.search(r'Account Holder:\s*(.+)', full_text, re.IGNORECASE)
    if holder_match:
        result["account_holder"] = holder_match.group(1).strip()
    
    # Extract account number
    account_match = re.search(r'Account Number:\s*(.+)', full_text, re.IGNORECASE)
    if account_match:
        result["account_number"] = account_match.group(1).strip()
    
    # Extract statement period
    period_match = re.search(r'Statement Period:\s*(.+)', full_text, re.IGNORECASE)
    if period_match:
        result["statement_period"] = period_match.group(1).strip()
    
    # Extract balances
    opening_match = re.search(r'Opening Balance:\s*\$?([\d,]+\.?\d*)', full_text, re.IGNORECASE)
    if opening_match:
        result["opening_balance"] = float(opening_match.group(1).replace(',', ''))
    
    closing_match = re.search(r'Closing Balance:\s*\$?([\d,]+\.?\d*)', full_text, re.IGNORECASE)
    if closing_match:
        result["closing_balance"] = float(closing_match.group(1).replace(',', ''))
    
    deposits_match = re.search(r'Total Deposits:\s*\$?([\d,]+\.?\d*)', full_text, re.IGNORECASE)
    if deposits_match:
        result["total_deposits"] = float(deposits_match.group(1).replace(',', ''))
    
    withdrawals_match = re.search(r'Total Withdrawals:\s*\$?([\d,]+\.?\d*)', full_text, re.IGNORECASE)
    if withdrawals_match:
        result["total_withdrawals"] = float(withdrawals_match.group(1).replace(',', ''))
    
    # Extract transactions
    # Pattern: Date | Description | Category | Amount | Balance
    # Example: Oct 1 DIRECT DEPOSIT - SALARY Income +$4,500.00 $9,734.56
    
    transaction_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+)\s+(.+?)\s+(Income|Housing|Groceries|Transportation|Utilities|Shopping|Dining|Health|Payment|Entertainment|Other)\s+([\+\-]\$[\d,]+\.?\d*)\s+\$([\d,]+\.?\d*)'
    
    transactions = re.findall(transaction_pattern, full_text)
    
    current_year = datetime.now().year
    
    for trans in transactions:
        date_str, description, category, amount_str, balance_str = trans
        
        # Parse amount
        amount = float(amount_str.replace('+', '').replace('-', '').replace('$', '').replace(',', ''))
        if '-' in amount_str:
            amount = -amount
        
        # Parse date
        try:
            trans_date = datetime.strptime(f"{date_str} {current_year}", "%b %d %Y")
        except:
            trans_date = datetime.now()
        
        result["transactions"].append({
            "date": trans_date.strftime("%Y-%m-%d"),
            "description": description.strip(),
            "category": category.strip(),
            "amount": amount,
            "balance": float(balance_str.replace(',', ''))
        })
    
    return result


def categorize_transaction(description: str) -> str:
    """
    Auto-categorize a transaction based on description keywords.
    """
    description_lower = description.lower()
    
    categories = {
        "Groceries": ["grocery", "supermarket", "loblaws", "metro", "sobeys", "costco", "walmart"],
        "Dining": ["restaurant", "cafe", "coffee", "starbucks", "tim hortons", "pizza", "burger"],
        "Transportation": ["gas", "fuel", "shell", "esso", "petro", "uber", "taxi", "transit"],
        "Utilities": ["hydro", "gas", "electric", "water", "internet", "phone", "rogers", "bell", "telus"],
        "Housing": ["rent", "mortgage", "condo", "property tax"],
        "Health": ["pharmacy", "drug", "medical", "doctor", "dental", "gym", "fitness"],
        "Shopping": ["amazon", "walmart", "target", "best buy", "clothing", "store"],
        "Entertainment": ["movie", "theatre", "netflix", "spotify", "game", "concert"],
        "Income": ["salary", "deposit", "payroll", "income", "transfer in"],
        "Payment": ["payment", "transfer", "credit card"]
    }
    
    for category, keywords in categories.items():
        if any(keyword in description_lower for keyword in keywords):
            return category
    
    return "Other"


if __name__ == "__main__":
    # Test the parser
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "mock_bank_statement.pdf"
    
    print(f"Parsing: {file_path}")
    result = parse_bank_statement(file_path)
    
    print(f"\n📊 Account Holder: {result['account_holder']}")
    print(f"🏦 Account Number: {result['account_number']}")
    print(f"📅 Period: {result['statement_period']}")
    print(f"💰 Opening Balance: ${result['opening_balance']:,.2f}")
    print(f"💰 Closing Balance: ${result['closing_balance']:,.2f}")
    print(f"📈 Total Deposits: ${result['total_deposits']:,.2f}")
    print(f"📉 Total Withdrawals: ${result['total_withdrawals']:,.2f}")
    print(f"\n📝 Transactions Found: {len(result['transactions'])}")
    
    for i, trans in enumerate(result['transactions'][:5], 1):
        print(f"  {i}. {trans['date']} | {trans['description'][:30]} | {trans['category']} | ${trans['amount']:,.2f}")
