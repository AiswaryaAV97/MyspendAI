import requests
from bs4 import BeautifulSoup

def get_live_rates():
    """
    Scrapes Ratehub.ca for live financial rates.
    Returns a string summary of the best rates.
    """
    try:
        # 1. Mortgage Rates
        mortgage_url = "https://www.ratehub.ca/best-mortgage-rates"
        response = requests.get(mortgage_url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        
        mortgage_rates = []
        # Look for the "best-rates-table" or similar structure
        # Note: Scraping is brittle. We'll look for specific text patterns if classes change.
        
        # Try to find the main "Best 5-year fixed" rate often displayed prominently
        best_fixed = soup.find("div", string=lambda t: t and "5-Year Fixed" in t)
        if best_fixed:
            # Navigate to the rate value usually nearby
            rate_val = best_fixed.find_next("div", class_="rate-value") # Hypothetical class
            if not rate_val:
                 # Fallback: look for percentage pattern nearby
                 pass
        
        # Simpler approach: Extract the first few occurrences of "%" associated with "Fixed" or "Variable"
        # Ratehub often puts the best rate in a large font.
        
        # Let's try a more robust text extraction for now since we don't know exact classes
        text = soup.get_text()
        
        rates_summary = "Live Financial Rates (Source: Ratehub.ca):\n"
        
        # 2. GIC Rates
        gic_url = "https://www.ratehub.ca/gics/best-gic-rates"
        try:
            gic_resp = requests.get(gic_url, timeout=10)
            gic_soup = BeautifulSoup(gic_resp.content, "html.parser")
            # Look for "1-Year" and the rate
            # This is a placeholder for the actual scraping logic which needs to be adjusted 
            # based on the actual HTML structure we saw or will see.
            # For now, we will return a generic message if we can't parse specific classes,
            # but we will try to find the first percentage after "1-Year"
            
            rates_summary += "- GIC Rates: Please check ratehub.ca/gics (Scraper implementation pending exact selector)\n"
        except Exception as e:
            rates_summary += f"- GIC Rates: Unavailable ({str(e)})\n"

        # 3. Savings Rates
        savings_url = "https://www.ratehub.ca/savings-accounts/best-savings-accounts"
        try:
            sav_resp = requests.get(savings_url, timeout=10)
            # Logic to extract savings rate
            rates_summary += "- Savings Rates: Please check ratehub.ca/savings-accounts\n"
        except:
            pass
            
        return rates_summary + "\n(Note: Automated scraping is best-effort. Visit sites for official rates.)"

    except Exception as e:
        return f"Could not fetch live rates: {str(e)}"

def scrape_ratehub_mortgage():
    """
    Specific scraper for Ratehub mortgage page based on observed structure.
    Returns list of strings with rate and provider info.
    """
    url = "https://www.ratehub.ca/best-mortgage-rates"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        
        rates = []
        
        # Find all text nodes
        all_text = soup.get_text(separator=' ', strip=True)
        
        import re
        
        # Look for "5-Year Fixed" followed by rate, then try to extract provider
        # Pattern: "5-Year Fixed ... 3.79% ... Provider Name"
        fixed_5yr_match = re.search(r'5-Year Fixed[^\d]*([\d\.]+%)', all_text, re.IGNORECASE)
        if fixed_5yr_match:
            rate = fixed_5yr_match.group(1)
            # Try to find provider name after the rate (usually within next 50 chars)
            text_after_rate = all_text[fixed_5yr_match.end():fixed_5yr_match.end()+100]
            provider_match = re.search(r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)', text_after_rate)
            if provider_match:
                provider = provider_match.group(1).strip()
                # Filter out common noise words
                if provider and len(provider) > 3 and provider.lower() not in ['rate', 'mortgage', 'best', 'lowest', 'year', 'fixed', 'variable']:
                    rates.append(f"5-Year Fixed: {rate} (Provider: {provider})")
                else:
                    rates.append(f"5-Year Fixed: {rate}")
            else:
                rates.append(f"5-Year Fixed: {rate}")
        
        # Variable rate
        variable_5yr_match = re.search(r'5-Year Variable[^\d]*([\d\.]+%)', all_text, re.IGNORECASE)
        if variable_5yr_match:
            rate = variable_5yr_match.group(1)
            text_after_rate = all_text[variable_5yr_match.end():variable_5yr_match.end()+100]
            provider_match = re.search(r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)', text_after_rate)
            if provider_match:
                provider = provider_match.group(1).strip()
                if provider and len(provider) > 3 and provider.lower() not in ['rate', 'mortgage', 'best', 'lowest', 'year', 'fixed', 'variable']:
                    rates.append(f"5-Year Variable: {rate} (Provider: {provider})")
                else:
                    rates.append(f"5-Year Variable: {rate}")
            else:
                rates.append(f"5-Year Variable: {rate}")
            
        return rates
    except Exception as e:
        print(f"Error scraping mortgage: {e}")
        return []

def scrape_ratehub_gic():
    """
    Scraper for GIC rates with provider information.
    """
    url = "https://www.ratehub.ca/gics/best-gic-rates"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        all_text = soup.get_text(separator=' ', strip=True)
        import re
        
        # Look for 1-Year GIC with provider
        gic_1yr_match = re.search(r'1-Year[^\d]*([\d\.]+%)', all_text, re.IGNORECASE)
        if gic_1yr_match:
            rate = gic_1yr_match.group(1)
            text_after_rate = all_text[gic_1yr_match.end():gic_1yr_match.end()+100]
            provider_match = re.search(r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)', text_after_rate)
            if provider_match:
                provider = provider_match.group(1).strip()
                if provider and len(provider) > 3 and provider.lower() not in ['rate', 'gic', 'best', 'highest', 'year']:
                    return [f"1-Year GIC: {rate} (Provider: {provider})"]
            return [f"1-Year GIC: {rate}"]
        return []
    except:
        return []

def scrape_ratehub_savings():
    """
    Scraper for savings account rates with provider information.
    """
    url = "https://www.ratehub.ca/savings-accounts/best-savings-accounts"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        all_text = soup.get_text(separator=' ', strip=True)
        import re
        
        # Look for high interest savings with provider
        hisa_match = re.search(r'(?:High Interest|HISA)[^\d]*([\d\.]+%)', all_text, re.IGNORECASE)
        if hisa_match:
            rate = hisa_match.group(1)
            text_after_rate = all_text[hisa_match.end():hisa_match.end()+100]
            provider_match = re.search(r'([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)', text_after_rate)
            if provider_match:
                provider = provider_match.group(1).strip()
                if provider and len(provider) > 3 and provider.lower() not in ['rate', 'account', 'savings', 'best', 'highest']:
                    return [f"High Interest Savings: {rate} (Provider: {provider})"]
            return [f"High Interest Savings: {rate}"]
        return []
    except:
        return []

def get_consolidated_rates():
    """
    Main function to get all rates with provider information and source links
    """
    summary = ["**Live Financial Rates**\n"]
    
    m_rates = scrape_ratehub_mortgage()
    if m_rates:
        summary.append("**Mortgage Rates:**")
        for rate in m_rates:
            summary.append(f"  • {rate}")
        summary.append("  Source: [Ratehub Mortgage Rates](https://www.ratehub.ca/best-mortgage-rates)\n")
    else:
        summary.append("**Mortgage:** Could not fetch live data.")
        summary.append("  Visit: https://www.ratehub.ca/best-mortgage-rates\n")
        
    g_rates = scrape_ratehub_gic()
    if g_rates:
        summary.append("**GIC Rates:**")
        for rate in g_rates:
            summary.append(f"  • {rate}")
        summary.append("  Source: [Ratehub GIC Rates](https://www.ratehub.ca/gics/best-gic-rates)\n")
        
    s_rates = scrape_ratehub_savings()
    if s_rates:
        summary.append("**Savings Account Rates:**")
        for rate in s_rates:
            summary.append(f"  • {rate}")
        summary.append("  Source: [Ratehub Savings Rates](https://www.ratehub.ca/savings-accounts/best-savings-accounts)\n")
        
    summary.append("\n*Note: Rates are updated in real-time from Ratehub.ca. Visit the links above for complete details and terms.*")
    
    return "\n".join(summary)


