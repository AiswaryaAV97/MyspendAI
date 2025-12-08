def calculate_min_down_payment(home_price):
    """
    Minimum down payment as per Canadian rules:
    - 5% of first $500k
    - 10% of amount above $500k
    """
    home_price = float(home_price)
    if home_price <= 500000:
        return home_price * 0.05
    else:
        return 500000 * 0.05 + (home_price - 500000) * 0.10


def calculate_cmhc(principal, down_payment, home_price):
    """
    CMHC insurance for down payment < 20%
    Roughly based on CMHC rates
    """
    try:
        home_price = float(home_price)
    except Exception:
        return 0
    if home_price == 0:
        return 0
    down_percent = down_payment / home_price
    if down_percent >= 0.2:
        return 0  # no CMHC required
    if down_percent >= 0.15:
        rate = 0.028
    elif down_percent >= 0.10:
        rate = 0.031
    else:
        rate = 0.04
    return round(principal * rate, 2)



def calculate_monthly_payment(principal, rate, years):

    return calculate_periodic_payment(principal, rate, years, 12)


def calculate_periodic_payment(principal, annual_rate, years, payments_per_year):
    """
    Generic periodic payment calculator for arbitrary payment frequency.
    payments_per_year: 12 for monthly, 26 for biweekly, etc.
    """
    try:
        principal = float(principal)
        annual_rate = float(annual_rate)
        years = int(years)
    except Exception:
        return 0
    if years <= 0:
        return 0
    r = annual_rate / 100 / payments_per_year
    n = years * payments_per_year
    if r == 0:
        payment = principal / n
    else:
        payment = principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return round(payment, 2)


def calculate_ltt(home_price, province, city=None, first_time=False):
    """
    Returns provincial LTT, municipal LTT, rebate, and total LTT
    (Simplified, approximate rules for a few provinces/cities)
    """
    provincial_tax = 0
    municipal_tax = 0
    rebate = 0
    try:
        home_price = float(home_price)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

    prov = (province or "").upper()


    # --- Provincial LTT simplified ---
    if province == "ON":
        if home_price <= 55000:
            provincial_tax = home_price * 0.005
        elif home_price <= 250000:
            provincial_tax = 275 + (home_price - 55000) * 0.01
        elif home_price <= 400000:
            provincial_tax = 2225 + (home_price - 250000) * 0.015
        else:
            provincial_tax = 4425 + (home_price - 400000) * 0.02
        if first_time:
           # first-time rebate fixed at $8,000, but cannot exceed combined taxes
           # (apply cap to avoid negative total)
           rebate = min(8000, provincial_tax)
        if city and city.lower() == "toronto":
            municipal_tax = home_price * 0.005
            if first_time:
                 # allow municipal portion to be included in rebate up to remaining cap
                municipal_rebate = min(municipal_tax, max(0, 8000 - rebate))
                rebate += municipal_rebate


    elif province == "BC":
        if home_price <= 200000:
            provincial_tax = 0
        elif home_price <= 2000000:
            provincial_tax = home_price * 0.02
        else:
            provincial_tax = home_price * 0.03
        if first_time and home_price <= 500000:
            rebate = provincial_tax  # full rebate

    elif province == "QC":
        if home_price <= 50000:
            provincial_tax = 0.005 * home_price
        elif home_price <= 250000:
            provincial_tax = 0.01 * home_price
        else:
            provincial_tax = 0.015 * home_price
        if first_time:
            rebate = min(provincial_tax, 4000)

    # Provinces / territories without LTT (simplified)
    elif province in ["AB", "SK", "NL", "MB", "NB", "NS", "PE", "NT", "YT", "NU"]:
        provincial_tax = 0
        municipal_tax = 0
        rebate = 0

    total_ltt = provincial_tax + municipal_tax - rebate
    # ensure not negative
    total_ltt = max(total_ltt, 0)
    return round(provincial_tax, 2), round(municipal_tax, 2), round(rebate, 2), round(total_ltt, 2)


def calculate_mortgage_with_cmhc(price, down_payment, rate, years, province="", city=None, first_time=False):
    """
    High-level calculator that returns:
    - price, down_payment, principal, min_down_payment
    - cmhc_fee, monthly_payment, interest_rate, years
    - ltt breakdown (provincial, municipal, rebate, total_ltt)
    Keeps backward compatibility with previous return fields.
    """
    # sanitize inputs
    try:
        price = float(price)
        down_payment = float(down_payment)
        rate = float(rate)
        years = int(years)
    except (TypeError, ValueError):
        raise ValueError("Invalid numeric inputs for mortgage calculation")

    if price <= 0:
        raise ValueError("Price must be > 0")
    if down_payment < 0 or down_payment > price:
        raise ValueError("Down payment must be between 0 and price")
    if years <= 0:
        raise ValueError("Years must be > 0")
    if rate < 0:
        raise ValueError("Rate must be >= 0")

    principal = price - down_payment

    # CMHC calculation
    cmhc_fee = calculate_cmhc(principal, down_payment, price)
    principal_with_cmhc = principal + cmhc_fee

    # monthly and biweekly payments
    monthly_payment = calculate_periodic_payment(principal_with_cmhc, rate, years, 12)
    biweekly_payment = calculate_periodic_payment(principal_with_cmhc, rate, years, 26)

    # min down payment
    min_down = calculate_min_down_payment(price)

    # LTT (land transfer tax) calculation (simplified)
    prov_tax, mun_tax, rebate, total_ltt = calculate_ltt(price, province, city, bool(first_time))

    result = {
        "price": price,
        "down_payment": down_payment,
        "min_down_payment": round(min_down, 2),
        "principal": round(principal, 2),
        "cmhc_fee": round(cmhc_fee, 2),
        "principal_with_cmhc": round(principal_with_cmhc, 2),
        "interest_rate": rate,
        "years": years,
        "monthly_payment": monthly_payment,
        "biweekly_payment": biweekly_payment,
        "ltt": {
            "provincial": prov_tax,
            "municipal": mun_tax,
            "rebate": rebate,
            "total": total_ltt
        }
    }

    return result
