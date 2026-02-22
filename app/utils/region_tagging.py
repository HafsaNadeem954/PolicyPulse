"""
Region Tagging Utility Module
Automatically infers province/region from text based on keyword rules
"""

def infer_region(text):
    """
    Infer region from text using predefined keyword rules.
    
    Regions targeted:
    - Punjab: 500 rows (keywords: lahore, punjab, faisalabad, etc.)
    - Sindh: 350 rows (keywords: karachi, sindh, hyderabad)
    - KPK: 300 rows (keywords: peshawar, kpk, khyber, swat)
    - Balochistan: 200 rows (keywords: quetta, balochistan)
    - Islamabad: 200 rows (keywords: islamabad, capital)
    - National: 250 rows (keywords: pakistan, national, country-wide)
    - Unknown: remainder
    
    Args:
        text (str): Text to analyze (comment, title, etc.)
    
    Returns:
        str: Region name or "Unknown"
    """
    if not text:
        return "Unknown"
    
    text_lower = text.lower()
    
    # Punjab region keywords
    punjab_keywords = ["lahore", "punjab", "faisalabad", "multan", "sialkot", 
                       "rawalpindi", "gujrat", "sargodha", "bahawalpur", "jhang"]
    if any(keyword in text_lower for keyword in punjab_keywords):
        return "Punjab"
    
    # Sindh region keywords
    sindh_keywords = ["karachi", "sindh", "hyderabad", "sukkur", "larkana", 
                      "nawabshah", "tando adam", "badin"]
    if any(keyword in text_lower for keyword in sindh_keywords):
        return "Sindh"
    
    # KPK region keywords
    kpk_keywords = ["peshawar", "kpk", "khyber", "swat", "mardan", "abbottabad",
                    "kohat", "bannu", "mingora", "dir"]
    if any(keyword in text_lower for keyword in kpk_keywords):
        return "KPK"
    
    # Balochistan region keywords
    balochistan_keywords = ["quetta", "balochistan", "gwadar", "zhob", "turbat", "sibi"]
    if any(keyword in text_lower for keyword in balochistan_keywords):
        return "Balochistan"
    
    # Islamabad region keywords
    islamabad_keywords = ["islamabad", "capital", "federal"]
    if any(keyword in text_lower for keyword in islamabad_keywords):
        return "Islamabad"
    
    # National/Country-wide keywords
    national_keywords = ["pakistan", "national", "country", "countrywide", "nationwide", 
                        "all pakistan", "whole country"]
    if any(keyword in text_lower for keyword in national_keywords):
        return "National"
    
    # Default unknown
    return "Unknown"


def batch_region_tagging(rows, text_fields):
    """
    Apply region tagging to a batch of rows.
    
    Args:
        rows (list): List of dictionaries to tag
        text_fields (list): List of field names to check for region inference
                           e.g., ["comment", "video_title", "channel"]
    
    Returns:
        list: Updated rows with region column
    """
    for row in rows:
        combined_text = " ".join([str(row.get(field, "")) for field in text_fields])
        row["region"] = infer_region(combined_text)
    
    return rows


def region_distribution_report(rows):
    """
    Generate a report of region distribution in collected data.
    
    Args:
        rows (list): List of dictionaries with 'region' column
    
    Returns:
        dict: Region counts
    """
    distribution = {}
    for row in rows:
        region = row.get("region", "Unknown")
        distribution[region] = distribution.get(region, 0) + 1
    
    return distribution
