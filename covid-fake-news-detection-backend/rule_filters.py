"""
Rule-based filters for obvious fake news patterns
"""
import re

def apply_rule_filters(text, prediction, confidence):
    """
    Apply rule-based filters to catch obvious fake news that the AI model might miss
    """
    text_lower = text.lower()
    
    # Rule 1: Individual person blame for COVID-19 origin/creation
    individual_blame_patterns = [
        r'\b\w+\s+(is|was)\s+(responsible|to blame)\s+for\s+covid',
        r'\b\w+\s+(created|made|caused)\s+covid',
        r'\b\w+\s+(started|began)\s+covid',
        r'covid\s+(was created|is caused|was made)\s+by\s+\w+',
    ]
    
    for pattern in individual_blame_patterns:
        if re.search(pattern, text_lower):
            return "FAKE", 0.8, "Rule: Individual blame for COVID-19 origin"
    
    # Rule 2: Technology responsibility for COVID
    tech_blame_patterns = [
        r'5g\s+(is|was)\s+(responsible|to blame)\s+for\s+covid',
        r'wifi\s+(is|was)\s+(responsible|caused)\s+covid',
        r'phone\s+(tower|signal)\s+.*(responsible|caused)\s+covid',
    ]
    
    for pattern in tech_blame_patterns:
        if re.search(pattern, text_lower):
            return "FAKE", 0.9, "Rule: Technology blamed for COVID-19"
    
    # Rule 3: Conspiracy theory keywords
    conspiracy_patterns = [
        r'covid\s+.*(hoax|fake|scam|lie)',
        r'(planned|designed|engineered)\s+.+covid',
        r'covid\s+.*(population control|depopulation)',
    ]
    
    for pattern in conspiracy_patterns:
        if re.search(pattern, text_lower):
            return "FAKE", 0.85, "Rule: COVID conspiracy theory"
    
    # Rule 4: Medical misinformation
    medical_misinfo_patterns = [
        r'vaccines\s+.*(kill|deadly|poison|toxic)',
        r'masks\s+.*(harmful|dangerous|kill)',
        r'covid\s+.*(not real|doesn\'t exist|fake)',
    ]
    
    for pattern in medical_misinfo_patterns:
        if re.search(pattern, text_lower):
            return "FAKE", 0.85, "Rule: Medical misinformation"
    
    # If no rules triggered, return original prediction
    return prediction, confidence, "AI Model"