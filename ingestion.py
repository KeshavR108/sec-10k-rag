import re
from bs4 import BeautifulSoup

ITEM_ANCHORS = [
    ("Business",             r"Item\s*1\.\s*Business",              True),
    ("Risk Factors",         r"Item\s*1A\.\s*Risk\s*Factors",       True),
    ("Unresolved Comments",  r"Item\s*1B\.\s*Unresolved",           False),
    ("Cybersecurity",        r"Item\s*1C\.\s*Cybersecurity",        False),
    ("Properties",           r"Item\s*2\.\s*Properties",            False),
    ("Legal Proceedings",    r"Item\s*3\.\s*Legal\s*Proceedings",   False),
    ("Mine Safety",          r"Item\s*4\.\s*Mine\s*Safety",         False),
    ("Market for Stock",     r"Item\s*5\.\s*Market\s*for",          False),
    ("Selected Data",        r"Item\s*6\.",                         False),
    ("MD&A",                 r"Item\s*7\.\s*Management",            True),
    ("Market Risk",          r"Item\s*7A\.\s*Quantitative",         False),
    ("Financial Statements", r"Item\s*8\.\s*Financial\s*Statements", True),
    ("Accountant Changes",   r"Item\s*9\.\s*Changes",               False),
]


def parse_10k_html(html_content: str, company_name: str) -> dict:
    soup = BeautifulSoup(html_content, "html.parser")
    full_text = soup.get_text(separator=" ", strip=True)
    full_text = re.sub(r"\s+", " ", full_text)

    positions = []
    for name, pattern, keep in ITEM_ANCHORS:
        matches = list(re.finditer(pattern, full_text, flags=re.IGNORECASE))
        if matches:
            positions.append((matches[-1].start(), name, keep))

    if not positions:
        return {}

    positions.sort(key=lambda x: x[0])

    extracted = {}
    for i, (start, name, keep) in enumerate(positions):
        if not keep:
            continue
        end = positions[i + 1][0] if i + 1 < len(positions) else len(full_text)
        body = full_text[start:end].strip()
        if len(body) > 50:
            extracted[name] = body

    return extracted
