import random

def generate_attention(text):

    words = text.split()

    if not words:
        return {}

    attention = {}

    for w in words:
        attention[w] = round(random.uniform(0.1, 1.0), 2)

    # Normalize (optional but better)
    total = sum(attention.values())

    for w in attention:
        attention[w] = round(attention[w] / total, 2)

    return attention