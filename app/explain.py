import re

depression_keywords = [
    "sad","alone","tired","worthless","hopeless",
    "cry","pain","depressed","lonely","suicide"
]

def explain_prediction(text):

    words = re.findall(r'\w+', text.lower())

    important = []

    for w in words:
        if w in depression_keywords:
            important.append(w)

    return list(set(important))