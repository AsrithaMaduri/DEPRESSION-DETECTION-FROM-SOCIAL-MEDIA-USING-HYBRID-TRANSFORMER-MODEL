import numpy as np
depression_words = [
"sad","lonely","tired","hopeless","worthless",
"depressed","cry","pain","empty"
]

def extract_features(text):
    words = text.split()

    word_count = len(words)

    first_person = sum(1 for w in words if w in ["i","me","my","myself"])

    neg_words = sum(1 for w in words if w in depression_words)

    avg_len = np.mean([len(w) for w in words]) if words else 0

    return np.array([word_count, first_person, neg_words, avg_len])
