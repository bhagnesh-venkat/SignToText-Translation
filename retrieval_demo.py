"""
Mini retrieval demo: given a query, find the most similar English sentences.

This is the "retrieval" idea of the project, shown on text only.
Later, the query will come from the sign features instead of typed text.

Setup (once, inside your slt-env):
    pip install sentence-transformers

Usage:
    python retrieval_demo.py                      # type queries, press Enter
    python retrieval_demo.py --query "Good Morning"   # one query, then exit

The first run downloads a small text model (about 90 MB), so it needs internet once.
"""
import argparse

import numpy as np
from sentence_transformers import SentenceTransformer

# Example sentences written for this demo. Later this database will be built
# from the TRAINING sentences of a real ISL dataset (never the test sentences).
SENTENCES = [
    "Hello, nice to meet you.",
    "Hello, how are you today?",
    "Good morning, did you sleep well?",
    "Good morning, how are you?",
    "Good morning everyone, let us begin the class.",
    "Good afternoon, welcome to the office.",
    "Good afternoon, have you had lunch?",
    "Good evening, please come in.",
    "Good night, see you tomorrow.",
    "How are you? I am fine, thank you.",
    "How are you feeling today?",
    "I am fine, how about you?",
    "Alright, I will see you tomorrow.",
    "Is everything alright?",
    "Yes, everything is alright.",
    "Everything is going alright.",
    "Thank you very much for your help.",
    "What is your name?",
    "My name is Ravi.",
    "Where do you live?",
    "I live in Chennai.",
    "Please sit down.",
    "Are you hungry?",
    "I am going to school.",
    "What time is it?",
    "It is ten o'clock in the morning.",
    "Wake up, it is already morning.",
    "The weather is hot today.",
    "Do you need any help?",
    "I am sorry, I did not understand.",
    "Please repeat that slowly.",
    "See you later.",
    "Welcome to our home.",
    "Happy birthday to you.",
    "Congratulations on your success.",
    "Excuse me, where is the bus stop?",
    "Take care of yourself.",
    "Have a nice day.",
    "I will call you in the afternoon.",
    "The doctor is available in the afternoon.",
]


def build_index(model, sentences):
    """Turn every sentence into a vector (a list of numbers that captures its meaning).
    Vectors are normalised to length 1, so a dot product equals cosine similarity."""
    vectors = model.encode(sentences, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)          # shape: (num_sentences, 384)


def search(model, index, sentences, query, k=3):
    """Return the k sentences most similar to the query, with their scores."""
    q = np.asarray(model.encode([query], normalize_embeddings=True)[0], dtype=np.float32)
    scores = index @ q                                     # one similarity score per sentence
    best = np.argsort(-scores)[:k]                         # indices of the highest scores
    return [(sentences[i], float(scores[i])) for i in best]


def show(results):
    for rank, (sentence, score) in enumerate(results, 1):
        print(f"  {rank}. {sentence}   (similarity {score:.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=None, help="Run one query and exit")
    parser.add_argument("--k", type=int, default=3, help="How many results to show")
    args = parser.parse_args()

    print("Loading text model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = build_index(model, SENTENCES)
    print(f"Indexed {len(SENTENCES)} sentences, vector size {index.shape[1]}\n")

    if args.query:
        print(f'Query: "{args.query}"')
        show(search(model, index, SENTENCES, args.query, args.k))
        return

    print('Type a query (for example: Good Morning). Press Enter on an empty line to quit.')
    while True:
        query = input("\nQuery> ").strip()
        if not query:
            break
        show(search(model, index, SENTENCES, query, args.k))


if __name__ == "__main__":
    main()