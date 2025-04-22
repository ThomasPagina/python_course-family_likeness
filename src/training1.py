import re
import numpy as np
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import pandas as pd

# ----------------------
# Tokenization
# ----------------------
def simple_tokenize(text: str) -> list:
    """Remove punctuation and split on whitespace."""
    clean = re.sub(r"[^\w\s]", "", text.lower())
    return clean.split()

# ----------------------
# Embedding
# ----------------------
def train_word2vec(sentences: list, window: int = 5, vector_size: int = 100) -> Word2Vec:
    """Train a Word2Vec model on tokenized sentences."""
    tokenized = [simple_tokenize(s) for s in sentences]
    return Word2Vec(sentences=tokenized, vector_size=vector_size, window=window, min_count=1, workers=2)

# ----------------------
# Sentence Vector
# ----------------------
def sentence_vector(sentence: str, model: Word2Vec) -> np.ndarray:
    """Compute sentence vector by averaging word vectors."""
    tokens = simple_tokenize(sentence)
    vecs = [model.wv[token] for token in tokens if token in model.wv]
    if not vecs:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

# ----------------------
# Evaluation
# ----------------------
def evaluate_model(model: Word2Vec, reference: str, test_sentences: list, test_target: str) -> tuple:
    """Return (rank, score) of test_target within test_sentences by cosine similarity."""
    ref_vec = sentence_vector(reference, model)
    sims = [(s, 1 - cosine(ref_vec, sentence_vector(s, model))) for s in test_sentences]
    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
    sentences_sorted = [s for s, _ in sims_sorted]
    rank = sentences_sorted.index(test_target) + 1
    score = dict(sims_sorted)[test_target]
    return rank, score

# ----------------------
# Strong Model Scoring
# ----------------------
def compute_strength_scores(sentences: list, reference: str, strong_model) -> dict:
    """Compute similarity scores of sentences to reference using strong_model."""
    embeddings = strong_model.encode([reference] + sentences)
    ref_emb = embeddings[0]
    sent_embs = embeddings[1:]
    return {sent: np.dot(ref_emb, emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb))
            for sent, emb in zip(sentences, sent_embs)}

# ----------------------
# Boosting Strategy
# ----------------------
def apply_boosting_strong(sentences: list, reference: str, strong_model, factor: float = 1.0) -> list:
    """Boost sentences above mean by number proportional to (score - mean)/std * factor."""
    scores = compute_strength_scores(sentences, reference, strong_model)
    values = np.array(list(scores.values()))
    mean, std = values.mean(), values.std()
    boosted = sentences.copy()
    for sent, score in scores.items():
        if score > mean:
            times = int(((score - mean) / std) * factor)
            boosted.extend([sent] * times)
    return boosted

# ----------------------
# Reduction Strategy
# ----------------------
def apply_reduction_strong(sentences: list, reference: str, strong_model, threshold: float = None) -> list:
    """Remove sentences with scores below (mean - std) or given threshold."""
    scores = compute_strength_scores(sentences, reference, strong_model)
    values = np.array(list(scores.values()))
    mean, std = values.mean(), values.std()
    cutoff = threshold if threshold is not None else mean - std
    return [s for s in sentences if scores[s] >= cutoff]

# ----------------------
# Doubling Strategy
# ----------------------
def apply_doubling(sentences: list, target_word: str, alt_word: str) -> list:
    """Duplicate sentences replacing target_word with alt_word if not present."""
    new = sentences.copy()
    for sent in sentences:
        if target_word in simple_tokenize(sent):
            replaced = re.sub(rf"\b{target_word}\b", alt_word, sent)
            if replaced not in new:
                new.append(replaced)
    return new

# ----------------------
# Phrase Swapping Strategy
# ----------------------
def apply_phrase_swapping(sentences: list, phrase_map: dict) -> list:
    """Swap key phrases according to phrase_map, adding new variants if missing."""
    new = sentences.copy()
    for sent in sentences:
        for src, tgt in phrase_map.items():
            if src in sent:
                replaced = sent.replace(src, tgt)
                if replaced not in new:
                    new.append(replaced)
    return new

# ----------------------
# Experiment Runner
# ----------------------
def run_experiments(reference: str,
                    test_sentences: list,
                    test_target: str,
                    base_sets: dict,
                    strong_model_name: str = 'all-MiniLM-L6-v2') -> pd.DataFrame:
    """Execute experiments with fixed window=5 and return DataFrame of results."""
    strong_model = SentenceTransformer(strong_model_name)
    results = []
    strategies = ['none', 'boosted', 'reduced', 'doubled', 'doubledoubled']
    window = 5

    phrase_map = {'in the garden': 'outside', 'outside': 'in the garden'}

    for base_name, base_data in base_sets.items():
        for strategy in strategies:
            data = base_data.copy()
            if strategy == 'boosted':
                data = apply_boosting_strong(data, reference, strong_model)
            elif strategy == 'reduced':
                data = apply_reduction_strong(data, reference, strong_model)
            elif strategy == 'doubled':
                data = apply_doubling(data, 'child', 'boy')
            elif strategy == 'doubledoubled':
                data = apply_doubling(data, 'child', 'boy')
                data = apply_doubling(data, 'boy', 'child')
                data = apply_phrase_swapping(data, phrase_map)

            model = train_word2vec(data, window=window)
            rank, score = evaluate_model(model, reference, test_sentences, test_target)
            results.append({'label': f"{base_name}-{strategy}-w{window}", 'rank': rank, 'score': score, 'datasize': len(data)})

    return pd.DataFrame(results)

# ----------------------
# Main
# ----------------------
def main():
    short_train = [
        "The happy child eats an apple.",
        "The dog runs through the garden.",
        "All day the cat plays happily with the ball.",
        "The child jumps up and down.",
        "The boy sits on the grass.",
        "A man sits outside reading a book.",
        "A woman smiles at the camera.",
        "The girl hits the boy.",
        "On sunday the boy goes to church.",  
        "The father scolds the child.",
        "The young woman reads outside.",
        "The young woman reads in the garden.",
        "A child sings outside.",
        "The big child is in the garden.",
        "Susan plays the guitar at school.",
        "A boy is playing with his sister.",
     

    ]
    long1_train = [
        "A woman reads a book.",
        "A car drives fast.",
        "Children play happily in the park.",
        "The teacher explains the task.",
        "The cat sleeps on the sofa.",
        "The children jump rope together.",
        "A boy chases a butterfly.",
        "The girl plays with her doll.",
        "Kids build a treehouse.",
        "A girl waters the flowers.",
        "Siblings ride bikes down the street.",
        "A child kicks a ball.",
        "The child draws a picture.",
        "The teacher calls the child.",
        "A child climbs a tree.",
        "A child listens to music.",
        "The child plays with dolls.",
        "A child plays outside in the garden.",
        "The cat is playing outside.",
        "The child is playing all day.",
        "The child is playing with her dolls.",
        "The girl is playing with her toys."
        "The big girl still plays with dolls.",
        "The young boy plays a game.",
        "A child sits outside under the tree.",
        "The children gather in the garden.",
        "The mother reads a book to the child.",
        "A child waters plants in the garden.",
        "A child reads outside on the porch.",
        "The child is laughing at a joke.",
        "A child is laughing heartily.",
        "A child runs in the garden.",
        "A child is playing a game on the computer.",
        "The child plays with a toy car.",
        "The mother sees her child in the garden.",
        "The small child sits in the sun.",
        "The sun shines on the little child.",

    ]
    long2_train = [
        "A man jogs through the forest.",
        "The bird sings in the tree.",
        "The sun shines brightly today.",
        "Kids are running around the yard.",
        "The boy builds a sandcastle.",
        "A child splashes in a puddle.",
        "A family picnics by the lake.",
        "The toddler plays with building blocks.",
        "A group of children fly kites.",
        "A young girl sings a song.",
        "Parents push swings at the playground.",
        "A child plays outside by the fence.",
        "A boy jumps in the garden.",
        "The kids run outside to the garden.",
        "A child sits outside reading a book.",
        "A boy waters flowers in the garden.",
        "A child plays soccer in the park.",
        "The child laughs at a funny joke.",
        "A little boy helps his friend with homework.",
        "The child draws a picture of a house.",
        "A girl plays with her puppy.",
        "The boy is playing with his toy car.",
        "The child is playing in the kitchen.",
        "The mother is playing with the child.",
        "The girl hits the boy with a ball.",
        "The man hits the child with a stick."
        "The man sings a song to the little child.",
        "The little boy has fun with his toy car.",
        "The church is open only on sundays.",
        "My sister has a little boy.",
        "My mother only has one child.",
        "A small child runs in the garden.",
        "Susan has only one child.",
        "Susan has a boy.",
        "Susan plays with the child in the garden.",
        "The little girl plays with the boy.",
        "The girl plays outside with the other children."
        "In the garden I can only see one child.",
    ]
    reference_sentence = "The child plays outside."
    test_sentences = [
        "A boy plays in the garden.",
        "The woman is reading outside.",
        "The dog barks loudly.",
        "Children are eating lunch.",
        "The cat lies in the sun."
    ]
    test_target = "A boy plays in the garden."

    base_sets = {
        'short': short_train,
        'short+long1': short_train + long1_train,
        'short+long2': short_train + long2_train
    }

    df_results = run_experiments(
        reference_sentence,
        test_sentences,
        test_target,
        base_sets
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # Rank plot
    axes[0].bar(df_results['label'], df_results['rank'], color='gray')
    axes[0].set_ylabel('Rank (lower is better)')
    axes[0].set_title('Ranking of Target Test Sentence')

    # Similarity score plot
    axes[1].bar(df_results['label'], df_results['score'], color='blue')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Similarity Score with Reference')

    # Datasize distribution plot
    axes[2].bar(df_results['label'], df_results['datasize'], color='green')
    axes[2].set_ylabel('Datasize')
    axes[2].set_title('Datasize Distribution of Test Sentences')

    # Shared x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
