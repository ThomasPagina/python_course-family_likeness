import gensim.downloader as api
import numpy as np
from graphviz import Digraph

def load_model(model_name: str):
    print(f"Loading model '{model_name}'...")
    model = api.load(model_name)
    print("Model loaded successfully.")
    return model

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def compute_predicted_vector(model, target, relation_vector):
    return model[target.lower()] + relation_vector

def assign_hypernym_for_word(model, predicted_vector, candidate_list):
    best_candidate = None
    best_similarity = -1
    for cand in candidate_list:
        cand_vec = model[cand.lower()]
        sim = cosine_similarity(predicted_vector, cand_vec)
        if sim > best_similarity:
            best_similarity = sim
            best_candidate = cand
    return best_candidate, best_similarity

def assign_hypernyms(model, target_words, relation_vector, candidate_list):
    assignments = {cand: [] for cand in candidate_list}
    predictions = {}
    for word in target_words:
        pred_vec = compute_predicted_vector(model, word, relation_vector)
        predictions[word] = pred_vec
        best_cand, _ = assign_hypernym_for_word(model, pred_vec, candidate_list)
        assignments[best_cand].append(word)
    return assignments, predictions

def visualize_assignments_graph(assignments):
    dot = Digraph(comment='Hypernym Assignment')
    for hypernym, hyponyms in assignments.items():
        dot.node(hypernym, hypernym, shape='box', style='filled', fillcolor='lightgrey')
        for word in hyponyms:
            dot.node(word, word, shape='ellipse', style='filled', fillcolor='white')
            dot.edge(hypernym, word)
    dot.render('hypernym_assignments.gv', view=True)

def main():
    model = load_model("glove-wiki-gigaword-50")
    candidate_list = ["instrument", "animal", "flower", "country", "city"]
    target_words = ["violin", "guitar", "piano", "cat", "dog", "elephant", "india", "labrador", "tulip", "daisy", "raven", "sparrow", "lily", "paris", "london", "milan", "germany", "italy", "france"]
    relation_vector = model["dog"] - model["poodle"]
    assignments, predictions = assign_hypernyms(model, target_words, relation_vector, candidate_list)
    print("Hypernym assignments:")
    for cand, words in assignments.items():
        print(f"{cand}: {words}")
    visualize_assignments_graph(assignments)

if __name__ == '__main__':
    main()
