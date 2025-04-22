import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_model(model_name: str):
    print(f"Loading model '{model_name}'...")
    model = api.load(model_name)
    print("Model loaded successfully.")
    return model

def compute_relation(model, word_a, word_b, word_c):
    return model[word_c.lower()] + (model[word_b.lower()] - model[word_a.lower()])

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def select_best_candidate(model, result_vector, candidate_list):
    similarities = {}
    for cand in candidate_list:
        cand_vec = model[cand.lower()]
        sim = cosine_similarity(result_vector, cand_vec)
        similarities[cand] = sim
    best_candidate = max(similarities, key=similarities.get)
    return best_candidate, similarities

def visualize_candidates(model, candidate_list, result_vector, best_candidate):
    candidate_vectors = np.array([model[cand.lower()] for cand in candidate_list])
    vectors = np.vstack([candidate_vectors, result_vector])
    labels = candidate_list + ["result"]
    pca = PCA(n_components=2, random_state=42)
    vectors_2d = pca.fit_transform(vectors)
    plt.figure(figsize=(8,6))
    plt.scatter(vectors_2d[:-1, 0], vectors_2d[:-1, 1], color="blue", s=100, alpha=0.8, marker="o")
    plt.scatter(vectors_2d[-1, 0], vectors_2d[-1, 1], color="red", s=120, alpha=1.0, marker="x")
    for i, label in enumerate(labels):
        plt.annotate(label, (vectors_2d[i, 0], vectors_2d[i, 1]), xytext=(5, -5), textcoords="offset points", fontsize=12)
    best_index = candidate_list.index(best_candidate)
    result_point = vectors_2d[-1]
    best_point = vectors_2d[best_index]
    plt.plot([result_point[0], best_point[0]], [result_point[1], best_point[1]], linestyle="--", color="green", alpha=0.6, linewidth=1)
    plt.title("Hypernym Candidate Selection via Word Arithmetic")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    model_name = "glove-wiki-gigaword-50"
    model = load_model(model_name)
    result_vector = compute_relation(model, "apple", "fruit", "tulip")
    similar_global = model.most_similar(positive=["tulip", "fruit"], negative=["apple"], topn=5)
    print("Top similar words (global) for 'tulip + (fruit - apple)':")
    for word, score in similar_global:
        print(f"{word}: {score:.4f}")
    candidate_list = ["flower", "fruit", "company", "character"]
    best_candidate, candidate_sims = select_best_candidate(model, result_vector, candidate_list)
    print("\nCandidate hypernym scores:")
    for cand, score in candidate_sims.items():
        print(f"{cand}: {score:.4f}")
    print(f"\nBest candidate: {best_candidate}")
    visualize_candidates(model, candidate_list, result_vector, best_candidate)

if __name__ == '__main__':
    main()
