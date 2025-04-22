import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_model(model_name: str):
    print(f"Loading model '{model_name}'...")
    model = api.load(model_name)
    print("Model loaded successfully.")
    return model

def find_best_hyperonym(model, target, candidates):
    target_vec = model[target.lower()]
    similarities = {}
    for cand in candidates:
        cand_vec = model[cand.lower()]
        sim = np.dot(target_vec, cand_vec) / (np.linalg.norm(target_vec) * np.linalg.norm(cand_vec))
        similarities[cand] = sim
    best_candidate = max(similarities, key=similarities.get)
    return best_candidate, similarities

def visualize_hyperonym(model, target, candidates, best_candidate):
    words = [target.lower()] + [cand.lower() for cand in candidates]
    vectors = np.array([model[word] for word in words])
    pca = PCA(n_components=2, random_state=42)
    vectors_2d = pca.fit_transform(vectors)
    plt.figure(figsize=(8,6))
    plt.scatter(vectors_2d[1:, 0], vectors_2d[1:, 1], color="blue", s=100, alpha=0.8, marker="o")
    plt.scatter(vectors_2d[0, 0], vectors_2d[0, 1], color="red", s=120, marker="*", alpha=1.0)
    labels = [target] + candidates
    for i, label in enumerate(labels):
        plt.annotate(label, (vectors_2d[i, 0], vectors_2d[i, 1]), xytext=(5, -5), textcoords="offset points", fontsize=12)
    best_index = candidates.index(best_candidate) + 1
    target_point = vectors_2d[0]
    best_point = vectors_2d[best_index]
    plt.plot([target_point[0], best_point[0]], [target_point[1], best_point[1]], linestyle="--", color="green", alpha=0.6, linewidth=1)
    plt.title(f"Hypernym Demonstration for '{target.capitalize()}'")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    model_name = "glove-wiki-gigaword-50"
    model = load_model(model_name)
    target1 = "apple"
    candidates1 = ["fruit", "device", "company", "tree", "food"]
    best1, sims1 = find_best_hyperonym(model, target1, candidates1)
    print(f"For '{target1}', the best hypernym candidate is: {best1}")
    print("Similarity scores:", sims1)
    visualize_hyperonym(model, target1, candidates1, best1)
    target2 = "raven"
    candidates2 = ["bird", "mammal", "insect", "animal", "plant"]
    best2, sims2 = find_best_hyperonym(model, target2, candidates2)
    print(f"For '{target2}', the best hypernym candidate is: {best2}")
    print("Similarity scores:", sims2)
    visualize_hyperonym(model, target2, candidates2, best2)
    target3 = "lily"
    candidates3 = ["flower", "plant", "color", "fragrance", "fruit"]
    best3, sims3 = find_best_hyperonym(model, target3, candidates3)
    print(f"For '{target3}', the best hypernym candidate is: {best3}")
    print("Similarity scores:", sims3)
    visualize_hyperonym(model, target3, candidates3, best3)

if __name__ == '__main__':
    main()
