import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_model(model_name: str):
    model = api.load(model_name)
    return model

def word_arithmetic(model, positive, negative):
    pos_vecs = [model[word] for word in positive]
    neg_vecs = [model[word] for word in negative]
    return np.sum(pos_vecs, axis=0) - np.sum(neg_vecs, axis=0)

def visualize_words(model, words, result_vector):
    vectors = np.array([model[word] for word in words])
    vectors = np.vstack([vectors, result_vector])
    labels = words + ["result"]
    pca = PCA(n_components=2, random_state=42)
    vectors_2d = pca.fit_transform(vectors)
    plt.figure(figsize=(8,6))
    plt.scatter(vectors_2d[:-1, 0], vectors_2d[:-1, 1], color="blue", s=100, alpha=0.8, marker="o")
    plt.scatter(vectors_2d[-1, 0], vectors_2d[-1, 1], color="red", s=120, alpha=1.0, marker="x")
    for i, label in enumerate(labels):
        plt.annotate(label, (vectors_2d[i, 0], vectors_2d[i, 1]), xytext=(5,-5), textcoords="offset points", fontsize=12)
    result_point = vectors_2d[-1]
    other_points = vectors_2d[:-1]
    distances = np.linalg.norm(other_points - result_point, axis=1)
    min_index = np.argmin(distances)
    closest_point = other_points[min_index]
    plt.plot([result_point[0], closest_point[0]],
             [result_point[1], closest_point[1]],
             linestyle="--", color="red", alpha=0.6, linewidth=1)
    plt.title("Word Arithmetic: Paris - France + Italy")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    model_name = "glove-wiki-gigaword-50"
    model = load_model(model_name)
    positive = ["paris", "italy"]
    negative = ["france"]
    result_vector = word_arithmetic(model, positive, negative)
    similar = model.most_similar(positive=positive, negative=negative, topn=5)
    print("Top similar words for 'Paris - France + Italy':")
    for word, score in similar:
        print(f"{word}: {score:.4f}")
    words_to_plot = ["paris", "france", "italy", "rome", "spain", "germany", "turin", "venice"]
    visualize_words(model, words_to_plot, result_vector)

if __name__ == '__main__':
    main()
