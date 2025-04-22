import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import textwrap

def get_corpus():
    """
    Returns a sample corpus as a list of document strings.
    """
    return [
        "The Sun is the central star of our solar system. It provides essential energy and light that is crucial for life on Earth. The Sun is a massive sphere of plasma, and its nuclear fusion in the core produces the energy we receive as sunlight. Its immense gravity holds the planets together and influences the entire solar system.",
        "Cats are popular pets known for their independence and agility. They are excellent hunters and have been domesticated for thousands of years. Cats provide companionship while also managing pests in many households. Their playful behavior and graceful movements make them charming animals.",
        "Python is a widely used programming language that is celebrated for its readability, simplicity, and versatility. Developers use Python for web development, data analysis, artificial intelligence, and many other applications. Python has an extensive standard library and an active community, which makes it a preferred language for both beginners and experts.",
        "Apples and bananas are nutritious fruits that offer a wealth of vitamins, fiber, and antioxidants. They are commonly eaten as snacks or incorporated into meals and desserts. Regular consumption of these fruits contributes to a balanced diet and helps maintain overall health.",
        "The Moon orbits Earth as its natural satellite and plays a significant role in influencing tides. It also helps stabilize the rotation of the Earth. Its changing phases have fascinated humans for centuries and have impacted various aspects of culture and science.",
        "Dogs are known for their loyalty and companionship toward humans. They come in many breeds and sizes, and each breed has unique traits and temperaments. Dogs serve as pets, helpers, and even working partners in many roles such as guide dogs and service animals.",
        "Many websites use JavaScript to create interactive user experiences and dynamic content. JavaScript is a core technology of the web and allows developers to build complex applications. It runs in all modern browsers and is essential for front-end development.",
        "Vegetables like broccoli and carrots are rich in vitamins and minerals. They are an essential part of a healthy diet because they provide nutrients that support body functions and overall wellbeing. Including a variety of vegetables in daily meals is recommended by nutrition experts.",
        "Planets such as Mars and Jupiter are key members of our solar system. Mars is recognized for its red appearance and evidence of past water activity, while Jupiter is noted for its massive size and distinctive atmospheric features. Studying these planets helps scientists understand the evolution of the solar system.",
        "Fish inhabit aquatic environments and are adapted to life underwater with features like gills for breathing. They form a diverse group and play critical roles in marine ecosystems. Fish provide important food resources and contribute significantly to global biodiversity.",
        "Artificial intelligence is a broad field within computer science that focuses on creating systems capable of performing tasks that usually require human intelligence. AI is used in applications ranging from natural language processing to autonomous vehicles and predictive analytics.",
        "A balanced diet typically includes proteins, carbohydrates, fats, vitamins, and minerals. Such a diet supports energy levels, promotes healthy body functions, and helps prevent chronic illnesses. Nutritional balance is essential for overall wellbeing and long-term health."
    ]

def preprocess_corpus(corpus):
    tagged_docs = []
    for idx, doc in enumerate(corpus):
        tokens = simple_preprocess(doc)
        tagged_docs.append(TaggedDocument(words=tokens, tags=[str(idx)]))
    return tagged_docs

def train_doc2vec(tagged_docs, vector_size=50, window=5, min_count=1, epochs=100, dm=1):
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count,
                    epochs=epochs, dm=dm, workers=4)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def reduce_dimensions_tsne(vectors, n_components=2, random_state=42):
    n_samples = vectors.shape[0]
    perplexity = max(5, min(30, n_samples // 2))
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=n_components, random_state=random_state,
                perplexity=perplexity, init='pca', learning_rate='auto')
    return tsne.fit_transform(vectors)


def plot_documents(vectors_2d, corpus, doc_ids):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=80, alpha=0.7, edgecolors='w')
    for i, doc_id in enumerate(doc_ids):
        ax.annotate(doc_id, (vectors_2d[i, 0], vectors_2d[i, 1]),
                    xytext=(5, -5), textcoords='offset points',
                    fontsize=9, ha='left', va='top')
    ax.set_title("Doc2Vec Document Embeddings (visualized with t-SNE)", fontsize=16)
    ax.set_xlabel("t-SNE Component 1", fontsize=12)
    ax.set_ylabel("t-SNE Component 2", fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def print_legend(corpus):
    print("\n--- Document Legend ---")
    wrapper = textwrap.TextWrapper(width=70)
    for idx, doc in enumerate(corpus):
        print(f"{idx}: {wrapper.fill(doc)}")
    print("-" * 40)

def main():
    corpus = get_corpus()
    tagged_docs = preprocess_corpus(corpus)
    model = train_doc2vec(tagged_docs, epochs=100)
    doc_ids = model.dv.index_to_key
    doc_vectors = model.dv.vectors
    vectors_2d = reduce_dimensions_tsne(doc_vectors)
    plot_documents(vectors_2d, corpus, doc_ids)
    print_legend(corpus)

if __name__ == '__main__':
    main()
