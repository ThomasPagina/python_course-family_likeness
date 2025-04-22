import gensim.downloader as api
import networkx as nx
import matplotlib.pyplot as plt
import spacy
from sklearn.decomposition import PCA
import numpy as np
import warnings

def configure_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", message=".*The parameter 'token_pattern'.*")

def get_config():
    config = {
        "word2vec_model_name": "fasttext-wiki-news-subwords-300",
        "spacy_model_name": "en_core_web_sm",
        "input_words_orig": ["Apple", "green", "doctor", "cook", "beautiful"],
        "num_similar": 15,
        "color_input_word": "lightgreen",
        "color_same_pos": "skyblue",
        "color_diff_pos": "lightcoral"
    }
    return config

def load_word2vec_model(model_name: str):
    print(f"Loading Word2Vec model '{model_name}'...")
    try:
        model = api.load(model_name)
        print("Word2Vec model loaded successfully.")
        return model
    except ValueError as e:
        print(f"Error loading Word2Vec model: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error loading Word2Vec model: {e}")
        raise

def load_spacy_model(model_name: str):
    print(f"Loading spaCy model '{model_name}'...")
    try:
        model = spacy.load(model_name, disable=['parser','ner'])
        print("spaCy model loaded successfully.")
        return model
    except OSError:
        msg = f"Error: spaCy model '{model_name}' not found.\nPlease download it using: python -m spacy download {model_name}"
        print(msg)
        raise
    except Exception as e:
        print(f"Unexpected error loading spaCy model: {e}")
        raise

def get_word_pos(nlp, word: str) -> str:
    doc = nlp(word)
    return doc[0].pos_ if len(doc) > 0 else "UNKNOWN"

def process_word(word: str, w2v_model, spacy_nlp, input_pos: str) -> dict:
    cleaned_word = word.lower().strip()
    if not cleaned_word:
        return None
    try:
        vector = w2v_model[cleaned_word]
        pos_tag = get_word_pos(spacy_nlp, cleaned_word)
        return {
            'cleaned_word': cleaned_word,
            'vector': vector,
            'pos': pos_tag,
            'is_similar_pos': (pos_tag == input_pos) if input_pos != "UNKNOWN" and pos_tag != "UNKNOWN" else False,
            'original_case': word
        }
    except KeyError:
        print(f"Warning: Word '{cleaned_word}' not found in Word2Vec model and will be skipped.")
        return None
    except Exception as e:
        print(f"Error processing word '{cleaned_word}': {e}")
        return None

def get_similar_words(w2v_model, word: str, num_similar: int):
    try:
        similar_words_raw = w2v_model.most_similar(word, topn=num_similar)
        similar_words = [w for w, score in similar_words_raw]
        return similar_words
    except KeyError:
        msg = f"Error: The word '{word}' is not in the Word2Vec model even though it was previously verified."
        print(msg)
        raise

def gather_word_data(w2v_model, spacy_nlp, input_word: str, similar_words: list):
    input_pos = get_word_pos(spacy_nlp, input_word)
    all_words = [input_word] + similar_words
    word_data = {}
    valid_words = []
    vectors = []
    for word in all_words:
        data = process_word(word, w2v_model, spacy_nlp, input_pos)
        if data:
            key = data['cleaned_word']
            word_data[key] = data
            valid_words.append(key)
            vectors.append(data['vector'])
    if not valid_words or input_word not in valid_words:
        raise ValueError("Not enough valid words (including the input word) for graph creation.")
    return word_data, valid_words, np.array(vectors), input_pos

def reduce_dimensions(vectors, n_components: int = 2):
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(vectors)

def create_graph(valid_words, word_data, vectors_2d, input_word, color_input_word, color_same_pos, color_diff_pos):
    G = nx.Graph()
    pos = {}
    node_colors = []
    node_labels = {}
    for i, word in enumerate(valid_words):
        G.add_node(word)
        pos[word] = vectors_2d[i]
        node_labels[word] = word_data[word]['original_case']
        if word == input_word:
            node_colors.append(color_input_word)
        elif word_data[word]['is_similar_pos']:
            node_colors.append(color_same_pos)
        else:
            node_colors.append(color_diff_pos)
    for word in valid_words:
        if word != input_word:
            G.add_edge(input_word, word)
    return G, pos, node_colors, node_labels

def create_legend_handles(input_word_orig, input_pos, color_input_word, color_same_pos, color_diff_pos):
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=f"Input: '{input_word_orig}' ({input_pos})", markerfacecolor=color_input_word, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label=f"Similar (same POS: {input_pos})", markerfacecolor=color_same_pos, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label="Similar (different POS)", markerfacecolor=color_diff_pos, markersize=10)
    ]
    return legend_handles

def plot_graph(G, pos, node_colors, node_labels, input_word_orig, input_pos, model_name, color_input_word, color_same_pos, color_diff_pos):
    plt.figure(figsize=(16, 12))
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.3, edge_color='gray')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_family='sans-serif')
    plt.title(f"Semantic Network for '{input_word_orig}' (POS: {input_pos}) - Model: {model_name}\nColors indicate whether the part-of-speech matches", fontsize=14)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    legend_handles = create_legend_handles(input_word_orig, input_pos, color_input_word, color_same_pos, color_diff_pos)
    plt.legend(handles=legend_handles, title="Part-of-Speech", loc='best')
    plt.tight_layout()
    plt.show()

def main():
    configure_warnings()
    config = get_config()
    word2vec_model_name = config["word2vec_model_name"]
    spacy_model_name = config["spacy_model_name"]
    input_words_orig = config.get("input_words_orig", ["Apple"])
    num_similar = config["num_similar"]
    color_input_word = config["color_input_word"]
    color_same_pos = config["color_same_pos"]
    color_diff_pos = config["color_diff_pos"]
    print("Loading models...")
    w2v_model = load_word2vec_model(word2vec_model_name)
    spacy_nlp = load_spacy_model(spacy_model_name)
    for input_word_orig in input_words_orig:
        input_word = input_word_orig.lower()
        if input_word not in w2v_model:
            print(f"ERROR: The word '{input_word_orig}' (as '{input_word}') was not found in the Word2Vec model '{word2vec_model_name}'.")
            continue
        print(f"\nDetermining part-of-speech for '{input_word_orig}'...")
        input_pos = get_word_pos(spacy_nlp, input_word)
        if input_pos != "UNKNOWN":
            print(f"'{input_word_orig}' recognized as '{input_pos}'.")
        else:
            print(f"Warning: Could not uniquely determine the part-of-speech for '{input_word_orig}'.")
        print(f"\nFetching top {num_similar} similar words to '{input_word}'...")
        similar_words = get_similar_words(w2v_model, input_word, num_similar)
        print(f"Similar words found: {similar_words}")
        print("\nCollecting vectors and determining POS tags for all relevant words...")
        try:
            word_data, valid_words, word_vectors, _ = gather_word_data(w2v_model, spacy_nlp, input_word, similar_words)
        except ValueError as ve:
            print(f"ERROR: {ve}")
            continue
        print("\nReducing dimensions with PCA for 2D visualization...")
        vectors_2d = reduce_dimensions(word_vectors, n_components=2)
        print("Constructing the graph...")
        G, pos, node_colors, node_labels = create_graph(valid_words, word_data, vectors_2d, input_word, color_input_word, color_same_pos, color_diff_pos)
        print(f"Plotting the graph for '{input_word_orig}'...")
        plot_graph(G, pos, node_colors, node_labels, input_word_orig, input_pos, word2vec_model_name, color_input_word, color_same_pos, color_diff_pos)
    print("\nProcessing complete for all input words.")
    print("Note: The positions in the 2D graph are a PCA approximation of semantic proximity.")

if __name__ == '__main__':
    main()
