import gensim.downloader as api
import networkx as nx
import matplotlib.pyplot as plt
import spacy
from sklearn.decomposition import PCA
import numpy as np
import warnings

# ----------------------------------------------------------------------------
# Configuration
def get_config():
    return {
        "w2v_model_name": "fasttext-wiki-news-subwords-300",
        "spacy_model_name": "en_core_web_sm",
        "input_words": [
            "doctor", "nurse", "engineer", "teacher","housewife","shopkeeper","professor","waitress","queen"
        ],
        "num_similar": 12,
        "color_female": "lightgreen",
        "color_male": "yellow",
        "color_same_pos": "skyblue",
        "color_diff_pos": "lightcoral",
        "neutral_threshold": 0.025
    }

# ----------------------------------------------------------------------------
# Utility functions

def configure_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", message=".*The parameter 'token_pattern'.*")

def load_models(w2v_name, spacy_name):
    print(f"Loading embeddings model '{w2v_name}'...")
    w2v = api.load(w2v_name)
    print("Loading spaCy model...")
    nlp = spacy.load(spacy_name, disable=["parser", "ner"])
    return w2v, nlp

# Compute normalized gender direction (woman - man)
def get_gender_direction(model):
    vec_w = model.get_vector("woman")
    vec_m = model.get_vector("man")
    d = vec_w - vec_m
    return d / np.linalg.norm(d)

# POS tag a single token
def get_pos(nlp, token):
    doc = nlp(token)
    return doc[0].pos_ if doc else "UNKNOWN"

# Projection of vector on gender axis
def gender_projection(vec, gender_dir):
    return float(np.dot(vec, gender_dir))

# Find counterpart: invert gender and choose same-POS, opposite-projection word
def find_counterpart(model, gender_dir, nlp, input_word, input_vec, input_proj, config):
    target_vec = input_vec - 2 * input_proj * gender_dir
    try:
        sims = model.similar_by_vector(target_vec, topn=config['num_similar'] * 3)
    except KeyError:
        return None
    pos_input = get_pos(nlp, input_word)
    for candidate, _ in sims:
        cand = candidate.lower()
        if cand == input_word:
            continue
        if get_pos(nlp, cand) != pos_input:
            continue
        vec_c = model.get_vector(cand)
        proj_c = gender_projection(vec_c, gender_dir)
        if proj_c * input_proj < 0 and abs(proj_c) > config['neutral_threshold']:
            return cand
    return None

# Build and plot a network of central word + neighbors
def build_and_plot(ax, model, nlp, central, config, gender_dir):
    word = central.lower()
    vec = model.get_vector(word)
    proj = gender_projection(vec, gender_dir)
    is_female = proj > config['neutral_threshold']
    is_male = proj < -config['neutral_threshold']
    if not (is_female or is_male):
        return

    try:
        neighbors = [w for w, _ in model.most_similar(word, topn=config['num_similar'])]
    except KeyError:
        neighbors = []
    nodes = [word] + neighbors

    data = {}
    pos_input = get_pos(nlp, word)
    for w in nodes:
        w_l = w.lower()
        try:
            v = model.get_vector(w_l)
        except KeyError:
            continue
        p = get_pos(nlp, w_l)
        data[w_l] = {'vec': v, 'pos': p}

    keys = list(data.keys())
    vecs = np.array([data[k]['vec'] for k in keys])
    coords = PCA(n_components=2, random_state=42).fit_transform(vecs)

    G = nx.Graph()
    color_map = []
    positions = {}
    labels = {}
    for i, k in enumerate(keys):
        G.add_node(k)
        positions[k] = coords[i]
        labels[k] = k
        if k == word:
            color_map.append(config['color_female'] if is_female else config['color_male'])
        else:
            same = (data[k]['pos'] == pos_input)
            color_map.append(config['color_same_pos'] if same else config['color_diff_pos'])
    for k in keys:
        if k != word:
            G.add_edge(word, k)

    nx.draw_networkx_nodes(G, positions, ax=ax, node_color=color_map, node_size=800, alpha=0.9)
    nx.draw_networkx_edges(G, positions, ax=ax, alpha=0.3)
    nx.draw_networkx_labels(G, positions, labels, ax=ax, font_size=8)
    ax.axis('off')

# Main execution
if __name__ == '__main__':
    configure_warnings()
    config = get_config()
    w2v, nlp = load_models(config['w2v_model_name'], config['spacy_model_name'])
    gender_dir = get_gender_direction(w2v)

    for word in config['input_words']:
        try:
            input_vec = w2v.get_vector(word)
        except KeyError:
            print(f"'{word}' not in vocabulary, skipping.")
            continue
        input_proj = gender_projection(input_vec, gender_dir)
        is_female = input_proj > config['neutral_threshold']
        is_male = input_proj < -config['neutral_threshold']
        if not (is_female or is_male):
            print(f"Skipping '{word}': neutral (proj={input_proj:.3f}).")
            continue
        counterpart = find_counterpart(w2v, gender_dir, nlp, word, input_vec, input_proj, config)
        if not counterpart:
            # single plot with no counterpart
            fig, ax = plt.subplots(figsize=(7, 6))
            build_and_plot(ax, w2v, nlp, word, config, gender_dir)
            symbol = '♀' if is_female else '♂'
            ax.set_title(f"{word}: {input_proj:+.3f} {symbol} — no counterpart available")
            plt.tight_layout()
            plt.show()
            continue

        # with counterpart
        vec_c = w2v.get_vector(counterpart)
        proj_c = gender_projection(vec_c, gender_dir)
        is_female_c = proj_c > config['neutral_threshold']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        build_and_plot(axes[0], w2v, nlp, word, config, gender_dir)
        symbol = '♀' if is_female else '♂'
        axes[0].set_title(f"{word}: {input_proj:+.3f} {symbol}")

        build_and_plot(axes[1], w2v, nlp, counterpart, config, gender_dir)
        symbol_c = '♀' if is_female_c else '♂'
        axes[1].set_title(f"{counterpart}: {proj_c:+.3f} {symbol_c}")

        plt.tight_layout()
        plt.show()
