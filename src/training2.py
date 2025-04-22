import json
from pathlib import Path
from gensim.models import Word2Vec
from graphviz import Digraph
from PIL import Image


def load_sentences(path: Path):
    """Lädt eine Liste von Sätzen aus einer JSON-Datei."""
    return json.loads(path.read_text(encoding='utf-8'))


def train_model(sentences):
    """Trainiert ein Word2Vec-Modell auf den gegebenen Sätzen."""
    tokenized = [s.lower().split() for s in sentences]
    return Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)


def get_top_neighbors(model, word, topn=5):
    """Gibt die topn ähnlichsten Wörter zu word zurück."""
    try:
        return model.wv.most_similar(word.lower(), topn=topn)
    except KeyError:
        return []


def style_and_color(sim, src_id, tgt_id):
    """Ermittelt Stil, Farbe und Dicke einer Kante basierend auf Score und Knoten-IDs."""
    src_word = src_id.split('_', 1)[1]
    tgt_word = tgt_id.split('_', 1)[1]
    style = 'solid' if sim > 0.10 else 'dotted'
    color = 'black'
    penwidth = '1'
    pair = {src_word, tgt_word}
    if pair == {'king', 'queen'}:
        color = 'violet'; penwidth = '2'
    elif sim > 0.25:
        color = 'red'
    elif 'king' in pair:
        color = 'blue'; penwidth = '2'
    elif 'queen' in pair:
        color = 'deeppink'; penwidth = '2'
    return style, color, penwidth


def add_subgraph(g: Digraph, model: Word2Vec, targets, prefix: str):
    """Fügt dem Graphen g einen Cluster mit Knoten und Kanten hinzu."""
    neighbors = {t.lower(): [n for n, _ in get_top_neighbors(model, t)] for t in targets}
    with g.subgraph(name=f"cluster_{prefix}") as c:
        c.attr(label=prefix, style='filled', color='lightgrey')
        # Target-Knoten
        for t in targets:
            lower = t.lower(); node_id = f"{prefix}_{lower}"
            if lower == 'king':
                c.node(node_id, label=t, shape='circle', style='filled', fillcolor='lightblue')
            elif lower == 'queen':
                c.node(node_id, label=t, shape='circle', style='filled', fillcolor='lightpink')
            else:
                c.node(node_id, label=t, shape='circle')
        # Kanten zu Nachbarn
        for t in targets:
            src_id = f"{prefix}_{t.lower()}"
            for neigh, sim in get_top_neighbors(model, t):
                tgt_id = f"{prefix}_{neigh}"
                c.node(tgt_id, label=neigh, shape='circle')
                style, color, penwidth = style_and_color(sim, src_id, tgt_id)
                c.edge(src_id, tgt_id, label=f"{sim:.2f}", style=style,
                       color=color, penwidth=penwidth, dir='both')
        # Kanten unter Nachbarn
        all_neigh = set(sum(neighbors.values(), []))
        for n1 in sorted(all_neigh):
            for n2 in sorted(all_neigh):
                if n1 < n2:
                    id1 = f"{prefix}_{n1}"; id2 = f"{prefix}_{n2}"
                    sim = model.wv.similarity(n1, n2)
                    style, color, penwidth = style_and_color(sim, id1, id2)
                    c.edge(id1, id2, label=f"{sim:.2f}", style=style,
                           color=color, penwidth=penwidth, dir='both')


def main():
    base = Path('./data')
    files = {
        'austen': base / 'austen_sentences.json',
        'tolkien': base / 'tolkien_sentences.json',
        'got': base / 'got_sentences.json',
    }
    # Gemeinsamer Graph
    g = Digraph('combined', format='png')
    g.attr(rankdir='TB')  # Vertikale Anordnung

    # Einzelmodelle
    for prefix, path in files.items():
        sentences = load_sentences(path)
        model = train_model(sentences)
        add_subgraph(g, model, ['king', 'queen'], prefix)

    # Kombiniertes Modell
    combined_sentences = []
    for path in files.values():
        combined_sentences.extend(load_sentences(path))
    combined_model = train_model(combined_sentences)
    add_subgraph(g, combined_model, ['king', 'queen'], 'combined')

    # Unsichtbare Kanten zur vertikalen Ausrichtung
    order = list(files.keys()) + ['combined']
    for a, b in zip(order, order[1:]):
        g.edge(f"{a}_king", f"{b}_king", style='invis')

    # Rendern und Anzeigen
    out = g.render('combined_graph', cleanup=True)
    Image.open(out).show()

if __name__ == '__main__':
    main()
