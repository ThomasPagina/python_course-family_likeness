from sentence_transformers import SentenceTransformer
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import textwrap

def load_models(embedding_model_name: str, spacy_model_name: str):
    try:
        embedding_model = SentenceTransformer(embedding_model_name)
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        exit()
    try:
        spacy_nlp = spacy.load(spacy_model_name)
    except OSError:
        print(f"Error: spaCy model '{spacy_model_name}' not found.")
        exit()
    except Exception as e:
        print(f"Unexpected error loading spaCy model: {e}")
        exit()
    return embedding_model, spacy_nlp

def split_text(text: str, spacy_nlp) -> list:
    doc = spacy_nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences

def closest_results(query: str, text: str, n: int, embedding_model, spacy_nlp):
    sentences = split_text(text, spacy_nlp)
    sentence_embeddings = embedding_model.encode(sentences)
    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
    indexed_scores = list(enumerate(similarities))
    sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    return sorted_scores[:n], sentences

def main():
    embedding_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    spacy_model_name = 'de_core_news_sm'
    top_n = 3
    fact_text = """
    The Sun is the star at the center of our solar system. It is an almost perfect spherical ball of plasma,
    primarily composed of hydrogen (approximately 73%) and helium (approximately 25%). In the Sun's core,
    nuclear fusion occurs, where hydrogen is fused into helium. This fusion releases enormous amounts of energy,
    mainly as light and heat, which makes life on Earth possible. The surface temperature of the Sun is about
    5,500°C, while the core temperature is estimated at around 15 million°C. The Sun has a diameter of about 1.39
    million kilometers, roughly 109 times that of Earth. Its mass constitutes about 99.86% of the total mass of the
    solar system. Sunlight takes about 8 minutes and 20 seconds to reach Earth. Without the Sun, there would be
    no liquid water and likely no life on our planet. Solar activity, including sunspots and solar flares, varies
    over an approximately 11-year cycle.
    """
    easy_language_facts = """
The sun is the star at the center of our solar system.
The sun is an almost perfect spherical ball of plasma.
The sun is primarily composed of hydrogen at approximately 73% and helium at approximately 25%.
Nuclear fusion occurs in the sun's core.
During nuclear fusion, hydrogen is fused into helium.
Nuclear fusion releases enormous amounts of energy.
The released energy is mainly in the form of light and heat.
The released energy makes life on Earth possible.
The surface temperature of the sun is about 5,500°C.
The core temperature of the sun is approximately 15 million°C.
The sun has a diameter of about 1.39 million kilometers.
The diameter of the sun is roughly 109 times the diameter of Earth.
The mass of the sun constitutes approximately 99.86% of the total mass of the solar system.
Sunlight takes about 8 minutes and 20 seconds to reach Earth.
Without the sun, there would be no liquid water on Earth.
Without the sun, there would likely be no life on Earth.
Solar activity varies over an approximately 11-year cycle.
Solar activity includes sunspots and solar flares.
    """
    search_query = "What is the Sun mainly composed of?"
    embedding_model, spacy_nlp = load_models(embedding_model_name, spacy_model_name)
    top_n_results, sentences = closest_results(search_query, fact_text, top_n, embedding_model, spacy_nlp)
    print("\n--- Retrieval Results ---")
    print(f"Search Query: '{search_query}'")
    wrapper = textwrap.TextWrapper(width=80, initial_indent="    ", subsequent_indent="    ")
    for i, (sentence_index, score) in enumerate(top_n_results):
        print(f"\n{i+1}. Similarity: {score:.4f}")
        print(wrapper.fill(f"Sentence {sentence_index}: {sentences[sentence_index]}"))

if __name__ == '__main__':
    main()
