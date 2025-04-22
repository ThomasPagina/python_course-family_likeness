import spacy
import gensim
import numpy as np
import os

# === Formatvorlagen ===
HIGHLIGHT_TEMPLATE = "<span style='background-color: #ffff99; font-weight: bold;'>{}</span>"
EXPLANATION_ITEM_TEMPLATE = "<li><strong>{}</strong> → <em>{}</em> (sim={:.2f})"
EXPLANATION_SIM_WARNING = " ⚠️ möglicherweise schwache semantische Ähnlichkeit"
CANDIDATE_LIST_ITEM = "<li>{} (sim={:.2f})</li>"
HTML_HEADER = "<html><body><h2>Erweiterte Formen</h2><div style='font-family: sans-serif;'>"
HTML_FOOTER = "</div></body></html>"


def load_spacy_model():
    return spacy.load("de_core_news_sm")

def load_german_embedding_model(path="cc.de.300.vec.gz"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file '{path}' not found. Download 'cc.de.300.vec.gz' from https://fasttext.cc/docs/en/crawl-vectors.html"
        )
    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)

def gender_direction(model, source_gender="Masc"):
    return model["frau"] - model["mann"] if source_gender == "Masc" else model["mann"] - model["frau"]

def get_article(token, target_gender):
    case = token.morph.get("Case") or ["Nom"]
    case = case[0]

    if target_gender == "Fem":
        return "die"
    if target_gender == "Masc":
        return "der" if case == "Nom" else "den"
    if target_gender == "Neut":
        return "das"
    return "die"

def is_valid_candidate_word(word, original_word, avoid):
    word_lc = word.lower()
    return (
        word_lc != original_word.lower() and
        word_lc not in avoid and
        word.isalpha() and
        len(word) > 2
    )

def is_valid_noun(doc):
    return doc and doc[0].pos_ == "NOUN"

def get_valid_gender_candidate(model, original_word, direction, nlp, avoid={"frau", "mann"}, topn=20):
    try:
        orig_vec = model[original_word.lower()]
        new_vec = orig_vec + direction
        candidates = model.similar_by_vector(new_vec, topn=topn)
        for word, score in candidates:
            if is_valid_candidate_word(word, original_word, avoid):
                doc = nlp(word)
                if is_valid_noun(doc):
                    return word, score, candidates
    except KeyError:
        pass
    return None, 0.0, []

def build_highlighted_replacement(original_phrase, replacement):
    return HIGHLIGHT_TEMPLATE.format(replacement), original_phrase

def build_explanation(original_phrase, replacement, score, all_candidates, nlp):
    explanation = EXPLANATION_ITEM_TEMPLATE.format(original_phrase, replacement, score)
    if score < 0.6:
        explanation += EXPLANATION_SIM_WARNING
    explanation += "<ul>"
    shown = 0
    for alt_word, alt_score in all_candidates:
        if alt_word.lower() in {"frau", "mann"} or not alt_word.isalpha():
            continue
        alt_doc = nlp(alt_word)
        if is_valid_noun(alt_doc):
            explanation += CANDIDATE_LIST_ITEM.format(alt_word, alt_score)
            shown += 1
        if shown >= 3:
            break
    explanation += "</ul></li>"
    return explanation

def append_explanations(html_output, explanations):
    if explanations:
        html_output.append("<ul>")
        html_output.extend(explanations)
        html_output.append("</ul>")

def extract_article(token):
    for child in token.children:
        if child.pos_ == "DET":
            return child.text
    for left in token.lefts:
        if left.pos_ == "DET":
            return left.text
    return None

def generate_bias_suggestion(token, nlp, model, direction, target_gender):
    article = extract_article(token)
    if not article:
        return None, None

    guess, score, all_candidates = get_valid_gender_candidate(model, token.text, direction, nlp)
    if not guess:
        return None, None

    new_article = get_article(token, target_gender)
    original_phrase = f"{article} {token.text}"
    replacement = f"{article} {token.text} oder {new_article} {guess.capitalize()}"
    highlighted_replacement, _ = build_highlighted_replacement(original_phrase, replacement)
    explanation = build_explanation(original_phrase, replacement, score, all_candidates, nlp)

    return original_phrase, (highlighted_replacement, explanation)

def process_sentence(sent, nlp, model, direction_m2f, direction_f2m):
    modified_sent = sent.text
    explanations = []

    for token in sent:
        if should_consider_token(token):
            direction, target_gender = get_direction_and_target_gender(token, direction_m2f, direction_f2m)
            original_phrase, result = generate_bias_suggestion(token, nlp, model, direction, target_gender)
            if result:
                highlighted_replacement, explanation = result
                modified_sent = modified_sent.replace(original_phrase, highlighted_replacement)
                explanations.append(explanation)

    return modified_sent, explanations

def should_consider_token(token):
    return (
        token.pos_ == "NOUN"
        and token.morph.get("Number") == ["Sing"]
        and token.morph.get("Gender")
        and token.morph.get("Case")
        and token.morph.get("Case")[0] in {"Nom", "Acc"}
    )

def get_direction_and_target_gender(token, direction_m2f, direction_f2m):
    gender = token.morph.get("Gender")[0]
    direction = direction_m2f if gender == "Masc" else direction_f2m
    target_gender = "Fem" if gender == "Masc" else "Masc"
    return direction, target_gender

def process_text_to_html(text, nlp, model):
    doc = nlp(text)
    direction_m2f = gender_direction(model, "Masc")
    direction_f2m = gender_direction(model, "Fem")

    html_output = [HTML_HEADER]

    for sent in doc.sents:
        modified_sent, explanations = process_sentence(sent, nlp, model, direction_m2f, direction_f2m)
        html_output.append(f"<p>{modified_sent}</p>")
        append_explanations(html_output, explanations)

    html_output.append(HTML_FOOTER)
    return "\n".join(html_output)

def create_bias_report(input_text, output_file):
    nlp = load_spacy_model()
    current_path = os.path.dirname(__file__)
    model_path = os.path.join(current_path, "cc.de.300.vec.gz")
    model = load_german_embedding_model(model_path)
    html_content = process_text_to_html(input_text, nlp, model)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ HTML file created: {output_file}")

def main():
    example_text = (
        "Der Arzt behandelt den Patienten. Die Hausfrau kauft Gemüse. "
        "Der Professor hält eine Vorlesung. Der Pilot fliegt das Flugzeug. "
        "Der Gast betritt das Hotel. Die Krankenschwester hilft dem Kind. "
        "Der Ingenieur plant das Projekt. Der Chef gibt Anweisungen. "
        "Die Sekretärin organisiert das Meeting. Der Kunde fragt nach dem Preis. "
        "Der Lehrer erklärt die Aufgabe. Der Rentner genießt den Ruhestand. "
        "Die Ingenieurin plant das Projekt. "
        "Die Kindergärtnerin spielt mit den Kindern. Der Gärtner pflegt die Pflanzen. "
        "Der Vater spielt mit dem Kind. Die Mutter kocht das Essen. "
        "Der Fahrer fährt das Auto. Die Studentin lernt für die Prüfung. "
        "Die Katze fängt die Maus."
    )
    output_file = "beidnennung_ausgabe.html"
    create_bias_report(example_text, output_file)

if __name__ == "__main__":
    main()
