import gensim.downloader as api
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure

def load_model():
    print("Loading GloVe model (50 dimensions)...")
    model = api.load("glove-wiki-gigaword-50")
    print("Model loaded successfully.")
    return model

def compute_bias_direction(model, word_pair):
    vec1 = model[word_pair[0].lower()]
    vec2 = model[word_pair[1].lower()]
    return vec1 - vec2

def project_words(model, words, bias_directions):
    word_vectors = np.array([model[word.lower()] for word in words])
    normed_directions = [vec / np.linalg.norm(vec) for vec in bias_directions]
    projections = np.array([[np.dot(vec, direction) for direction in normed_directions]
                            for vec in word_vectors])
    return projections, word_vectors

def visualize_bias_interactive(words, projections, axis_labels, title):
    df = pd.DataFrame(projections, columns=axis_labels)
    df["word"] = words

    role_map = {
        "lawyer": "professional", "janitor": "service", "professor": "academic", "cashier": "service",
        "artist": "creative", "banker": "financial", "cleaner": "service", "doctor": "medical",
        "plumber": "technical", "executive": "executive", "waiter": "service", "scientist": "academic",
        "manager": "executive", "clerk": "administrative", "ceo": "executive", "farmer": "manual",
        "mechanic": "technical", "researcher": "academic"
    }
    df["role"] = df["word"].map(role_map)

    fig = px.scatter(
        df,
        x=axis_labels[0],
        y=axis_labels[1],
        color=axis_labels[0],
        hover_data=["word", "role"],
        text="word",
        title=title,
        width=950,
        height=600,
        color_continuous_scale="RdBu"
    )

    fig.update_traces(marker=dict(size=12), textposition="top center")

    # Dropdown buttons to filter roles
    role_options = df["role"].unique().tolist()
    buttons = [
        dict(label="All", method="restyle", args=[{"visible": [True]}])
    ]
    for role in role_options:
        visibility = df["role"] == role
        buttons.append(
            dict(label=role.capitalize(),
                 method="update",
                 args=[{"transforms": [dict(
                     type='filter',
                     target=df["role"],
                     operation='=',
                     value=role
                 )]}])
        )

    fig.update_layout(
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "x": 1.15,
            "y": 0.7,
            "showactive": True,
            "xanchor": "left",
            "yanchor": "middle"
        }],
        xaxis_title=axis_labels[0] + "  (→ rich)",
        yaxis_title=axis_labels[1] + "  (→ powerful)",
        coloraxis_colorbar=dict(title=axis_labels[0])
    )

    fig.show()

def main():
    model = load_model()

    bias_rich_poor = compute_bias_direction(model, ("rich", "poor"))
    bias_powerful_powerless = compute_bias_direction(model, ("powerful", "powerless"))

    words = [
        "lawyer", "janitor", "professor", "cashier", "artist", "banker",
        "cleaner", "doctor", "plumber", "executive", "waiter", "scientist",
        "manager", "clerk", "ceo", "farmer", "mechanic", "researcher"
    ]

    projections, _ = project_words(model, words, [bias_rich_poor, bias_powerful_powerless])

    print("Word projections on bias axes ('rich−poor', 'powerful−powerless'):")
    for word, (rp_proj, pow_proj) in zip(words, projections):
        print(f"{word:12s} → rich-poor: {rp_proj:+.4f}, powerful-powerless: {pow_proj:+.4f}")

    visualize_bias_interactive(
        words,
        projections,
        axis_labels=["rich − poor", "powerful − powerless"],
        title="Social Bias Map: Rich vs Poor and Powerful vs Powerless"
    )

if __name__ == '__main__':
    main()
