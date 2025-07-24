import plotly.express as px
import pandas as pd

def plot_slice_probabilities(preds: np.ndarray, labels: list[str]) -> None:
    """
    preds: array of shape (S, C) with per‑slice class probabilities
    labels: list of C class names
    """
    # build a DataFrame with slice index and per‑class probs
    df = pd.DataFrame(preds, columns=labels)
    df["slice"] = df.index

    fig = px.line(
        df,
        x="slice",
        y=labels,
        markers=True,
        title="Per-slice Class Probabilities",
        labels={"value": "Probability", "slice": "Slice Index", "variable": "Class"}
    )
    fig.update_layout(legend_title_text="Class", hovermode="x unified")
    return fig
