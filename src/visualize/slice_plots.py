# src/visualize/bar_chart.py
import numpy as np
import plotly.graph_objects as go

def plot_slice_bar_chart(preds: np.ndarray, slice_idx: int, labels: list[str]) -> go.Figure:
    """
    Build a horizontal bar chart of the class probabilities for a single slice.
    preds: array of shape (S, C)
    slice_idx: which slice to plot
    labels: list of length C of the class names
    """
    # grab that slice's probs
    slice_probs  = preds[slice_idx]
    # sort descending
    sorted_idx    = np.argsort(slice_probs)[::-1]
    sorted_labels = [labels[i] for i in sorted_idx]
    sorted_probs  = slice_probs[sorted_idx]

    # build the figure
    fig = go.Figure(go.Bar(
        x=sorted_probs,
        y=sorted_labels,
        orientation='h',
        marker_color=['red' if lbl == sorted_labels[0] else 'blue'
                      for lbl in sorted_labels]
    ))
    fig.update_layout(
        title=f"Slice {slice_idx} Probabilities",
        xaxis_title="Probability",
        yaxis_title="Class",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=100, r=20, t=50, b=20)
    )

    # add labels
    for i, p in enumerate(sorted_probs):
        fig.add_annotation(
            x=p, y=i,
            text=f"{p:.4f}",
            showarrow=False,
            xanchor="left",
            xshift=5
        )
    return fig
