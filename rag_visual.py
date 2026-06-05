"""
=============================================================================
 SEMANTIC SPACE EXPLORER  —  a visual demonstration of RAG retrieval
=============================================================================

RAG ("Retrieval-Augmented Generation") works by turning text into vectors and
finding the ones closest in MEANING to your question. That "meaning space" is
usually invisible. This script makes it VISIBLE.

It embeds a small library of sentences, flattens the high-dimensional meaning
space down to 2D, and then -- for any question you ask -- it lights up the
documents the model would retrieve, drawing glowing beams from your query to
its nearest neighbours. You can literally watch semantic search happen.

WHAT YOU SEE
------------
   * every document is a star, positioned by MEANING (similar ideas cluster)
   * brightness + size  = how relevant that document is to your question
   * the gold star      = your question, placed in the same meaning space
   * glowing beams       = the top matches RAG would feed to the LLM
   * the side panels     = ranked similarity scores + the retrieved context

SETUP (run once)
----------------
    pip install sentence-transformers scikit-learn matplotlib

RUN
---
    python rag_visual.py

A high-resolution image is saved for every example query into  visuals/ ,
and the first one is opened in a window.
=============================================================================
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util


# --------------------------------------------------------------------------
# Look & feel
# --------------------------------------------------------------------------
BG       = "#0d1117"     # deep GitHub-dark background
FG       = "#e6edf3"     # light text
ACCENT   = "#f5c542"     # gold (the query)
GRID     = "#21262d"
CMAP     = "plasma"      # similarity colour map (dark purple -> bright yellow)
OUT_DIR  = "visuals"
TOP_K    = 3             # how many documents RAG retrieves


# --------------------------------------------------------------------------
# A small, deliberately MULTI-TOPIC library so meaning-clusters are visible.
# --------------------------------------------------------------------------
CORPUS = [
    # --- space ---
    "Astronauts aboard the space station watch sixteen sunrises every day.",
    "A rover drilled into red Martian rock searching for ancient microbes.",
    "The telescope captured galaxies whose light is billions of years old.",
    "Rockets burn liquid oxygen and hydrogen to escape Earth's gravity.",
    # --- cooking ---
    "Slowly caramelised onions give the soup a deep, sweet richness.",
    "Fold the egg whites gently so the souffle rises tall and airy.",
    "A cast-iron pan sears the steak into a crackling brown crust.",
    "Fresh basil and ripe tomatoes make the simplest pasta sing.",
    # --- music ---
    "The cellist drew a long, mournful note across the silent hall.",
    "A heavy bassline made the entire dance floor move as one.",
    "She layered soft harmonies over a gentle acoustic guitar.",
    "The drummer counted in and the whole band roared to life.",
    # --- AI / machine learning ---
    "Neural networks adjust their weights by learning from example data.",
    "A language model predicts the next word from everything before it.",
    "Gradient descent nudges parameters downhill to reduce the error.",
    "Training on more data usually helps a model generalise better.",
    # --- ocean ---
    "Bioluminescent plankton lit the night waves an electric blue.",
    "A humpback whale breached, crashing back into the cold grey sea.",
    "Coral reefs shelter a quarter of all known marine species.",
]

# Great, varied example questions (note: they share almost NO words with the
# documents they should match -- proving this is meaning, not keywords).
EXAMPLE_QUERIES = [
    "How does a computer learn patterns from examples?",
    "What can I make for a cosy homemade dinner tonight?",
    "Tell me about humanity exploring other worlds.",
    "Describe the mysterious creatures of the deep ocean.",
]


# --------------------------------------------------------------------------
# Embed the library once and flatten it to 2D for plotting.
# --------------------------------------------------------------------------
def build_space(model: SentenceTransformer):
    doc_vecs = model.encode(CORPUS, convert_to_numpy=True, normalize_embeddings=True)
    pca = PCA(n_components=2, random_state=0).fit(doc_vecs)
    doc_xy = pca.transform(doc_vecs)
    return doc_vecs, pca, doc_xy


def retrieve(model, query, doc_vecs):
    """Return cosine similarities and the indices of the top matches."""
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sims = (doc_vecs @ q[0])                       # cosine sim (vectors are unit length)
    order = np.argsort(sims)[::-1]
    return sims, order, q


# --------------------------------------------------------------------------
# Draw one beautiful frame for a single query.
# --------------------------------------------------------------------------
def draw(query, sims, order, doc_xy, pca, q_vec, index, total):
    # Normalise similarities to 0..1 for colour/size mapping.
    s_min, s_max = sims.min(), sims.max()
    norm = (sims - s_min) / (s_max - s_min + 1e-9)

    # Place the query inside the SAME 2D meaning space.
    q_xy = pca.transform(q_vec)[0]

    fig = plt.figure(figsize=(15, 8.5), facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.1, 1.0], height_ratios=[1.4, 1.0],
                          wspace=0.18, hspace=0.28)

    # ===================== MAIN PANEL: the meaning galaxy =================
    ax = fig.add_subplot(gs[:, 0], facecolor=BG)
    ax.set_title("Semantic Space  —  where meaning lives",
                 color=FG, fontsize=15, fontweight="bold", pad=14, loc="left")

    topk = order[:TOP_K]

    # Glowing beams from the query to each retrieved document.
    for rank, idx in enumerate(topk):
        x = [q_xy[0], doc_xy[idx, 0]]
        y = [q_xy[1], doc_xy[idx, 1]]
        for lw, a in [(7, 0.06), (4, 0.12), (2, 0.25), (0.9, 0.7)]:   # halo -> core
            ax.plot(x, y, color=ACCENT, linewidth=lw, alpha=a, zorder=2,
                    solid_capstyle="round")

    # Soft halos around every document (brighter = more relevant).
    cmap = plt.get_cmap(CMAP)
    for size, a in [(1100, 0.05), (650, 0.08), (340, 0.14)]:
        ax.scatter(doc_xy[:, 0], doc_xy[:, 1], s=size * (0.35 + norm),
                   c=norm, cmap=CMAP, alpha=a, zorder=3, edgecolors="none")
    # Core document points.
    sc = ax.scatter(doc_xy[:, 0], doc_xy[:, 1], s=90 + 320 * norm,
                    c=norm, cmap=CMAP, zorder=4, edgecolors=BG, linewidths=1.2)

    # Label only the retrieved documents (keeps it clean).
    for idx in topk:
        label = CORPUS[idx][:42] + ("..." if len(CORPUS[idx]) > 42 else "")
        ax.annotate(label, (doc_xy[idx, 0], doc_xy[idx, 1]),
                    xytext=(8, 8), textcoords="offset points",
                    color=FG, fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#161b22",
                              ec=ACCENT, alpha=0.85), zorder=6)

    # The query itself: a gold star.
    ax.scatter([q_xy[0]], [q_xy[1]], marker="*", s=900, c=ACCENT,
               edgecolors="white", linewidths=1.5, zorder=7)
    ax.annotate("  your query", (q_xy[0], q_xy[1]), color=ACCENT,
                fontsize=11, fontweight="bold", zorder=7)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("relevance to your question", color=FG, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=GRID)
    plt.setp(cb.ax.get_yticklabels(), color=FG, fontsize=8)
    cb.outline.set_edgecolor(GRID)

    # ===================== TOP-RIGHT: ranked similarity bars =============
    ax2 = fig.add_subplot(gs[0, 1], facecolor=BG)
    ax2.set_title("Retrieval ranking", color=FG, fontsize=12,
                  fontweight="bold", loc="left")
    show = order[:6]
    labels = [CORPUS[i][:24] + "..." for i in show]
    vals = sims[show]
    colors = [ACCENT if i in topk else "#3b4252" for i in show]
    ypos = np.arange(len(show))[::-1]
    ax2.barh(ypos, vals, color=colors, height=0.7)
    for yp, v in zip(ypos, vals):
        ax2.text(v + 0.01, yp, f"{v:.2f}", va="center", color=FG, fontsize=8)
    ax2.set_yticks(ypos); ax2.set_yticklabels(labels, color=FG, fontsize=8)
    ax2.set_xlim(0, 1)
    ax2.tick_params(colors=FG)
    for spine in ax2.spines.values():
        spine.set_color(GRID)

    # ===================== BOTTOM-RIGHT: the RAG context ================
    ax3 = fig.add_subplot(gs[1, 1], facecolor=BG)
    ax3.axis("off")
    ax3.set_title("Context passed to the LLM", color=FG, fontsize=12,
                  fontweight="bold", loc="left")
    text = f"QUESTION:\n  {query}\n\nRETRIEVED CONTEXT:\n"
    for rank, idx in enumerate(topk, 1):
        text += f"  [{rank}] {CORPUS[idx]}\n"
    ax3.text(0.0, 0.86, text, va="top", ha="left", color=FG, fontsize=9,
             family="monospace", transform=ax3.transAxes, wrap=True)

    fig.suptitle("SEMANTIC SPACE EXPLORER", color=ACCENT, fontsize=20,
                 fontweight="bold", x=0.012, ha="left", y=0.985)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"rag_query_{index}.png")
    fig.savefig(path, dpi=130, facecolor=BG, bbox_inches="tight")
    print(f"  saved {path}")
    return fig


# --------------------------------------------------------------------------
def main() -> None:
    print("Loading embedding model... (first run downloads ~90 MB)")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding the document library and building 2D meaning space...")
    doc_vecs, pca, doc_xy = build_space(model)

    print(f"\nRendering {len(EXAMPLE_QUERIES)} example queries:\n")
    first_fig = None
    for i, query in enumerate(EXAMPLE_QUERIES, 1):
        sims, order, q_vec = retrieve(model, query, doc_vecs)
        fig = draw(query, sims, order, doc_xy, pca, q_vec, i, len(EXAMPLE_QUERIES))
        if first_fig is None:
            first_fig = fig
        else:
            plt.close(fig)

    print(f"\nAll images are in the '{OUT_DIR}/' folder.")
    print("Opening the first one... (close the window to finish)")
    plt.show()


if __name__ == "__main__":
    main()
