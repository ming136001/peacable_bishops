import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from collections import defaultdict
import pandas as pd

# === Helper functions ===

def get_pos_diag(r, c):
    return r - c

def get_neg_diag(r, c):
    return r + c

def greedy_balanced_placement(n, c, shuffle=True):
    """
    One greedy attempt to place c armies on an n×n board.
    Returns (b, armies) where b = min army size.
    """
    if n == 0 or c == 0:
        return 0, [[] for _ in range(c)]

    armies = [[] for _ in range(c)]
    pos_diag_owner = {}
    neg_diag_owner = {}

    squares = [(r, col) for r in range(n) for col in range(n)]
    if shuffle:
        random.shuffle(squares)

    for r, col in squares:
        pd = get_pos_diag(r, col)
        nd = get_neg_diag(r, col)

        # Try to balance armies
        army_sizes = [(i, len(armies[i])) for i in range(c)]
        army_sizes.sort(key=lambda x: x[1])

        for army_id, _ in army_sizes:
            pd_free = (pd not in pos_diag_owner) or (pos_diag_owner[pd] == army_id)
            nd_free = (nd not in neg_diag_owner) or (neg_diag_owner[nd] == army_id)

            if pd_free and nd_free:
                armies[army_id].append((r, col))
                pos_diag_owner[pd] = army_id
                neg_diag_owner[nd] = army_id
                break

    b = min(len(army) for army in armies)
    return b, armies


# === Improved search for optimal B(c, n) ===

def find_best_arrangement(n, c, num_trials=5000):
    """
    Run multiple greedy trials to approximate the optimal B(c, n).
    Keeps the best (highest b) configuration.
    """
    best_b = 0
    best_armies = None

    # Track how many times we've seen each result
    result_counts = defaultdict(int)

    progress = st.progress(0, text=f"Searching optimal B({c}, {n})...")
    for t in range(1, num_trials + 1):
        b, armies = greedy_balanced_placement(n, c)
        result_counts[b] += 1
        if b > best_b:
            best_b = b
            best_armies = armies
        progress.progress(t / num_trials, text=f"Trial {t}/{num_trials} — best so far: {best_b}")
    progress.empty()

    if best_armies is None:
        best_armies = [[] for _ in range(c)]

    return best_b, best_armies, result_counts


# === Visualization Function ===

def plot_armies_board(n, armies, title=None):
    """
    Visualize armies using colored squares instead of bishop figures.
    Each army is represented by a different color.
    Only shows the minimum number of bishops per army to ensure equal sizes.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Find the minimum army size
    min_size = min(len(army) for army in armies) if armies else 0

    # Trim each army to the minimum size
    trimmed_armies = [army[:min_size] for army in armies]

    # Generate distinct colors for each army using a colormap
    num_armies = len(trimmed_armies)
    if num_armies <= 10:
        # Use tab10 for up to 10 armies
        army_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:num_armies]
    else:
        # Use tab20 for more armies
        army_colors = plt.cm.tab20(np.linspace(0, 1, 20))[:num_armies]

    # Draw chessboard
    for i in range(n):
        for j in range(n):
            color = 'cornsilk' if (i + j) % 2 == 0 else 'saddlebrown'
            rect = patches.Rectangle((j, n - 1 - i), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)

    # Draw colored squares for each army (using trimmed armies)
    for idx, army in enumerate(trimmed_armies):
        for (r, c) in army:
            # Draw a colored circle in the center of the square
            circle = patches.Circle((c + 0.5, n - 1 - r + 0.5), 0.35,
                                   facecolor=army_colors[idx],
                                   edgecolor='white',
                                   linewidth=2,
                                   alpha=0.9)
            ax.add_patch(circle)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    title = title or f"Army arrangement for n={n}, armies={len(armies)}"
    ax.set_title(title, fontsize=14)

    # Add legend (showing only army colors)
    legend_elements = [patches.Patch(facecolor=army_colors[i],
                                     edgecolor='white',
                                     label=f'Army {i+1}')
                      for i in range(len(trimmed_armies))]
    ax.legend(handles=legend_elements,
             loc='center left',
             bbox_to_anchor=(1, 0.5),
             fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)


def plot_bcn_matrix(best_results, max_n=15, max_c=15):
    """
    Create a heatmap visualization of B(c, n) values.
    """
    # Create matrix with NaN for missing values
    matrix = np.full((max_c, max_n), np.nan)

    # Fill in known values
    for (c_val, n_val), (b_val, _) in best_results.items():
        if c_val <= max_c and n_val <= max_n:
            matrix[c_val - 1, n_val - 1] = b_val

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create heatmap with custom colormap
    cmap = plt.cm.YlOrRd
    cmap.set_bad(color='lightgray')

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')

    # Set ticks and labels
    ax.set_xticks(np.arange(max_n))
    ax.set_yticks(np.arange(max_c))
    ax.set_xticklabels(np.arange(1, max_n + 1))
    ax.set_yticklabels(np.arange(1, max_c + 1))

    ax.set_xlabel('n (Board Size)', fontsize=12, fontweight='bold')
    ax.set_ylabel('c (Number of Armies)', fontsize=12, fontweight='bold')
    ax.set_title('B(c, n) Matrix - Maximum Pieces per Army', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(max_c):
        for j in range(max_n):
            if not np.isnan(matrix[i, j]):
                text = ax.text(j, i, int(matrix[i, j]),
                             ha="center", va="center", color="black", fontsize=9, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('B(c, n) Value', rotation=270, labelpad=20, fontsize=11)

    # Add grid
    ax.set_xticks(np.arange(max_n) - 0.5, minor=True)
    ax.set_yticks(np.arange(max_c) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=1.5)

    plt.tight_layout()
    return fig


# === Streamlit UI ===

st.set_page_config(page_title="Army Visualizer", layout="centered")

# Initialize session state for caching best results
if 'best_results' not in st.session_state:
    st.session_state.best_results = {}

# Initialize the matrix if not already done
if 'matrix_initialized' not in st.session_state:
    st.session_state.matrix_initialized = False

def initialize_matrix(max_c=15, max_n=15, num_trials=50):
    """
    Pre-compute all B(c, n) values for the matrix.
    """
    total_cells = max_c * max_n
    progress_bar = st.progress(0, text="Initializing B(c, n) matrix...")
    status_text = st.empty()

    computed = 0
    for c_val in range(1, max_c + 1):
        for n_val in range(1, max_n + 1):
            key = (c_val, n_val)
            if key not in st.session_state.best_results:
                b, armies, _ = find_best_arrangement(n_val, c_val, num_trials=num_trials)
                st.session_state.best_results[key] = (b, armies)

            computed += 1
            progress_bar.progress(computed / total_cells,
                                 text=f"Computing B({c_val}, {n_val})... [{computed}/{total_cells}]")
            status_text.text(f"Current: B({c_val}, {n_val}) = {st.session_state.best_results[key][0]}")

    progress_bar.empty()
    status_text.empty()
    st.session_state.matrix_initialized = True

st.title("♗ Peaceable Bishop Armies on Chessboard")
st.markdown("""
This app estimates **B(c, n)** — the maximum number of bishops per army
that can be placed on an n×n chessboard
so that no two bishops of the same army attack each other diagonally.
Each army is represented by a distinct color.
It runs multiple randomized greedy trials to find the *best* configuration
automatically.""")

col1, col2 = st.columns(2)
n = col1.number_input("Board size (n)", min_value=1, max_value=15, value=8)
c = col2.number_input("Number of armies (c)", min_value=1, max_value=12, value=5)
num_trials = 1000  # Fixed number of trials

# Show cached results if any
if st.session_state.best_results:
    with st.expander("📊 Best Results Found in This Session"):
        for (c_val, n_val), (b_val, _) in sorted(st.session_state.best_results.items()):
            st.write(f"B({c_val}, {n_val}) = **{b_val}**")

if st.button("Compute Optimal B(c, n)"):
    key = (int(c), int(n))

    with st.spinner(f"Searching best arrangement for B({c},{n})..."):
        b, armies, result_counts = find_best_arrangement(int(n), int(c), num_trials=num_trials)

    # Update cache if this is a new best
    if key not in st.session_state.best_results or b > st.session_state.best_results[key][0]:
        st.session_state.best_results[key] = (b, armies)
        st.success(f"✅ NEW MAXIMUM B({c}, {n}) = {b} 🎉")
    elif b == st.session_state.best_results[key][0]:
        st.success(f"✅ Confirmed Maximum B({c}, {n}) = {b} ✓")
    else:
        st.warning(f"⚠️ Found B = {b}, but previous best was {st.session_state.best_results[key][0]}")
        st.info("Displaying the best result found so far...")
        b, armies = st.session_state.best_results[key]

    st.write(f"Army sizes: {[len(a) for a in armies]}")

    # Show the plot first
    plot_armies_board(int(n), armies, title=f"Best arrangement for B({c}, {n}) = {b}")

    # Show distribution of results to indicate confidence (below the plot)
    with st.expander("📊 Result Distribution (this run)"):
        sorted_results = sorted(result_counts.items(), reverse=True)
        for value, count in sorted_results:
            percentage = (count / num_trials) * 100
            marker = "← BEST" if value == b else ""
            st.write(f"B = {value}: {count} times ({percentage:.1f}%) {marker}")

st.markdown("---")
st.caption("Developed by Nong Ming · Optimized Greedy Balanced Algorithm with Color Visualization")

