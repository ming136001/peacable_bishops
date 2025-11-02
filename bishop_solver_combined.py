import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from ortools.sat.python import cp_model
import time
import random
from collections import defaultdict

# ============================================================================
# CP-SAT SOLVER (Slower but Optimal)
# ============================================================================

def solve_b_c_n_cpsat(num_armies, board_size, time_limit_seconds=60):
    """
    Solve B(c, n) using CP-SAT and return the placement.
    Returns: (best_solution, upper_bound, status, solve_time, armies_positions)
    """
    model = cp_model.CpModel()

    # Variables: bishops[army][row][col]
    bishops = {}
    for army in range(num_armies):
        bishops[army] = {}
        for r in range(board_size):
            bishops[army][r] = {}
            for c in range(board_size):
                bishops[army][r][c] = model.NewBoolVar(f'bishop_a{army}_r{r}_c{c}')

    # Variable for army size
    army_size = model.NewIntVar(0, board_size * board_size, 'army_size')

    # Constraint 1: Each army has exactly 'army_size' bishops
    for army in range(num_armies):
        army_bishops = []
        for r in range(board_size):
            for c in range(board_size):
                army_bishops.append(bishops[army][r][c])
        model.Add(sum(army_bishops) == army_size)

    # Constraint 2: At most one army can occupy each cell
    for r in range(board_size):
        for c in range(board_size):
            cell_occupancy = [bishops[army][r][c] for army in range(num_armies)]
            model.Add(sum(cell_occupancy) <= 1)

    # Constraint 3: Different armies cannot share diagonals
    slash_diagonals = {}
    backslash_diagonals = {}

    for r in range(board_size):
        for c in range(board_size):
            slash_key = r + c
            backslash_key = r - c

            if slash_key not in slash_diagonals:
                slash_diagonals[slash_key] = []
            slash_diagonals[slash_key].append((r, c))

            if backslash_key not in backslash_diagonals:
                backslash_diagonals[backslash_key] = []
            backslash_diagonals[backslash_key].append((r, c))

    for diag_dict in [slash_diagonals, backslash_diagonals]:
        for cells in diag_dict.values():
            for army1 in range(num_armies):
                for army2 in range(army1 + 1, num_armies):
                    for r1, c1 in cells:
                        for r2, c2 in cells:
                            model.AddImplication(bishops[army1][r1][c1],
                                               bishops[army2][r2][c2].Not())

    # Objective: Maximize army_size
    model.Maximize(army_size)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 8

    start_time = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - start_time

    armies_positions = [[] for _ in range(num_armies)]

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        best_solution = solver.Value(army_size)
        upper_bound = int(solver.BestObjectiveBound())

        # Extract the positions
        for army in range(num_armies):
            for r in range(board_size):
                for c in range(board_size):
                    if solver.Value(bishops[army][r][c]) == 1:
                        armies_positions[army].append((r, c))

        return (best_solution, upper_bound, solver.StatusName(status), solve_time, armies_positions)
    else:
        return (0, 0, solver.StatusName(status), solve_time, armies_positions)


# ============================================================================
# MONTE CARLO FAIR PLACEMENT (MCFP) ALGORITHM (Faster but Heuristic)
# ============================================================================

def get_pos_diag(r, c):
    return r - c

def get_neg_diag(r, c):
    return r + c

def greedy_balanced_placement(n, c, shuffle=True):
    """
    One greedy attempt to place c armies on an n×n board using MCFP.
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


def find_best_arrangement_mcfp(n, c, num_trials=5000):
    """
    Run multiple Monte Carlo trials to approximate the optimal B(c, n).
    Keeps the best (highest b) configuration.
    """
    best_b = 0
    best_armies = None

    # Track how many times we've seen each result
    result_counts = defaultdict(int)

    progress = st.progress(0, text=f"Running MCFP algorithm for B({c}, {n})...")
    for t in range(1, num_trials + 1):
        b, armies = greedy_balanced_placement(n, c)
        result_counts[b] += 1
        if b > best_b:
            best_b = b
            best_armies = armies
        if t % 100 == 0:  # Update every 100 trials
            progress.progress(t / num_trials, text=f"Trial {t}/{num_trials} — best so far: {best_b}")
    progress.empty()

    if best_armies is None:
        best_armies = [[] for _ in range(c)]

    return best_b, best_armies, result_counts


# ============================================================================
# VISUALIZATION FUNCTION (Shared)
# ============================================================================

def plot_armies_board(n, armies, title=None, status_info=None):
    """
    Visualize armies using colored circles on a chessboard.
    Each army is represented by a different color.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Generate distinct colors for each army using a colormap
    num_armies = len(armies)
    if num_armies <= 10:
        army_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:num_armies]
    else:
        army_colors = plt.cm.tab20(np.linspace(0, 1, 20))[:num_armies]

    # Draw chessboard
    for i in range(n):
        for j in range(n):
            color = 'cornsilk' if (i + j) % 2 == 0 else 'saddlebrown'
            rect = patches.Rectangle((j, n - 1 - i), 1, 1, facecolor=color,
                                    edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)

    # Draw colored circles for each army
    for idx, army in enumerate(armies):
        for (r, c) in army:
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

    # Create title with status info
    if title is None:
        title = f"B({len(armies)}, {n})"
    if status_info:
        title += f"\n{status_info}"
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add legend
    legend_elements = [patches.Patch(facecolor=army_colors[i],
                                     edgecolor='white',
                                     label=f'Army {i+1} ({len(armies[i])} bishops)')
                      for i in range(len(armies))]
    ax.legend(handles=legend_elements,
             loc='center left',
             bbox_to_anchor=(1, 0.5),
             fontsize=10)

    plt.tight_layout()
    return fig


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="Peaceable Bishop Solver", layout="centered")

# Initialize session state
if 'cpsat_results' not in st.session_state:
    st.session_state.cpsat_results = {}
if 'mcfp_results' not in st.session_state:
    st.session_state.mcfp_results = {}

st.title("♗ Peaceable Bishop Armies Solver")

st.markdown("""
This app solves the **Peaceable Bishop Problem**: finding **B(c, n)** — the maximum number
of bishops per army that can be placed on an n×n chessboard with c armies, such that
bishops from different armies don't attack each other diagonally.

Choose between two algorithms with different trade-offs:
""")

# Algorithm comparison table
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🐢 CP-SAT Solver")
    st.markdown("""
    **Characteristics:**
    - ⏱️ **Speed:** Slower (can take minutes)
    - ✅ **Accuracy:** Optimal solutions
    - 🎯 **Best for:** Small-medium problems (n≤10, c≤6)
    - 🔬 **Method:** Constraint programming
    """)

with col2:
    st.markdown("### 🚀 MCFP Algorithm")
    st.markdown("""
    **Characteristics:**
    - ⚡ **Speed:** Very fast (seconds)
    - 📊 **Accuracy:** High-quality heuristic
    - 🎯 **Best for:** Larger problems & quick results
    - 🎲 **Method:** Monte Carlo Fair Placement
    """)

st.markdown("---")

# Algorithm selection
algorithm_choice = st.radio(
    "**Select Algorithm:**",
    ["CP-SAT Solver (Slower but Optimal)", "MCFP Algorithm (Faster but Heuristic)"],
    horizontal=True
)

# Input parameters
col1, col2 = st.columns(2)
with col1:
    if "CP-SAT" in algorithm_choice:
        n = st.number_input("Board size (n)", min_value=1, max_value=55, value=8,
                           help="Size of the chessboard (n×n)")
    else:
        n = st.number_input("Board size (n)", min_value=1, max_value=50, value=8,
                           help="Size of the chessboard (n×n)")
with col2:
    if "CP-SAT" in algorithm_choice:
        c = st.number_input("Number of armies (c)", min_value=1, max_value=10, value=3,
                           help="Number of peaceful bishop armies")
    else:
        c = st.number_input("Number of armies (c)", min_value=1, max_value=15, value=3,
                           help="Number of peaceful bishop armies")

# Algorithm-specific settings
if "CP-SAT" in algorithm_choice:
    with st.expander("⚙️ CP-SAT Settings"):
        time_limit = st.slider("Time Limit (seconds)",
                              min_value=10, max_value=300, value=60, step=10,
                              help="Maximum time the solver will run")
else:
    with st.expander("⚙️ MCFP Settings"):
        num_trials = st.slider("Number of Trials",
                              min_value=100, max_value=10000, value=5000, step=100,
                              help="More trials = better results but slower")

st.markdown("---")

# Compute button
button_label = "🚀 Solve with CP-SAT" if "CP-SAT" in algorithm_choice else "⚡ Solve with MCFP"
if st.button(button_label, type="primary"):

    if "CP-SAT" in algorithm_choice:
        # ======== CP-SAT SOLVER ========
        key = (int(c), int(n))

        # Check if already computed
        if key in st.session_state.cpsat_results:
            st.info(f"ℹ️ Using cached result for B({c}, {n})")
            result = st.session_state.cpsat_results[key]
            lower_bound = result['lower_bound']
            upper_bound = result['upper_bound']
            status = result['status']
            solve_time = result['solve_time']
            armies = result['armies']
        else:
            with st.spinner(f"🔍 CP-SAT solver is working on B({c}, {n})..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Run solver
                lower_bound, upper_bound, status, solve_time, armies = solve_b_c_n_cpsat(
                    int(c), int(n), time_limit_seconds=time_limit
                )

                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()

                # Cache the result
                st.session_state.cpsat_results[key] = {
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'status': status,
                    'solve_time': solve_time,
                    'armies': armies
                }

        # Display results
        st.markdown("---")
        st.header("📊 CP-SAT Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Lower Bound", lower_bound)
        with col2:
            st.metric("Upper Bound", upper_bound)
        with col3:
            if status == 'OPTIMAL':
                st.metric("Status", "✅ OPTIMAL")
            elif status == 'FEASIBLE':
                st.metric("Status", "⏱️ FEASIBLE")
            else:
                st.metric("Status", f"❌ {status}")
        with col4:
            st.metric("Solve Time", f"{solve_time:.2f}s")

        # Interpretation
        if status == 'OPTIMAL':
            st.success(f"✅ **Optimal Solution:** B({c}, {n}) = {lower_bound}")
            st.write("The solver has verified this is the maximum possible value.")
        elif status == 'FEASIBLE':
            if lower_bound == upper_bound:
                st.success(f"✅ **Likely Optimal:** B({c}, {n}) = {lower_bound}")
                st.write("Within the time limit, upper and lower bounds match.")
            else:
                st.warning(f"⚠️ **Bounds:** {lower_bound} ≤ B({c}, {n}) ≤ {upper_bound}")
                st.write(f"The solver found a solution with {lower_bound} bishops per army. "
                        f"The true value is in this range.")
        else:
            st.error(f"❌ Solver failed with status: {status}")

        # Show army sizes
        army_sizes = [len(army) for army in armies]
        st.write(f"**Army sizes:** {army_sizes}")

        if min(army_sizes) == lower_bound and max(army_sizes) == lower_bound:
            st.write("✓ All armies have equal size (as required)")

        # Visualization
        if armies and any(army for army in armies):
            st.markdown("---")
            st.header("🎨 Board Visualization")

            status_info = f"Status: {status} | Time: {solve_time:.2f}s | Value: {lower_bound}"
            fig = plot_armies_board(int(n), armies,
                                   title=f"CP-SAT Solution: B({c}, {n}) = {lower_bound}",
                                   status_info=status_info)
            st.pyplot(fig)

            # Show detailed positions in expander
            with st.expander("🔍 View Detailed Bishop Positions"):
                for idx, army in enumerate(armies):
                    st.write(f"**Army {idx+1}** ({len(army)} bishops):")
                    positions = sorted(army)
                    st.write(positions)
        else:
            st.warning("No solution found or solution is empty.")

    else:
        # ======== MCFP ALGORITHM ========
        key = (int(c), int(n))

        with st.spinner(f"⚡ Running MCFP algorithm for B({c}, {n})..."):
            b, armies, result_counts = find_best_arrangement_mcfp(int(n), int(c), num_trials=num_trials)
            solve_time = sum(result_counts.values()) / 1000  # Approximate time

        # Update cache
        if key not in st.session_state.mcfp_results or b > st.session_state.mcfp_results[key][0]:
            st.session_state.mcfp_results[key] = (b, armies, result_counts)
            st.success(f"✅ NEW MAXIMUM B({c}, {n}) = {b} 🎉")
        elif b == st.session_state.mcfp_results[key][0]:
            st.success(f"✅ Confirmed Maximum B({c}, {n}) = {b} ✓")
        else:
            st.warning(f"⚠️ Found B = {b}, but previous best was {st.session_state.mcfp_results[key][0]}")
            b, armies, result_counts = st.session_state.mcfp_results[key]

        # Display results
        st.markdown("---")
        st.header("📊 MCFP Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("B(c, n) Value", b)
        with col2:
            st.metric("Trials Run", num_trials)
        with col3:
            confidence = (result_counts[b] / num_trials) * 100
            st.metric("Best Result Frequency", f"{confidence:.1f}%")

        st.info(f"💡 **Heuristic Result:** B({c}, {n}) ≈ {b}")
        st.write("Note: This is a high-quality heuristic solution. For optimal results, use CP-SAT solver.")

        # Show army sizes
        army_sizes = [len(army) for army in armies]
        st.write(f"**Army sizes:** {army_sizes}")

        # Visualization
        st.markdown("---")
        st.header("🎨 Board Visualization")

        fig = plot_armies_board(int(n), armies,
                               title=f"MCFP Solution: B({c}, {n}) ≈ {b}",
                               status_info=f"Trials: {num_trials} | Best frequency: {confidence:.1f}%")
        st.pyplot(fig)

        # Show distribution
        with st.expander("📊 Result Distribution"):
            sorted_results = sorted(result_counts.items(), reverse=True)
            for value, count in sorted_results:
                percentage = (count / num_trials) * 100
                marker = "← BEST" if value == b else ""
                st.write(f"B = {value}: {count} times ({percentage:.1f}%) {marker}")

        # Show detailed positions in expander
        with st.expander("🔍 View Detailed Bishop Positions"):
            for idx, army in enumerate(armies):
                st.write(f"**Army {idx+1}** ({len(army)} bishops):")
                positions = sorted(army)
                st.write(positions)

# Sidebar with cached results
with st.sidebar:
    st.header("📚 Cached Results")

    if st.session_state.cpsat_results:
        st.subheader("CP-SAT Results")
        for (c_val, n_val), result in sorted(st.session_state.cpsat_results.items()):
            status_emoji = "✅" if result['status'] == 'OPTIMAL' else "⏱️"
            st.write(f"{status_emoji} B({c_val}, {n_val}) = {result['lower_bound']}")

    if st.session_state.mcfp_results:
        st.subheader("MCFP Results")
        for (c_val, n_val), (b_val, _, _) in sorted(st.session_state.mcfp_results.items()):
            st.write(f"⚡ B({c_val}, {n_val}) ≈ {b_val}")

    if not st.session_state.cpsat_results and not st.session_state.mcfp_results:
        st.write("No results yet")

    if st.session_state.cpsat_results or st.session_state.mcfp_results:
        if st.button("🗑️ Clear All Cache"):
            st.session_state.cpsat_results = {}
            st.session_state.mcfp_results = {}
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
### 📚 About the Algorithms

**CP-SAT Solver:**
- Uses Google OR-Tools Constraint Programming SAT solver
- Provides near optimal solutions
- Best for rigorous mathematical results
- Can be slow for large instances

**Monte Carlo Fair Placement (MCFP) Algorithm:**
- Uses randomized greedy placement with fairness balancing
- Runs thousands of trials to find best configuration
- Excellent practical performance
- Results are high-quality heuristic approximations

### 🎯 When to Use Each Algorithm

- **Research/Publications:** Use CP-SAT for optimal values
- **Quick Exploration:** Use MCFP for fast results on large boards
- **Verification:** Run both and compare results!
""")

st.caption("Developed by Nong Ming · Powered by OR-Tools & Monte Carlo Methods")
