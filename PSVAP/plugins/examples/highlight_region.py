"""
Example plugin: Highlight atoms by coordinate range.

Run this script from the PLUGINS tab to highlight atoms with z > 5 Å.
Modify the threshold or axis to explore different regions.

Available API:
    get_atoms()          → list of Atom objects
    get_positions()      → (N,3) numpy array
    get_selection(query) → boolean mask
    highlight(mask)      → recolors atoms in viewport
    log(msg)             → prints to console
    export(data, fname)  → saves to plugin_output/
"""

# ── Highlight atoms with z > 5 Å ─────────────────────────────────────────
positions = get_positions()

if len(positions) == 0:
    log("No data loaded. Open a file first.")
else:
    threshold = 5.0
    axis      = 2          # 0=x, 1=y, 2=z
    axis_name = ["x", "y", "z"][axis]

    mask = positions[:, axis] > threshold
    n_sel = int(mask.sum())
    n_tot = len(mask)

    highlight(mask)
    log(f"Highlighted {n_sel} / {n_tot} atoms where {axis_name} > {threshold} Å")

    # Export the selection indices
    selected_indices = [i for i, m in enumerate(mask) if m]
    export(selected_indices, "highlighted_indices.txt")
    log(f"Saved {len(selected_indices)} indices to plugin_output/highlighted_indices.txt")