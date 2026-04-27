import numpy as np

def main():
    log("=" * 50)
    log("=" * 50)

    atoms = get_atoms()
    n_f = n_frames()
    n_atoms = len(atoms)
    
    if n_atoms < 2 or n_f < 2:
        log("ERROR: Need at least 2 atoms and a trajectory.")
        return

    # --- DEFINE GROUPS ---
    # Tracking from the start of the chain to the end
    idx_a = 0 
    idx_b = n_atoms - 1
    
    log(f"Tracking: Atom {idx_a} to Atom {idx_b}")
    log(f"Processing {n_f} frames...")

    distances = []
    rgs = []

    for f in range(n_f):
        pos = get_frame(f)
        
        # 1. Distance (r)
        p_a = pos[idx_a]
        p_b = pos[idx_b]
        d = np.sqrt(np.sum((p_a - p_b)**2))
        distances.append(float(d))
        
        # 2. Radius of Gyration (Rg)
        center = pos.mean(axis=0)
        rg = np.sqrt(np.mean(np.sum((pos - center)**2, axis=1)))
        rgs.append(float(rg))

    # 3. Normalization (xi = r / avg_Rg)
    avg_rg = sum(rgs) / len(rgs)
    xi = [d / avg_rg for d in distances]
    
    log(f"Average Rg calculated: {avg_rg:.3f} Å")

    # 4. Energy Calculation (PMF)
    # We use np.histogram but process the results with pure Python to avoid sandbox errors
    counts_raw, bins_raw = np.histogram(xi, bins=20)
    
    counts = [float(x) for x in counts_raw]
    bins = [float(x) for x in bins_raw]
    total = sum(counts)
    
    if total == 0:
        log("ERROR: No distribution found.")
        return

    log("\n--- ENERGY LANDSCAPE (PMF) ---")
    log(" r/Rg  | Energy (kJ/mol)")
    log("-------|----------------")

    bin_centers = []
    pmf_values = []

    for i in range(len(counts)):
        # Calculate center of the bin
        center = (bins[i] + bins[i+1]) / 2
        bin_centers.append(center)
        
        # Calculate Probability
        p = counts[i] / total
        if p <= 0: p = 1e-9 # Safety for log
        
        # PMF = -RT * ln(P). Then normalize so min energy is 0.
        # (We will normalize the final list below)
        energy = -0.00831 * 300 * np.log(p)
        pmf_values.append(energy)

    # Normalize: Energy - min(Energy)
    min_e = min(pmf_values)
    final_pmf = [e - min_e for e in pmf_values]

    for i in range(len(bin_centers)):
        log(f"{bin_centers[i]:6.2f} | {final_pmf[i]:.4f}")

    # 5. Export for TA
    export({"xi": bin_centers, "pmf": final_pmf}, "pmf_final.json")
    log("\n✓ SUCCESS: Results saved to plugin_output/pmf_final.json")

main()