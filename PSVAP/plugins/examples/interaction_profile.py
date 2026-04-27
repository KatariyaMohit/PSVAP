import numpy as np

def main():
    log("=" * 50)
    log("=" * 50)

    atoms = get_atoms()
    n_f = n_frames()
    n_atoms = len(atoms)
    
    if n_atoms < 300 or n_f < 2:
        log("ERROR: Need a larger trajectory for this analysis.")
        return

    idx_a = list(range(6, 135)) 
    idx_b = list(range(147, 276))
    
    log(f"Analyzing Group A (atoms 6-134) vs Group B (atoms 147-275)")
    log(f"Processing {n_f} frames...")

    r_com_list = []
    for f in range(n_f):
        pos = get_frame(f)
        
        # Calculate centers of mass manually to be extra safe
        # Center of Mass = mean of positions
        com_a = pos[idx_a].mean(axis=0)
        com_b = pos[idx_b].mean(axis=0)
        
        # Distance calculation
        diff = com_a - com_b
        dist = np.sqrt(np.sum(diff**2))
        r_com_list.append(float(dist))

    # Placeholder Interaction Energy (1/r)
    e_elec_list = [1000.0 / r for r in r_com_list] 

    # --- BINNING LOGIC (Unpacking-Free) ---
    # Instead of min, max = ..., we do it on separate lines
    r_min = min(r_com_list)
    r_max = max(r_com_list)
    
    log(f"Distance Range: {r_min:.2f} to {r_max:.2f} Å")

    bins = np.linspace(r_min, r_max, 30)
    
    e_binned = []
    bin_centers = []

    for i in range(len(bins) - 1):
        b_start = bins[i]
        b_end = bins[i+1]
        bin_centers.append(0.5 * (b_start + b_end))
        
        # Collect energies in this distance bin
        energies_in_bin = []
        for j in range(len(r_com_list)):
            r = r_com_list[j]
            if r >= b_start and r < b_end:
                energies_in_bin.append(e_elec_list[j])
        
        if len(energies_in_bin) > 0:
            avg_e = sum(energies_in_bin) / len(energies_in_bin)
            e_binned.append(avg_e)
        else:
            e_binned.append(0.0)

    log("\n--- RESULTS: MEAN INTERACTION VS DISTANCE ---")
    log("Dist (Å) | Mean Energy (kJ/mol)")
    log("---------|----------------------")
    for i in range(len(bin_centers)):
        if e_binned[i] != 0:
            log(f"{bin_centers[i]:8.2f} | {e_binned[i]:12.4f}")

    export({"distance": bin_centers, "energy": e_binned}, "interaction_profile.json")
    log("\n✓ SUCCESS: Results saved to interaction_profile.json")

main()