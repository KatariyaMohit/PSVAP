import numpy as np

def calculate_dihedral(p1, p2, p3, p4):
    """Strict IUPAC convention dihedral angle calculation."""
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3
    
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-6 or n2_norm < 1e-6:
        return None
        
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    
    cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    
    # Determine sign using strict IUPAC convention
    if np.dot(np.cross(n1, n2), v2) < 0:
        angle = -angle
        
    return angle

def main():
    log("=" * 50)
    log("FINAL SECONDARY STRUCTURE (MATH + FILTER)")
    log("=" * 50)
    
    atoms = get_atoms()
    pos = get_positions()
    
    if not atoms:
        log("ERROR: No structure loaded.")
        return
        
    residues = {}
    for i, atom in enumerate(atoms):
        res_id = getattr(atom, 'residue_id', getattr(atom, 'resSeq', None))
        if res_id is None:
            continue
            
        if res_id not in residues:
            residues[res_id] = {'all_indices': []}
        
        residues[res_id]['all_indices'].append(i)
        
        name = atom.name.strip()
        if name in ['N', 'CA', 'C']:
            residues[res_id][name] = pos[i]

    res_ids = list(residues.keys())

    raw_helix_idx = []
    raw_sheet_idx = []

    log("Calculating true IUPAC angles & filtering coils...")
    
    for i in range(1, len(res_ids) - 1):
        prev_res = residues[res_ids[i-1]]
        curr_res = residues[res_ids[i]]
        next_res = residues[res_ids[i+1]]
        
        if 'C' in prev_res and 'N' in curr_res and 'CA' in curr_res and 'C' in curr_res and 'N' in next_res:
            
            # Ensure chain is continuous (peptide bond ~1.33A)
            dist = np.linalg.norm(prev_res['C'] - curr_res['N'])
            if dist > 2.0:
                continue 
            
            phi = calculate_dihedral(prev_res['C'], curr_res['N'], curr_res['CA'], curr_res['C'])
            psi = calculate_dihedral(curr_res['N'], curr_res['CA'], curr_res['C'], next_res['N'])
            
            if phi is not None and psi is not None:
                # IUPAC Alpha bounds
                if -160 < phi < -20 and -110 < psi < 45:
                    raw_helix_idx.append(i)
                # IUPAC Beta bounds
                elif -180 < phi < -40 and 60 < psi < 180:
                    raw_sheet_idx.append(i)

    # --- THE CONSECUTIVE FILTER ---
    def filter_runs(indices, min_len):
        if not indices: return []
        valid = []
        run = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] == run[-1] + 1:
                run.append(indices[i])
            else:
                if len(run) >= min_len:
                    valid.extend(run)
                run = [indices[i]]
        if len(run) >= min_len:
            valid.extend(run)
        return valid

    # Alpha helices must be >= 4 residues. Beta sheets must be >= 3.
    valid_helix_idx = filter_runs(raw_helix_idx, 4)
    valid_sheet_idx = filter_runs(raw_sheet_idx, 3)

    helix_atom_indices = []
    sheet_atom_indices = []
    
    for i in valid_helix_idx:
        helix_atom_indices.extend(residues[res_ids[i]]['all_indices'])
    for i in valid_sheet_idx:
        sheet_atom_indices.extend(residues[res_ids[i]]['all_indices'])

    log(f"Verified {len(helix_atom_indices)} atoms in true Alpha Helices.")
    log(f"Verified {len(sheet_atom_indices)} atoms in true Beta Sheets.")

    n_atoms = len(atoms)
    helix_mask = np.zeros(n_atoms, dtype=bool)
    sheet_mask = np.zeros(n_atoms, dtype=bool)
    
    for idx in helix_atom_indices:
        helix_mask[idx] = True
    for idx in sheet_atom_indices:
        sheet_mask[idx] = True

    if len(helix_atom_indices) > 0:
        highlight(helix_mask, color='red')
    if len(sheet_atom_indices) > 0:
        highlight(sheet_mask, color='blue')
        
    log("DONE.")

main()