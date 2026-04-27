atoms = get_atoms()
n_tot = len(atoms)
mask = np.zeros(n_tot, dtype=bool)

target_id = 50  # The residue you want to see

for i, atom in enumerate(atoms):
    # Check if the atom belongs to the target residue
    if getattr(atom, 'residue_id', None) == target_id:
        mask[i] = True

highlight(mask, color='magenta')
log(f"Highlighted residue {target_id}")