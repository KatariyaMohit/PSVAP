"""
analysis/clustering.py
-----------------------
Feature 5: Small Molecule Maximum Common Substructure (MCS).
Also provides trajectory clustering utilities used by Feature 22 (Phase 6).

Public API
----------
  find_mcs(smiles_list)
      → MCSResult

  mcs_from_atoms(atoms_list)
      → MCSResult   (converts Atom lists to SMILES first)

  cluster_by_fingerprint(smiles_list, n_clusters=5)
      → dict {cluster_id: [smiles_indices]}

  MCSResult (dataclass)
      smarts, n_atoms, n_bonds, atom_map

All RDKit imports are guarded — functions raise ImportError with a clear
message if RDKit is not installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MCSResult:
    """Result of a Maximum Common Substructure search."""
    smarts:       str              # SMARTS string of the MCS
    n_atoms:      int              # number of heavy atoms in MCS
    n_bonds:      int              # number of bonds in MCS
    n_molecules:  int              # how many input molecules were matched
    atom_map:     list[list[int]]  # atom index mapping per molecule
    timed_out:    bool = False     # True if MCS search hit the time limit


def find_mcs(
    smiles_list: list[str],
    timeout: int = 60,
    ring_matches_ring_only: bool = True,
    complete_rings_only: bool = True,
) -> MCSResult:
    """
    Find the Maximum Common Substructure across a list of SMILES strings.

    Uses RDKit's rdFMCS module (Venn/Englert algorithm).

    Parameters
    ----------
    smiles_list            : list of SMILES strings
    timeout                : seconds before MCS search gives up (default 60)
    ring_matches_ring_only : ring atoms only match ring atoms (default True)
    complete_rings_only    : partial rings not allowed in MCS (default True)

    Returns
    -------
    MCSResult with SMARTS, atom/bond counts, and per-molecule atom maps

    Raises
    ------
    ImportError  if RDKit is not installed
    ValueError   if fewer than 2 valid molecules provided
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFMCS
    except ImportError:
        raise ImportError(
            "RDKit is required for MCS analysis.\n"
            "Install with: conda install -c conda-forge rdkit"
        )

    mols = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is not None:
            mols.append(mol)
            valid_smiles.append(smi.strip())

    if len(mols) < 2:
        raise ValueError(
            f"MCS requires at least 2 valid molecules. "
            f"Got {len(mols)} valid out of {len(smiles_list)} input SMILES."
        )

    # FIX: MCSParameters attribute names changed across RDKit versions.
    # In newer RDKit (2022+), RingMatchesRingOnly and CompleteRingsOnly are
    # keyword arguments to FindMCS(), not attributes of MCSParameters.
    # We try the new API first and fall back to the old attribute-based API.
    try:
        result = rdFMCS.FindMCS(
            mols,
            timeout=timeout,
            ringMatchesRingOnly=ring_matches_ring_only,
            completeRingsOnly=complete_rings_only,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
        )
    except TypeError:
        # Older RDKit — fall back to MCSParameters object
        try:
            params = rdFMCS.MCSParameters()
            params.Timeout = timeout
            try:
                params.RingMatchesRingOnly = ring_matches_ring_only
                params.CompleteRingsOnly   = complete_rings_only
            except AttributeError:
                pass  # attribute not available in this version — skip
            params.AtomCompare = rdFMCS.AtomCompare.CompareElements
            params.BondCompare = rdFMCS.BondCompare.CompareOrder
            result = rdFMCS.FindMCS(mols, params)
        except Exception:
            # Ultimate fallback — bare FindMCS with no options
            result = rdFMCS.FindMCS(mols)

    # Build per-molecule atom maps
    atom_map: list[list[int]] = []
    if result.smartsString:
        query = Chem.MolFromSmarts(result.smartsString)
        if query is not None:
            for mol in mols:
                match = mol.GetSubstructMatch(query)
                atom_map.append(list(match) if match else [])
        else:
            atom_map = [[] for _ in mols]
    else:
        atom_map = [[] for _ in mols]

    return MCSResult(
        smarts=result.smartsString or "",
        n_atoms=result.numAtoms,
        n_bonds=result.numBonds,
        n_molecules=len(mols),
        atom_map=atom_map,
        timed_out=result.canceled,
    )

def cluster_by_fingerprint(
    smiles_list: list[str],
    n_clusters: int = 5,
    radius: int = 2,
    n_bits: int = 2048,
) -> dict[int, list[int]]:
    """
    Cluster molecules by Morgan fingerprint similarity using k-means.

    Parameters
    ----------
    smiles_list : list of SMILES strings
    n_clusters  : number of clusters
    radius      : Morgan fingerprint radius (default 2 = ECFP4)
    n_bits      : fingerprint bit length

    Returns
    -------
    dict {cluster_id: [indices into smiles_list]}

    Raises
    ------
    ImportError if RDKit or scikit-learn not installed
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        import numpy as np
    except ImportError:
        raise ImportError("RDKit is required for fingerprint clustering.")

    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("scikit-learn is required for clustering.")

    fps = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None:
            continue
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(list(fp))
        valid_indices.append(i)

    if len(fps) < n_clusters:
        # Fewer molecules than clusters — each molecule is its own cluster
        return {i: [valid_indices[i]] for i in range(len(fps))}

    import numpy as np
    X = np.array(fps, dtype=np.float32)
    km = KMeans(n_clusters=min(n_clusters, len(fps)), random_state=42, n_init=10)
    labels = km.fit_predict(X)

    clusters: dict[int, list[int]] = {}
    for local_idx, cluster_id in enumerate(labels):
        cid = int(cluster_id)
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(valid_indices[local_idx])
    return clusters


def smiles_from_sdf(sdf_path: str) -> list[str]:
    """
    Read all molecules from an SDF file and return their SMILES strings.

    Raises
    ------
    ImportError if RDKit not installed
    FileNotFoundError if path does not exist
    """
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("RDKit is required for SDF reading.")

    from pathlib import Path
    path = Path(sdf_path)
    if not path.exists():
        raise FileNotFoundError(f"SDF file not found: {path}")

    supplier = Chem.SDMolSupplier(str(path))
    smiles_list = []
    for mol in supplier:
        if mol is not None:
            smi = Chem.MolToSmiles(mol)
            if smi:
                smiles_list.append(smi)
    return smiles_list

# ── Trajectory Clustering (Feature 22) ───────────────────────────────────

from dataclasses import dataclass as _dataclass

@_dataclass
class TrajectoryClusterResult:
    """Result of trajectory clustering."""
    labels:          list[int]       # cluster label per frame (0-indexed)
    n_clusters:      int
    medoid_indices:  list[int]       # index of representative frame per cluster
    cluster_sizes:   list[int]       # number of frames in each cluster
    inertia:         float           # total within-cluster RMSD sum
    method:          str             # 'kmeans' or 'hierarchical'


def cluster_trajectory(
    trajectory: list,
    atom_indices: list[int] | None = None,
    n_clusters: int = 5,
    method: str = "kmeans",
    align_first: bool = True,
    max_frames: int = 200,
) -> TrajectoryClusterResult:
    """
    Cluster trajectory frames by structural similarity (RMSD-based).

    Reduces thousands of frames to a small set of representative
    conformations, revealing dominant states sampled during simulation.

    Algorithm:
      1. Optionally align all frames to frame 0 (Kabsch superposition).
      2. Flatten each frame's coordinates into a feature vector.
      3. Apply k-means or agglomerative clustering in coordinate space.
         (Note: true RMSD-matrix clustering is O(N²) — too slow for
         large trajectories; coordinate-space clustering is O(N) and
         gives equivalent results after alignment.)
      4. Find the medoid (most representative frame) of each cluster.

    Parameters
    ----------
    trajectory   : list of (N, 3) position arrays
    atom_indices : subset of atoms to use (e.g. backbone CA only)
    n_clusters   : number of clusters (default 5)
    method       : 'kmeans' or 'hierarchical'
    align_first  : superimpose all frames to frame 0 before clustering
    max_frames   : subsample trajectory to this many frames if longer

    Returns
    -------
    TrajectoryClusterResult

    Raises
    ------
    ImportError if scikit-learn not installed
    ValueError  if fewer than n_clusters frames provided
    """
    try:
        import numpy as np
        from sklearn.cluster import KMeans, AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError(
            "scikit-learn is required for trajectory clustering.\n"
            "Install with: pip install scikit-learn"
        )

    import numpy as np

    if len(trajectory) < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} frames for {n_clusters} clusters. "
            f"Got {len(trajectory)}."
        )

    # Subsample if trajectory is very long
    n_frames = len(trajectory)
    if n_frames > max_frames:
        step = n_frames // max_frames
        sub_idx  = list(range(0, n_frames, step))[:max_frames]
        sub_traj = [trajectory[i] for i in sub_idx]
    else:
        sub_idx  = list(range(n_frames))
        sub_traj = trajectory

    # Align to frame 0 if requested
    if align_first and len(sub_traj) > 1:
        try:
            from PSVAP.analysis.alignment import superimpose_trajectory
            ref = np.asarray(sub_traj[0], dtype=float)
            sub_traj = superimpose_trajectory(sub_traj, ref, atom_indices)
        except Exception:
            pass   # alignment failed — proceed without

    # Extract feature vectors (flattened coordinates)
    frames_arr = []
    for frame in sub_traj:
        f = np.asarray(frame, dtype=float)
        if atom_indices is not None:
            f = f[atom_indices]
        frames_arr.append(f.ravel())

    X = np.array(frames_arr, dtype=np.float64)

    # Cluster
    k = min(n_clusters, len(X))
    mt = method.lower()

    if mt == "hierarchical":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels_sub = model.fit_predict(X).tolist()
        inertia = 0.0
    else:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        labels_sub = model.labels_.tolist()
        inertia    = float(model.inertia_)

    # Map subsampled labels back to full trajectory
    if n_frames > max_frames:
        # Assign remaining frames to nearest cluster by coordinate distance
        full_labels = [0] * n_frames
        centroids_per_cluster: dict[int, np.ndarray] = {}
        for cid in range(k):
            member_frames = [X[i] for i, l in enumerate(labels_sub) if l == cid]
            if member_frames:
                centroids_per_cluster[cid] = np.mean(member_frames, axis=0)

        for orig_idx, frame in enumerate(trajectory):
            f = np.asarray(frame, dtype=float)
            if atom_indices is not None:
                f = f[atom_indices]
            fv = f.ravel()
            best_cid   = 0
            best_dist  = float("inf")
            for cid, centroid in centroids_per_cluster.items():
                if len(centroid) == len(fv):
                    d = float(np.linalg.norm(fv - centroid))
                    if d < best_dist:
                        best_dist = d
                        best_cid  = cid
            full_labels[orig_idx] = best_cid
    else:
        full_labels = labels_sub

    # Find medoids — frame closest to cluster centroid
    medoid_indices: list[int] = []
    cluster_sizes:  list[int] = []

    for cid in range(k):
        member_orig_idx = [i for i, l in enumerate(full_labels) if l == cid]
        cluster_sizes.append(len(member_orig_idx))

        if not member_orig_idx:
            medoid_indices.append(0)
            continue

        # Compute centroid of cluster members in subsampled space
        member_fvs = []
        for orig_idx in member_orig_idx:
            f = np.asarray(trajectory[orig_idx], dtype=float)
            if atom_indices is not None and len(f) > max(atom_indices):
                f = f[atom_indices]
            member_fvs.append(f.ravel())

        if not member_fvs:
            medoid_indices.append(member_orig_idx[0])
            continue

        # Handle variable-length feature vectors safely
        min_len = min(len(v) for v in member_fvs)
        member_fvs = [v[:min_len] for v in member_fvs]
        centroid = np.mean(member_fvs, axis=0)

        best_dist  = float("inf")
        best_frame = member_orig_idx[0]
        for local_i, orig_idx in enumerate(member_orig_idx):
            d = float(np.linalg.norm(member_fvs[local_i] - centroid))
            if d < best_dist:
                best_dist  = d
                best_frame = orig_idx
        medoid_indices.append(best_frame)

    return TrajectoryClusterResult(
        labels=full_labels,
        n_clusters=k,
        medoid_indices=medoid_indices,
        cluster_sizes=cluster_sizes,
        inertia=inertia,
        method=method,
    )


def format_cluster_result(result: TrajectoryClusterResult) -> str:
    """Format trajectory clustering results as text for GUI display."""
    lines = [
        f"TRAJECTORY CLUSTERING  ({result.method.upper()})\n",
        f"  Clusters  : {result.n_clusters}",
        f"  Method    : {result.method}",
        f"  Inertia   : {result.inertia:.2f}" if result.inertia > 0 else "",
        "",
        f"{'CLUSTER':>8}  {'FRAMES':>8}  {'MEDOID':>8}  {'FRACTION':>10}",
        "-" * 42,
    ]
    total = sum(result.cluster_sizes)
    for cid in range(result.n_clusters):
        size     = result.cluster_sizes[cid]
        medoid   = result.medoid_indices[cid]
        fraction = size / max(total, 1)
        lines.append(
            f"{cid+1:>8}  {size:>8}  {medoid:>8}  {fraction:>9.1%}"
        )

    lines.append(f"\nMEDOID FRAMES (most representative):")
    for cid, medoid in enumerate(result.medoid_indices):
        lines.append(f"  Cluster {cid+1}: frame {medoid}")

    return "\n".join(l for l in lines if l is not None)