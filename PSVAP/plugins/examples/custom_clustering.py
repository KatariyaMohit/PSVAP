"""
Example plugin: K-means clustering of atom positions.

Clusters atoms by their 3D coordinates and logs the cluster sizes.
Requires scikit-learn (already in PSVAP requirements).

Available API:
    get_atoms()      → list of Atom objects
    get_positions()  → (N,3) numpy array
    log(msg)         → prints to console
    export(data,fn)  → saves to plugin_output/
    highlight(mask)  → recolors atoms in viewport
"""

positions = get_positions()
atoms     = get_atoms()

if len(positions) == 0:
    log("No data loaded. Open a file first.")
else:
    try:
        from sklearn.cluster import KMeans

        n_clusters = 3
        log(f"Running K-means with {n_clusters} clusters on {len(positions)} atoms...")

        km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(positions)

        sizes = [int((labels == i).sum()) for i in range(n_clusters)]
        for i, size in enumerate(sizes):
            log(f"  Cluster {i+1}: {size} atoms  "
                f"(center: {km.cluster_centers_[i].round(2).tolist()})")

        # Highlight the largest cluster
        largest_cluster = int(np.argmax(sizes))
        mask = labels == largest_cluster
        highlight(mask)
        log(f"Highlighted cluster {largest_cluster + 1} ({sizes[largest_cluster]} atoms)")

        # Export labels
        export(
            {f"atom_{i}": int(labels[i]) for i in range(len(labels))},
            "cluster_labels.json"
        )
        log("Saved cluster labels to plugin_output/cluster_labels.json")

    except ImportError:
        log("scikit-learn not available. Install: pip install scikit-learn")
    except Exception as e:
        log(f"Error: {e}")