import os
import matplotlib.pyplot as plt
import numpy as np


def plot_results(diameters_nm, image_path, df=None):
    """
    Generate histograms: overall + morphology-separated.

    PRESERVES: Original overall histogram
    ADDS: Morphology breakdown plots (if df provided)
    """
    base = os.path.splitext(os.path.basename(image_path))[0]

    # === EXISTING: Overall histogram (UNCHANGED) ===
    plt.figure(figsize=(10, 4))
    plt.hist(diameters_nm, bins=30, color="skyblue", edgecolor="black")
    plt.xlabel("Diameter (nm)")
    plt.ylabel("Count")
    plt.title(f"Histogram of Nanoparticle Diameters: {base}")
    plt.grid(True)
    plt.tight_layout()
    out_path = f"outputs/figures/{base}_diameter_histogram.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    # === NEW: Morphology-specific plots ===
    if df is not None and "Morphology" in df.columns:
        # Figure 2: 4-panel morphology histograms
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Morphology Analysis - {base}", fontsize=16)

        colors = {"spherical": "green", "rod-like": "blue", "aggregate": "red"}

        # Panel 1: Overlay all types
        ax = axes[0, 0]
        for morph in ["spherical", "rod-like", "aggregate"]:
            data = df[df["Morphology"] == morph]["Diameter (nm)"]
            if len(data) > 0:
                ax.hist(
                    data,
                    bins=20,
                    alpha=0.6,
                    label=morph.capitalize(),
                    color=colors[morph],
                    edgecolor="black",
                )
        ax.set_xlabel("Diameter (nm)")
        ax.set_ylabel("Count")
        ax.set_title("All Particles (Color-coded)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Panel 2: Spherical only
        ax = axes[0, 1]
        spherical_data = df[df["Morphology"] == "spherical"]["Diameter (nm)"]
        if len(spherical_data) > 0:
            ax.hist(
                spherical_data, bins=20, color="green", edgecolor="black", alpha=0.7
            )
            ax.set_title(f"Spherical (n={len(spherical_data)})")
        else:
            ax.text(0.5, 0.5, "No spherical particles", ha="center", va="center")
        ax.set_xlabel("Diameter (nm)")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

        # Panel 3: Rod-like only
        ax = axes[1, 0]
        rod_data = df[df["Morphology"] == "rod-like"]["Diameter (nm)"]
        if len(rod_data) > 0:
            ax.hist(rod_data, bins=20, color="blue", edgecolor="black", alpha=0.7)
            ax.set_title(f"Rod-like (n={len(rod_data)})")
        else:
            ax.text(0.5, 0.5, "No rod-like particles", ha="center", va="center")
        ax.set_xlabel("Diameter (nm)")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

        # Panel 4: Aggregate only
        ax = axes[1, 1]
        agg_data = df[df["Morphology"] == "aggregate"]["Diameter (nm)"]
        if len(agg_data) > 0:
            ax.hist(agg_data, bins=20, color="red", edgecolor="black", alpha=0.7)
            ax.set_title(f"Aggregate (n={len(agg_data)})")
        else:
            ax.text(0.5, 0.5, "No aggregate particles", ha="center", va="center")
        ax.set_xlabel("Diameter (nm)")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out_path = f"outputs/figures/{base}_morphology_histograms.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")

        # Figure 3: Pie chart
        plt.figure(figsize=(8, 8))
        morph_counts = df["Morphology"].value_counts()
        colors_list = [colors[m] for m in morph_counts.index]
        plt.pie(
            morph_counts,
            labels=[m.capitalize() for m in morph_counts.index],
            autopct="%1.1f%%",
            colors=colors_list,
            startangle=90,
        )
        plt.title(f"Morphology Distribution - {base}", fontsize=14)

        out_path = f"outputs/figures/{base}_morphology_pie.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved: {out_path}")
