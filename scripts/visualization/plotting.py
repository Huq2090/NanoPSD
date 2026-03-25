# SPDX-License-Identifier: GPL-3.0-or-later
#
# NanoPSD: Automated Nanoparticle Shape Distribution Analysis
# Copyright (C) 2026 Md Fazlul Huq
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects


def plot_results(diameters_nm, image_path, df=None):
    """
    Generate histograms: overall + morphology-separated.

    PRESERVES: Original overall histogram
    ADDS: Morphology breakdown plots (if df provided)
    """
    base = os.path.splitext(os.path.basename(image_path))[0]

    # Overall histogram with clear bins and statistics
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate statistics
    mean_val = np.mean(diameters_nm)
    std_val = np.std(diameters_nm)
    median_val = np.median(diameters_nm)
    min_val = np.min(diameters_nm)
    max_val = np.max(diameters_nm)
    n_particles = len(diameters_nm)

    # Create SMART bin edges (nice round numbers)
    data_range = max_val - min_val

    if data_range <= 20:
        bin_width = 1  # 1 nm bins for small range
    elif data_range <= 50:
        bin_width = 2  # 2 nm bins for medium range
    else:
        bin_width = 5  # 5 nm bins for large range

    # Create bins starting from floor of min, ending at ceil of max
    bin_start = np.floor(min_val / bin_width) * bin_width
    bin_end = np.ceil(max_val / bin_width) * bin_width
    bins = np.arange(bin_start, bin_end + bin_width, bin_width)

    # Create histogram
    n, bins_edges, patches = ax.hist(
        diameters_nm,
        bins=bins,
        color="skyblue",
        edgecolor="black",
        linewidth=1.2,
        alpha=0.8,
    )

    # Add vertical line at mean
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_val:.2f} nm",
        zorder=5,
    )

    # Add vertical line at median
    ax.axvline(
        median_val,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Median = {median_val:.2f} nm",
        zorder=5,
    )

    # Set x-ticks at EVERY bin edge for clarity
    ax.set_xticks(bins)
    ax.set_xticklabels(
        [f"{int(b)}" if b == int(b) else f"{b:.1f}" for b in bins],
        rotation=45,
        ha="right",
        fontsize=16,
    )

    # Labels and title
    ax.set_xlabel("Equivalent Diameter (nm)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Particle Count", fontsize=18, fontweight="bold")
    ax.set_title(
        f"Particle Size Distribution: {base}", fontsize=20, fontweight="bold", pad=20
    )

    # Y-axis ticks
    ax.tick_params(axis="y", labelsize=16)

    # Grid aligned with bins
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.grid(axis="x", alpha=0.2, linestyle=":")

    # Add statistics text box
    # Add statistics text box (without mean/median in main box)
    stats_text = (
        f"  Statistics (n = {n_particles})  \n"
        f"  ─────────────────────────  \n"
        f"                              \n"  # Space for Mean (will overlay in red)
        f"                              \n"  # Space for Median (will overlay in blue)
        f"  Std Dev : {std_val:6.2f} nm\n"
        f"  Min     : {min_val:6.2f} nm\n"
        f"  Max     : {max_val:6.2f} nm\n"
        f"  ─────────────────────────  \n"
        f"  Bin     : {bin_width:6.0f} nm"
    )

    # Create the background text box
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85, pad=0.6),
        family="monospace",
        linespacing=1.2,
    )

    # Overlay Mean in RED
    ax.text(
        0.98,
        0.880,
        f"  Mean    : {mean_val:6.2f} nm",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        color="red",
        fontweight="bold",
        family="monospace",
    )

    # Overlay Median in BLUE
    ax.text(
        0.98,
        0.830,
        f"  Median  : {median_val:6.2f} nm",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        color="blue",
        fontweight="bold",
        family="monospace",
    )

    plt.tight_layout()
    out_path = f"outputs/figures/{base}_diameter_histogram.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Generate additional metric histograms and pie chart 
    if df is not None:
        plot_aspect_ratio_histogram(df, image_path)
        plot_circularity_histogram(df, image_path)
        plot_solidity_histogram(df, image_path)
        plot_morphology_pie_single(df, image_path)

    # Box Plot for Size Distribution (SINGLE IMAGE) - with enhanced styling and statistics
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Create box plot for particle diameters
    box_parts = ax.boxplot(
        [diameters_nm],  # Single dataset in a list
        # labels=["Particle Diameters"],
        patch_artist=True,  # Enable filling
        notch=False,
        showmeans=True,  # Show mean as a marker
        meanline=False,
    )

    # Remove x-axis tick labels (no "1" showing)
    ax.set_xticklabels([])

    # Customize appearance
    box_parts["boxes"][0].set_facecolor("lightblue")
    box_parts["boxes"][0].set_alpha(0.7)

    for whisker in box_parts["whiskers"]:
        whisker.set(linewidth=1.5, linestyle="--", color="gray")
    for cap in box_parts["caps"]:
        cap.set(linewidth=1.5, color="gray")
    for median in box_parts["medians"]:
        median.set(linewidth=2, color="red")
    for mean in box_parts["means"]:
        mean.set(
            marker="D", markerfacecolor="blue", markeredgecolor="blue", markersize=8
        )

    # Labels and formatting
    ax.set_xlabel("", fontsize=18, fontweight="bold")
    ax.set_ylabel("Equivalent Diameter (nm)", fontsize=18, fontweight="bold")
    ax.set_title(
        f"Particle Size Distribution - {base}", fontsize=20, fontweight="bold", pad=20
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="y", labelsize=16)

    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="red", linewidth=2, label="Median"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="blue",
            markeredgecolor="blue",
            markersize=8,
            label="Mean",
        ),
        Patch(facecolor="lightblue", alpha=0.7, label="25th-75th percentile (IQR)"),
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=1.5,
            linestyle="--",
            label="Whiskers (1.5×IQR)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=16, framealpha=0.9)

    # Add statistical annotations on the plot
    median_val = np.median(diameters_nm)
    q1 = np.percentile(diameters_nm, 25)
    q3 = np.percentile(diameters_nm, 75)
    mean_val = np.mean(diameters_nm)

    # Text box with statistics
    stats_text = f"n = {len(diameters_nm)}\n"
    stats_text += f"Median = {median_val:.2f} nm\n"
    stats_text += f"Mean = {mean_val:.2f} nm\n"
    stats_text += f"Q1 = {q1:.2f} nm\n"
    stats_text += f"Q3 = {q3:.2f} nm\n"
    stats_text += f"IQR = {q3-q1:.2f} nm"

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    out_path = f"outputs/figures/{base}_boxplot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_aspect_ratio_histogram(df, image_path):
    """Generate aspect ratio distribution histogram (mean + median only)."""
    base = os.path.splitext(os.path.basename(image_path))[0]
    aspect_ratios = df["Aspect_Ratio"].values

    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate statistics
    mean_val = np.mean(aspect_ratios)
    std_val = np.std(aspect_ratios)
    median_val = np.median(aspect_ratios)
    min_val = np.min(aspect_ratios)
    max_val = np.max(aspect_ratios)
    n_particles = len(aspect_ratios)

    # Smart bin selection for aspect ratio
    data_range = max_val - min_val
    if data_range <= 2:
        bin_width = 0.1
    elif data_range <= 5:
        bin_width = 0.2
    else:
        bin_width = 0.5

    bin_start = np.floor(min_val / bin_width) * bin_width
    bin_end = np.ceil(max_val / bin_width) * bin_width
    bins = np.arange(bin_start, bin_end + bin_width, bin_width)

    # Create histogram
    n, bins_edges, patches = ax.hist(
        aspect_ratios,
        bins=bins,
        color="skyblue",
        edgecolor="black",
        linewidth=1.2,
        alpha=0.8,
    )

    # Add mean line (RED)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, zorder=5)

    # Add median line (BLUE)
    ax.axvline(median_val, color="blue", linestyle="--", linewidth=2, zorder=5)

    # Set x-ticks
    ax.set_xticks(bins)
    ax.set_xticklabels([f"{b:.1f}" for b in bins], rotation=45, ha="right", fontsize=16)

    # Labels
    ax.set_xlabel("Aspect Ratio", fontsize=18, fontweight="bold")
    ax.set_ylabel("Particle Count", fontsize=18, fontweight="bold")
    ax.set_title(
        f"Aspect Ratio Distribution: {base}", fontsize=20, fontweight="bold", pad=20
    )
    ax.tick_params(axis="y", labelsize=16)

    # Grid
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.grid(axis="x", alpha=0.2, linestyle=":")

    # Statistics box
    stats_text = (
        f"  Statistics (n = {n_particles})  \n"
        f"  ─────────────────────────  \n"
        f"                              \n"
        f"                              \n"
        f"  Std Dev : {std_val:6.2f}\n"
        f"  Min     : {min_val:6.2f}\n"
        f"  Max     : {max_val:6.2f}\n"
        f"  ─────────────────────────  \n"
        f"  Bin     : {bin_width:6.2f}"
    )

    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85, pad=0.6),
        family="monospace",
        linespacing=1.4,
    )

    # Overlay Mean in RED
    ax.text(
        0.98,
        0.880,
        f"  Mean    : {mean_val:6.2f}",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        color="red",
        fontweight="bold",
        family="monospace",
    )

    # Overlay Median in BLUE
    ax.text(
        0.98,
        0.815,
        f"  Median  : {median_val:6.2f}",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        color="blue",
        fontweight="bold",
        family="monospace",
    )

    plt.tight_layout()
    out_path = f"outputs/figures/{base}_aspect_ratio_histogram.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_circularity_histogram(df, image_path):
    """Generate circularity distribution histogram (mean + median only)."""
    base = os.path.splitext(os.path.basename(image_path))[0]
    circularities = df["Circularity"].values

    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate statistics
    mean_val = np.mean(circularities)
    std_val = np.std(circularities)
    median_val = np.median(circularities)
    min_val = np.min(circularities)
    max_val = np.max(circularities)
    n_particles = len(circularities)

    # Smart bin selection (circularity is 0-1)
    bin_width = 0.05
    bins = np.arange(0, 1.05, bin_width)

    # Create histogram
    n, bins_edges, patches = ax.hist(
        circularities,
        bins=bins,
        color="skyblue",
        edgecolor="black",
        linewidth=1.2,
        alpha=0.8,
    )

    # Add mean line (RED)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, zorder=5)

    # Add median line (BLUE)
    ax.axvline(median_val, color="blue", linestyle="--", linewidth=2, zorder=5)

    # Set x-ticks
    ax.set_xticks(bins)
    ax.set_xticklabels([f"{b:.2f}" for b in bins], rotation=45, ha="right", fontsize=16)

    # Labels
    ax.set_xlabel("Circularity", fontsize=18, fontweight="bold")
    ax.set_ylabel("Particle Count", fontsize=18, fontweight="bold")
    ax.set_title(
        f"Circularity Distribution: {base}", fontsize=20, fontweight="bold", pad=20
    )
    ax.tick_params(axis="y", labelsize=16)

    # Grid
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.grid(axis="x", alpha=0.2, linestyle=":")

    # Statistics box
    stats_text = (
        f"  Statistics (n = {n_particles})  \n"
        f"  ─────────────────────────  \n"
        f"                              \n"
        f"                              \n"
        f"  Std Dev : {std_val:6.3f}\n"
        f"  Min     : {min_val:6.3f}\n"
        f"  Max     : {max_val:6.3f}\n"
        f"  ─────────────────────────  \n"
        f"  Bin     : {bin_width:6.2f}"
    )

    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85, pad=0.6),
        family="monospace",
        linespacing=1.4,
    )

    # Overlay Mean in RED
    ax.text(
        0.98,
        0.880,
        f"  Mean    : {mean_val:6.3f}",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        color="red",
        fontweight="bold",
        family="monospace",
    )

    # Overlay Median in BLUE
    ax.text(
        0.98,
        0.815,
        f"  Median  : {median_val:6.3f}",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        color="blue",
        fontweight="bold",
        family="monospace",
    )

    plt.tight_layout()
    out_path = f"outputs/figures/{base}_circularity_histogram.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_solidity_histogram(df, image_path):
    """Generate solidity distribution histogram (mean + median only)."""
    base = os.path.splitext(os.path.basename(image_path))[0]
    solidities = df["Solidity"].values

    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate statistics
    mean_val = np.mean(solidities)
    std_val = np.std(solidities)
    median_val = np.median(solidities)
    min_val = np.min(solidities)
    max_val = np.max(solidities)
    n_particles = len(solidities)

    # Smart bin selection (solidity is 0-1)
    bin_width = 0.05
    bins = np.arange(0, 1.05, bin_width)

    # Create histogram
    n, bins_edges, patches = ax.hist(
        solidities,
        bins=bins,
        color="skyblue",
        edgecolor="black",
        linewidth=1.2,
        alpha=0.8,
    )

    # Add mean line (RED)
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, zorder=5)

    # Add median line (BLUE)
    ax.axvline(median_val, color="blue", linestyle="--", linewidth=2, zorder=5)

    # Set x-ticks
    ax.set_xticks(bins)
    ax.set_xticklabels([f"{b:.2f}" for b in bins], rotation=45, ha="right", fontsize=16)

    # Labels
    ax.set_xlabel("Solidity", fontsize=18, fontweight="bold")
    ax.set_ylabel("Particle Count", fontsize=18, fontweight="bold")
    ax.set_title(
        f"Solidity Distribution: {base}", fontsize=20, fontweight="bold", pad=20
    )
    ax.tick_params(axis="y", labelsize=16)

    # Grid
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.grid(axis="x", alpha=0.2, linestyle=":")

    # Statistics box
    stats_text = (
        f"  Statistics (n = {n_particles})  \n"
        f"  ─────────────────────────  \n"
        f"                              \n"
        f"                              \n"
        f"  Std Dev : {std_val:6.3f}\n"
        f"  Min     : {min_val:6.3f}\n"
        f"  Max     : {max_val:6.3f}\n"
        f"  ─────────────────────────  \n"
        f"  Bin     : {bin_width:6.2f}"
    )

    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85, pad=0.6),
        family="monospace",
        linespacing=1.4,
    )

    # Overlay Mean in RED
    ax.text(
        0.98,
        0.880,
        f"  Mean    : {mean_val:6.3f}",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        color="red",
        fontweight="bold",
        family="monospace",
    )

    # Overlay Median in BLUE
    ax.text(
        0.98,
        0.815,
        f"  Median  : {median_val:6.3f}",
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        horizontalalignment="right",
        color="blue",
        fontweight="bold",
        family="monospace",
    )

    plt.tight_layout()
    out_path = f"outputs/figures/{base}_solidity_histogram.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_morphology_pie_single(df, image_path):
    """Generate morphology pie chart with non-overlapping percentage labels."""
    base = os.path.splitext(os.path.basename(image_path))[0]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Count morphologies
    total_spherical = len(df[df["Morphology"] == "spherical"])
    total_rodlike = len(df[df["Morphology"] == "rod-like"])
    total_aggregate = len(df[df["Morphology"] == "aggregate"])

    counts = [total_spherical, total_rodlike, total_aggregate]
    labels = ["Spherical", "Rod-like", "Aggregate"]
    colors = ["limegreen", "dodgerblue", "tomato"]

    # Create pie WITHOUT autopct (we'll add manually at different radii)
    wedges, texts = ax.pie(
        counts,
        labels=labels,
        colors=colors,
        startangle=90,
        textprops={"fontsize": 14, "fontweight": "bold"},
        explode=(0.05, 0.05, 0.05),
        labeldistance=1.15,  # Push slice labels further out
    )

    # Style the slice label text
    for text in texts:
        text.set_fontsize(14)
        text.set_fontweight("bold")

    # Manually add percentage labels at DIFFERENT radii to avoid overlap
    # Different pctdistance for each slice: [0.5, 0.65, 0.75]
    radii = [0.45, 0.65, 0.80]  # Different distances from center

    total = sum(counts)
    for i, (wedge, count, radius) in enumerate(zip(wedges, counts, radii)):
        # Calculate angle for positioning
        angle = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1

        # Convert to radians
        x = radius * np.cos(np.deg2rad(angle))
        y = radius * np.sin(np.deg2rad(angle))

        # Calculate percentage
        pct = 100 * count / total

        # Add text at custom position
        txt = ax.text(
            x,
            y,
            f"{pct:.1f}%\n(n={count})",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white",
        )

        # Add black outline for readability on ANY background
        txt.set_path_effects(
            [
                path_effects.Stroke(linewidth=3, foreground="black"),
                path_effects.Normal(),
            ]
        )

    ax.set_title(
        f"Morphology Distribution: {base}", fontsize=18, fontweight="bold", pad=20
    )

    plt.tight_layout()
    out_path = f"outputs/figures/{base}_morphology_pie.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _generate_batch_report(self) -> None:
    """
    Generate aggregate outputs for batch processing.

    Creates:
    - Combined CSV with all particles
    - Summary statistics table
    - Comparison visualizations
    """
    import pandas as pd
    from scripts.visualization.plotting import plot_batch_comparison

    logging.info(f"\n{'='*60}")
    logging.info("GENERATING BATCH REPORT")
    logging.info(f"{'='*60}")

    # Combine all dataframes
    df_all = pd.concat(self.batch_results, ignore_index=True)

    # Save combined CSV
    combined_csv = "outputs/results/batch_all_particles.csv"
    df_all.to_csv(combined_csv, index=False)
    logging.info(f"Saved combined CSV: {combined_csv}")

    # Generate summary statistics per image
    summary_data = []
    for img_name in df_all["Image"].unique():
        img_df = df_all[df_all["Image"] == img_name]

        summary_data.append(
            {
                "Image": img_name,
                "Total_Particles": len(img_df),
                "Mean_Diameter_nm": img_df["Diameter (nm)"].mean(),
                "Std_Diameter_nm": img_df["Diameter (nm)"].std(),
                "Min_Diameter_nm": img_df["Diameter (nm)"].min(),
                "Max_Diameter_nm": img_df["Diameter (nm)"].max(),
                "Spherical_Count": len(img_df[img_df["Morphology"] == "spherical"]),
                "RodLike_Count": len(img_df[img_df["Morphology"] == "rod-like"]),
                "Aggregate_Count": len(img_df[img_df["Morphology"] == "aggregate"]),
            }
        )

    df_summary = pd.DataFrame(summary_data)
    summary_csv = "outputs/results/batch_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    logging.info(f"Saved summary statistics: {summary_csv}")

    # Print summary table
    print(f"\n{'='*80}")
    print("BATCH SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(df_summary.to_string(index=False))
    print(f"{'='*80}")
    print(f"Total Images Processed: {len(df_summary)}")
    print(f"Total Particles Detected: {len(df_all)}")
    print(f"Overall Mean Diameter: {df_all['Diameter (nm)'].mean():.2f} nm")
    print(f"Overall Std Deviation: {df_all['Diameter (nm)'].std():.2f} nm")
    print(f"{'='*80}\n")

    # Generate comparison plots
    plot_batch_comparison(df_all, df_summary)

    logging.info("Batch report generation complete")
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
        ax.hist(spherical_data, bins=20, color="green", edgecolor="black", alpha=0.7)
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


def plot_batch_comparison(df_all, df_summary):
    """
    Generate comparison visualizations for batch processing.

    Creates:
    - Overlaid histograms
    - Morphology comparison bar chart
    - Summary statistics table plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    images = df_all["Image"].unique()
    n_images = len(images)

    # Figure 1: Box plot comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))  # Single subplot

    # Prepare data for box plot
    data_by_image = [
        df_all[df_all["Image"] == img]["Diameter (nm)"].values for img in images
    ]

    # Create box plot
    box_parts = ax.boxplot(
        data_by_image,
        labels=[img[:20] for img in images],  # Truncate long names to 20 chars
        patch_artist=True,  # Enable filling boxes with color
        notch=False,  # Set to True for notched boxes (shows confidence interval of median)
        showmeans=True,  # Show mean as a separate marker
        meanline=False,  # Show mean as a point (not a line)
    )

    # Customize box plot appearance
    colors = plt.cm.Set3(
        np.linspace(0, 1, n_images)
    )  # Use Set3 colormap for distinct colors
    for patch, color in zip(box_parts["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Customize other elements
    for whisker in box_parts["whiskers"]:
        whisker.set(linewidth=1.5, linestyle="--", color="gray")
    for cap in box_parts["caps"]:
        cap.set(linewidth=1.5, color="gray")
    for median in box_parts["medians"]:
        median.set(linewidth=2, color="red")
    for mean in box_parts["means"]:
        mean.set(
            marker="D", markerfacecolor="blue", markeredgecolor="blue", markersize=6
        )

    # Labels and formatting
    ax.set_xlabel("Images", fontsize=18, fontweight="bold")
    ax.set_ylabel("Equivalent Diameter (nm)", fontsize=18, fontweight="bold")
    ax.set_title(
        "Particle Size Distribution Comparison (Box Plots)",
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(fontsize=16)

    # Add legend explaining box plot elements
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="red", linewidth=2, label="Median"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="blue",
            markeredgecolor="blue",
            markersize=6,
            label="Mean",
        ),
        Patch(facecolor="lightgray", alpha=0.7, label="25th-75th percentile (IQR)"),
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=1.5,
            linestyle="--",
            label="Whiskers (1.5×IQR)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=16, framealpha=0.7)

    plt.tight_layout()
    out_path = "outputs/figures/batch_boxplot_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # =====================================================================
    # Figure 2A: Stacked Bar Chart - Morphology Distribution by Image
    # =====================================================================
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # Single subplot

    # Prepare data
    x = np.arange(n_images)
    width = 0.6

    spherical = [df_summary.iloc[i]["Spherical_Count"] for i in range(n_images)]
    rodlike = [df_summary.iloc[i]["RodLike_Count"] for i in range(n_images)]
    aggregate = [df_summary.iloc[i]["Aggregate_Count"] for i in range(n_images)]
    totals = [spherical[i] + rodlike[i] + aggregate[i] for i in range(n_images)]

    # Create stacked bars
    bars1 = ax.bar(x, spherical, width, label="Spherical", color="limegreen", alpha=0.8)
    bars2 = ax.bar(
        x,
        rodlike,
        width,
        bottom=spherical,
        label="Rod-like",
        color="dodgerblue",
        alpha=0.8,
    )
    bottom = np.array(spherical) + np.array(rodlike)
    bars3 = ax.bar(
        x, aggregate, width, bottom=bottom, label="Aggregate", color="tomato", alpha=0.8
    )

    # Add percentage labels on each segment
    for i in range(n_images):
        total = totals[i]
        if total == 0:
            continue  # Skip if no particles

        # Spherical percentage label (bottom segment)
        if spherical[i] > 0:
            pct_sph = (spherical[i] / total) * 100
            y_pos = spherical[i] / 2
            if pct_sph > 5:  # Only show label if segment is large enough
                ax.text(
                    i,
                    y_pos,
                    f"{pct_sph:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )

        # Rod-like percentage label (middle segment)
        if rodlike[i] > 0:
            pct_rod = (rodlike[i] / total) * 100
            y_pos = spherical[i] + (rodlike[i] / 2)
            if pct_rod > 5:  # Only show label if segment is large enough
                ax.text(
                    i,
                    y_pos,
                    f"{pct_rod:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )

        # Aggregate percentage label (top segment)
        if aggregate[i] > 0:
            pct_agg = (aggregate[i] / total) * 100
            y_pos = spherical[i] + rodlike[i] + (aggregate[i] / 2)
            if pct_agg > 5:  # Only show label if segment is large enough
                ax.text(
                    i,
                    y_pos,
                    f"{pct_agg:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )

        # Total count label on top of each bar
        ax.text(
            i,
            total + (max(totals) * 0.02),
            f"n={total}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    ax.set_xlabel("Image", fontsize=14, fontweight="bold")
    ax.set_ylabel("Particle Count", fontsize=14, fontweight="bold")
    ax.set_title("Morphology Distribution by Image", fontsize=18, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [img[:15] for img in images], rotation=45, ha="right", fontsize=12
    )
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(totals) * 1.1)  # Add 10% headroom for total labels

    plt.tight_layout()
    out_path = "outputs/figures/batch_morphology_stacked_bars.png"  
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # =====================================================================
    # Figure 2B: Pie Chart - Overall Morphology Distribution
    # =====================================================================
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Prepare data
    total_spherical = df_all[df_all["Morphology"] == "spherical"].shape[0]
    total_rodlike = df_all[df_all["Morphology"] == "rod-like"].shape[0]
    total_aggregate = df_all[df_all["Morphology"] == "aggregate"].shape[0]

    counts = [total_spherical, total_rodlike, total_aggregate]
    labels = ["Spherical", "Rod-like", "Aggregate"]
    colors = ["limegreen", "dodgerblue", "tomato"]

    # Create pie WITHOUT autopct (we'll add manually at different radii)
    wedges, texts = ax.pie(
        counts,
        labels=labels,
        colors=colors,
        startangle=90,
        textprops={"fontsize": 14, "fontweight": "bold"},
        explode=(0.05, 0.05, 0.05),
        labeldistance=1.15,  # Push slice labels further out
    )

    # Style the slice label text
    for text in texts:
        text.set_fontsize(14)
        text.set_fontweight("bold")

    # Manually add percentage labels at DIFFERENT radii to avoid overlap
    radii = [0.45, 0.65, 0.8]  # Different distances from center

    total = sum(counts)
    for i, (wedge, count, radius) in enumerate(zip(wedges, counts, radii)):
        # Calculate angle for positioning
        angle = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1

        # Convert to radians and calculate position
        x = radius * np.cos(np.deg2rad(angle))
        y = radius * np.sin(np.deg2rad(angle))

        # Calculate percentage
        pct = 100 * count / total

        # Add text at custom position
        txt = ax.text(
            x,
            y,
            f"{pct:.1f}%\n(n={count})",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white",
        )

        # Add black outline for readability on ANY background
        txt.set_path_effects(
            [
                path_effects.Stroke(linewidth=3, foreground="black"),
                path_effects.Normal(),
            ]
        )

    ax.set_title(
        "Overall Morphology Distribution (All Images)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    out_path = "outputs/figures/batch_morphology_pie_chart.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Figure 3: Statistics table visualization
    fig, ax = plt.subplots(figsize=(8, max(2, n_images * 0.5)))
    ax.axis("tight")
    ax.axis("off")

    # Format summary table for display
    table_data = df_summary[
        [
            "Image",
            "Total_Particles",
            "Mean_Diameter_nm",
            "Std_Diameter_nm",
            "Median_Diameter_nm",
            "Spherical_Count",
            "RodLike_Count",
            "Aggregate_Count",
        ]
    ].copy()
    table_data["Mean_Diameter_nm"] = table_data["Mean_Diameter_nm"].round(2)
    table_data["Std_Diameter_nm"] = table_data["Std_Diameter_nm"].round(2)
    table_data["Median_Diameter_nm"] = table_data["Median_Diameter_nm"].round(2)

    table = ax.table(
        cellText=table_data.values,
        colLabels=[
            "Image",
            "Total",
            "Mean (nm)",
            "Std (nm)",
            "Median (nm)",
            "Spherical",
            "Rod-like",
            "Aggregate",
        ],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # AUTO-FIT column widths to content
    table.auto_set_column_width(col=list(range(len(table_data.columns))))

    out_path = "outputs/figures/batch_summary_table.png"
    plt.savefig(
        out_path, dpi=300, bbox_inches="tight", pad_inches=0.15
    )  # Small padding
    plt.close()
    print(f"Saved: {out_path}")

    # Reduce space between title and table
    # plt.title(
    # "Batch Processing Summary Statistics", fontsize=16, pad=5
    # )  # Reduced from pad=20
