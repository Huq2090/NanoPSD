import os
import matplotlib.pyplot as plt

def plot_results(diameters_nm, image_path):
    base = os.path.splitext(os.path.basename(image_path))[0]
    plt.figure(figsize=(10, 4))
    plt.hist(diameters_nm, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Diameter (nm)')
    plt.ylabel('Count')
    plt.title(f'Histogram of Nanoparticle Diameters: {base}')
    plt.grid(True)
    plt.tight_layout()
    out_path = f"outputs/figures/{base}_diameter_histogram.png"
    plt.savefig(out_path)
    plt.close()
