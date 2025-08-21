import os
from pipeline.analyzer import NanoparticleAnalyzer

os.makedirs("outputs/results", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# Define path to input SEM image and known scale bar length in nanometers
# image_path = "data/raw/SEM_nano_particles.png"
image_path = "./batch_images"  
scale_bar_length_nm = 100  # known physical length

# Select mode of analyzing images
mode = "classical"   # choices later: "classical" | "ai" | "both" | "compare"
batch = True        # True to process a folder

# Instantiate analyzer and run the full pipeline
analyzer = NanoparticleAnalyzer(image_path, scale_bar_length_nm, batch=batch, mode = mode)
analyzer.run()