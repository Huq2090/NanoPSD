# NanoPSD
**Software Package for Analyzing Plasma-Synthesized Nanoparticle Shape Distribution**

NanoPSD is a production-ready Python package designed to extract **particle shape distributions (PSD)** of **Nanoparticles (NPs)** from **TEM/STEM images**.
It supports both **single-image** and **batch image** analysis, providing a modular and object-oriented pipeline for nanoparticle research and metrology.

---

## Features
- Automated **scale bar & text exclusion** from images
- **Manual calibration mode** for images without scale bars (direct nm/pixel input)
- **Particle segmentation** using classical methods (Otsu thresholding, preprocessing filters)
- **Particle segmentation** using classical methods (Otsu thresholding, preprocessing filters)
- **Size extraction & visualization** (histograms, plots, CSV export)
- **Classification of nanoparticles** based on **morphology**
- Works with both **single images** and **batch folders**
- Modular, **object-oriented codebase** for easy extension
- Ready for future **AI/ML-based segmentation integration**

---

## NanoPSD Pipeline
The processing workflow follows these main steps:

1. **Input Acquisition** – TEM/STEM image(s) provided as single or batch mode.
2. **Preprocessing** – Contrast enhancement (CLAHE, filters) to improve particle visibility.
3. **Segmentation** – Classical thresholding (Otsu) to identify particle regions.
4. **Scale Bar & Text Exclusion** – Automatic masking of scale bar and annotation text.
5. **Particle Size Measurement** – Extract particle sizes and compute statistics.
6. **Morphology Classification** - Classify particles based on their morphology and report morphology statistics.
7. **Visualization & Export** – Histograms, CSV tables, and segmented overlay images.

---

## Dependencies

### Core Dependencies (Required)
- `opencv-python` (≥4.5.0)
- `numpy` (≥1.21.0)
- `matplotlib` (≥3.4.0)
- `scikit-image` (≥0.18.0)
- `scipy` (≥1.7.0)
- `pandas` (≥1.3.0)
- `Pillow` (≥8.3.0)

### OCR Dependencies (Optional - only needed for automatic detection of the text mentioning scale bar size)

**Option 1: Tesseract (Recommended for CPU systems)**
```bash
# System installation required first
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
brew install tesseract              # macOS
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

# Then install Python wrapper
pip install pytesseract
```

**Option 2: EasyOCR (Requires GPU for good performance)**
```bash
pip install easyocr torch torchvision
# Note: Very slow on CPU (hours vs. seconds). Only use with CUDA GPU.
```

**Option 3: Skip OCR entirely (Recommended)**
Provide `--scale-bar-nm` parameter to the CLI manually. This is the fastest and most reliable method.

---

## Project Structure
```bash
NanoPSD/
├── README.md                  # Project overview & usage
├── requirements.txt           # Python dependencies
├── imglab_environment.yml     # Conda environment
├── nanopsd.py                 # Entry point (calls CLI & pipeline)
├── sample images              # Sample images for single image processing
│   ├── sample_image_1.tif
│   └── sample_image_2.png
├── batch_images/              # Sample folder for batch mode testing
│   ├── batch_sample_1.tif
│   ├── batch_sample_2.tif
│   └── batch_sample_3.tif
│
├── pipeline/                  # Orchestrates the full workflow
│   ├── __init__.py
│   └── analyzer.py            # NanoparticleAnalyzer class
│
├── scripts/                   # Modular processing steps
│   ├── __init__.py
│   ├── cli.py                 # Command-line argument parser
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── clahe_filter.py    # Contrast enhancement (CLAHE)
│   │
│   ├── segmentation/
│   │   ├── __init__.py
│   │   ├── base.py            # Segmentation base interface
│   │   ├── otsu_impl.py       # Otsu thresholding implementation
│   │   └── otsu_segment.py    # Segmentation workflow
│   │
│   ├── analysis/
│   │   └── size_measurement.py # Particle measurement & LaTeX export
│   │
│   └── visualization/
│       └── plotting.py        # Histogram and plot outputs
│
├── utils/                     # Helper utilities
│   ├── __init__.py
│   ├── ocr.py                 # OCR for scale bar text (EasyOCR/Tesseract)
│   └── scale_bar.py           # Scale bar detection (hybrid)
│
├── notebooks/
│   └── PSD_Interactive_Analysis.ipynb # Jupyter notebook demo
│
└── outputs/                   # Generated results & reports
    ├── debug/                 # Debug intermediate images
    ├── figures/               # Plots, overlays, batch comparisons
    │   ├── histogram.png
    │   ├── batch_histogram_comparison.png
    │   ├── batch_morphology_comparison.png
    │   └── batch_summary_table.png
    ├── preprocessed/          # Preprocessed images
    ├── results/               # .tex & CSV summaries
    │   ├── nanoparticle_data.csv
    │   ├── batch_all_particles.csv      # Combined batch data
    │   ├── batch_summary.csv            # Per-image statistics
    │   └── sample_image_*_summary.tex
    └── report.tex             # Example LaTeX report
```

---

## Project Architecture
```bash
NanoPSD/
├── nanopsd.py              # Main entry point (start here!)
├── analyzer.py             # Core pipeline orchestrator
├── cli.py                  # Command-line argument parser
│
├── Preprocessing:
│   └── clahe_filter.py     # Contrast enhancement + thresholding
│
├── Scale Bar Detection:
│   ├── scale_bar.py        # Geometric detection
│   └── ocr.py              # Text recognition (Tesseract/EasyOCR)
│
├── Segmentation:
│   ├── base.py             # Abstract interface
│   ├── otsu_impl.py        # Otsu implementation
│   └── otsu_segment.py     # Classical segmentation
│
├── Measurement & Analysis:
│   ├── size_measurement.py # Particle measurement + morphology
│   └── plotting.py         # Histograms + visualizations
│
└── Configuration:
    ├── requirements.txt
    └── imglab_environment.yml
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Huq2090/NanoPSD.git
cd NanoPSD
```

### 2. View usage examples (optional)
```bash
python3 nanopsd.py
# Displays comprehensive usage examples and common commands
```

### 3. Create and activate Conda environment
```bash
conda create -n imglab python=3.10
conda activate imglab
```

### 4. Install dependencies
```bash
conda install -c conda-forge opencv numpy matplotlib scikit-image scipy pandas pillow
```

Or recreate directly from the environment file:
```bash
conda env create -f imglab_environment.yml
conda activate imglab
```

### 5. (Optional) Install OCR dependencies
Only if you want automatic scale detection:

**For CPU systems (Tesseract):**
```bash
# System installation
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
brew install tesseract              # macOS

# Python wrapper
pip install pytesseract
```

**For GPU systems (EasyOCR):**
```bash
pip install easyocr torch torchvision
```

---

## Usage

### Determining the Scale Bar Size Parameter

**The scale bar size parameter is written on the image**

#### Method 1: Manual Reading (Recommended - Fastest & Most Reliable)

1. Open your image in ImageJ, Fiji, or any image viewer
2. Read the scale bar annotation (e.g., "200 nm")
4. Use this value with `--scale-bar-nm` parameter

**Example:**
```bash
python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --scale-bar-nm 200
```

#### Method 2: Automatic Detection (Requires OCR)

Use `--scale-bar-nm -1` to enable automatic scale bar detection:

```bash
# CPU-friendly (Tesseract)
python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --scale-bar-nm -1 --ocr-backend tesseract

# GPU-accelerated (EasyOCR - requires CUDA)
python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --scale-bar-nm -1 --ocr-backend easyocr

# With verification prompt
python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --scale-bar-nm -1 --ocr-backend tesseract --verify-scale-bar
```

**Important Notes on Automatic Detection:**
- Works best with **dark scale bars on light backgrounds**
- **White or light-colored scale bars often fail** - use manual calculation instead
- EasyOCR on CPU is extremely slow (hours) - use Tesseract or manual calculation
- Manual scale bar size parameter input is always faster and more reliable

### Method 3: Manual Calibration (For Images Without Scale Bars)

If your images **do not have scale bars** but you know the calibration factor from microscope settings:
````bash
# Provide nm-per-pixel directly
python3 nanopsd.py --mode single --input no_scalebar.tif --algo classical --min-size 3 --nm-per-pixel 2.5

# Batch mode with manual calibration
python3 nanopsd.py --mode batch --input ./images --algo classical --min-size 3 --nm-per-pixel 1.8
````

**When to use this:**
- Images without visible scale bars
- You have calibration data from microscope metadata
- Faster processing (skips scale bar detection entirely)

**How to determine nm-per-pixel:**
- Check microscope settings/metadata
- Calculate from known magnification: `nm_per_pixel = (pixel_size_µm × 1000) / magnification`
- Example: 5µm pixel at 10,000x magnification = (5 × 1000) / 10000 = 0.5 nm/pixel

**⚠️ Important for batch mode:**
- All images in the folder must have the **same calibration** (same magnification/nm-per-pixel)
- If images have different magnifications, process them in separate batches with different `--nm-per-pixel` values
- For mixed magnifications with scale bars, use `--scale-bar-nm -1` (auto-detection) instead

---

### Single Image Analysis

**Recommended (with manual scale):**
````bash
python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --scale-bar-nm 200
````

**For images without scale bars:**
````bash
python3 nanopsd.py --mode single --input no_scalebar.tif --algo classical --min-size 3 --nm-per-pixel 2.5
````

**With automatic scale detection:**
````bash
python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --scale-bar-nm -1 --ocr-backend tesseract
````

### Batch Image Analysis

Process multiple images in a folder and generate both **individual outputs** for each image and **aggregate comparisons** across all images.

**Basic batch processing:**
```bash
# With manual scale (all images must have same scale)
python3 nanopsd.py --mode batch --input ./batch_images --algo classical --min-size 3 --scale-bar-nm 200

# With automatic scale detection (each image detected separately)
python3 nanopsd.py --mode batch --input ./batch_images --algo classical --min-size 3 --scale-bar-nm -1 --ocr-backend tesseract

# For images without scale bars (known calibration)
python3 nanopsd.py --mode batch --input ./batch_images --algo classical --min-size 3 --nm-per-pixel 1.8
```
---

**Per-Image Outputs** (same as single mode):
- Individual histograms (`{image}_histogram.png`)
- Individual contour overlays (`{image}_contours.png`)
- Individual morphology overlays (`{image}_morphology_overlay.png`)
- Individual morphology charts (`{image}_morphology_*.png`)

**Aggregate Batch Outputs:**
- `batch_all_particles.csv` - Combined dataset from all images
- `batch_summary.csv` - Summary statistics per image
- `batch_histogram_comparison.png` - Side-by-side histograms
- `batch_morphology_comparison.png` - Morphology distribution comparison
- `batch_summary_table.png` - Statistical summary table

**Example batch folder:**
```bash
batch_images/
    batch_sample_1.tif
    batch_sample_2.tif
    batch_sample_3.tif
```

### OCR Backend Options

| Backend     | Hardware      | Speed                            | Accuracy  | Recommended For                       |
|-------------|---------------|----------------------------------|-----------|---------------------------------------|
| `tesseract` | CPU only      | Fast                             | Good      | **CPU systems (default)**             |
| `easyocr`   | GPU preferred | Very fast (GPU) / Very slow (CPU)| Excellent | CUDA GPU systems only                 |
| `auto`      | Automatic     | Varies                           | Varies    | Tries easyocr, falls back to tesseract|

**Performance Example:**
- Tesseract: ~2-5 seconds per image
- EasyOCR (GPU): ~3-8 seconds per image
- EasyOCR (CPU): ~300-600 seconds per image

**Recommendations:**
- **Best practice**: Input scale bar size manually, skip OCR entirely
- **CPU-only systems**: Use `--ocr-backend tesseract`
- **GPU systems (CUDA)**: Use `--ocr-backend easyocr`

---

## Command-Line Parameters

| Parameter           | Description                                     | Example                  | Required |
|---------------------|-------------------------------------------------|--------------------------|----------|
| `--mode`            | Processing mode: `single` or `batch`            | `--mode single`          | Yes      |
| `--input`           | Input image path or folder                      | `--input image.png`      | Yes      |
| `--algo`            | Segmentation algorithm: `classical`             | `--algo classical`       | Yes      |
| `--min-size`        | Minimum particle size (nm)                      | `--min-size 3`           | Yes      |
| `--scale-bar-nm`    | Scale bar size in nm or `-1` for auto-detection. Use with images that have scale bars. | `--scale-bar-nm 200` | One of these* |
| `--nm-per-pixel`    | Direct calibration (nm/pixel). Use for images WITHOUT scale bars. | `--nm-per-pixel 2.5` | One of these* |
| `--ocr-backend`     | OCR engine: `tesseract`, `easyocr`, or `auto`   | `--ocr-backend tesseract`| No       |
| `--verify-scale-bar`| Prompt user to verify detected scale            | `--verify-scale-bar`     | No       |

\* **Must provide either `--scale-bar-nm` OR `--nm-per-pixel` (not both)**

<!-- | Parameter           | Description                                     | Example                  |
|---------------------|-------------------------------------------------|--------------------------|
| `--mode`            | Processing mode: `single` or `batch`            | `--mode single`          |
| `--input`           | Input image path or folder                      | `--input image.png`      |
| `--algo`            | Segmentation algorithm: `classical`             | `--algo classical`       |
| `--min-size`        | Minimum particle size (nm)                      | `--min-size 3`           |
| `--scale-bar-nm`    | Scale bar size in nm or `-1` for auto-detection | `--scale-bar-nm 200`     |
| `--ocr-backend`     | OCR engine: `tesseract`, `easyocr`, or `auto`   | `--ocr-backend tesseract`|
| `--verify-scale-bar`| Prompt user to verify detected scale            | `--verify-scale-bar`     | -->

---

## Outputs

Analysis results are saved in the `outputs/` folder:

- **Particle size histogram** (`histogram.png`)
- **Tabulated results** (`results.csv` with particle diameters & statistics)
- **Visualization plots** (segmented overlays showing detected particles)
- **LaTeX summary** (`.tex` files for academic reports)

### Example CSV Output
```csv
Particle_ID, Diameter_nm
1, 42.5
2, 37.8
3, 56.2
...
```

### Example Statistics
- Mean diameter
- Standard deviation
- Median diameter
- Particle count
- Size distribution histogram

---

## Morphology Classification

NanoPSD automatically classifies particles into three morphological categories based on shape analysis.

### Classification Categories

| Type | Description | Criteria |
|------|-------------|----------|
| **Spherical** | Round, compact particles | Aspect ratio < 1.5, Circularity > 0.75, Solidity > 0.90 |
| **Rod-like** | Elongated particles | Aspect ratio ≥ 2.0, Smooth boundaries (Solidity > 0.85) |
| **Aggregate** | Clustered or irregular particles | Low solidity < 0.85 or irregular boundaries (Circularity < 0.60) |

### Shape Metrics

The classification uses four geometric measurements:

- **Aspect Ratio**: Major axis / Minor axis (elongation measure)
- **Circularity**: 4π × Area / Perimeter² (1.0 = perfect circle)
- **Solidity**: Area / Convex Hull Area (1.0 = smooth outline)
- **Extent**: Area / Bounding Box Area (space filling)

### Visualization

Particles are color-coded in the morphology overlay:
- **Green**: Spherical particles
- **Blue**: Rod-like particles
- **Red**: Aggregate particles

### Output Files

Morphology analysis generates additional outputs:
1. `{image}_morphology_overlay.png` - Color-coded particle contours
2. `{image}_morphology_histograms.png` - 4-panel size distributions by type
3. `{image}_morphology_pie.png` - Pie chart showing type distribution
4. `nanoparticle_data.csv` - Includes morphology classification and shape metrics

### Example Console Output
**Input Command:**

`python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --scale-bar-nm 200`

**Console Output:**
```bash
2026-01-03 22:59:23,582 [INFO] Processing: sample_image_1.tif
2026-01-03 22:59:23,582 [INFO] 📏 Scale bar detection mode
Excluded text region: (820, 2017, 145, 21)
2026-01-03 22:59:24,102 [INFO] Calibration: 0.6289 nm/pixel (bar: 318 px, value: 200.0 nm)
2026-01-03 22:59:24,154 [INFO] Excluding scale bar region from particle detection...
2026-01-03 22:59:24,157 [INFO] Using geometric text exclusion around scale bar
2026-01-03 22:59:24,157 [INFO] Excluded area: (564,2017) to (1282,2115)
2026-01-03 22:59:24,695 [INFO] Segmented 824 regions after exclusion.
Saved all contour types:
 - outputs/figures/sample_image_1_true_contours.tif
 - outputs/figures/sample_image_1_circular_equivalent.tif
 - outputs/figures/sample_image_1_elliptical_equivalent.tif
 - outputs/figures/sample_image_1_all_contour_types.tif
Morphology distribution: {'aggregate': 327, 'spherical': 37, 'rod-like': 13}
 - outputs/figures/sample_image_1_morphology_overlay.tif
2026-01-03 22:59:33,443 [INFO] Measured 377 particles (post-filter).
Saved: outputs/figures/sample_image_1_diameter_histogram.png

============================================================
MORPHOLOGY SUMMARY
============================================================
Spherical   :   37 (  9.8%)  Avg:   7.88 nm
Rod-like    :   13 (  3.4%)  Avg:   9.63 nm
Aggregate   :  327 ( 86.7%)  Avg:  13.61 nm
============================================================
2026-01-03 22:59:34,805 [INFO] Completed: sample_image_1.tif | Count=377
```

## Batch Mode Outputs

When processing multiple images in batch mode, NanoPSD generates comprehensive comparison reports.

### Combined Particle Data
**File:** `outputs/results/batch_all_particles.csv`

All particles from all images with source tracking:
```csv
Image,Particle_ID,Diameter_nm,Morphology,Aspect_Ratio,Circularity,Solidity,Extent
sample_1.png,1,42.5,Spherical,1.2,0.85,0.92,0.78
sample_2.png,1,56.2,Rod-like,2.3,0.65,0.87,0.71
```

## Troubleshooting

### Problem: Scale bar detection failed or incorrect scale detected

**Solutions:**

1. **Use manual scale calculation** (most reliable):
   ```bash
   # Open the image and read the scale bar size from the image, and fed to the CLI command:
   python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --scale-bar-nm 200
   ```

2. **Verify detected scale bar**:
   ```bash
   python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --scale-bar-nm -1 --ocr-backend tesseract --verify-scale-bar
   ```
   This will prompt you to confirm the detected scale bar.

---

### Problem: OCR taking hours to complete

**Cause**: You're using EasyOCR on a CPU-only system.

**Solutions:**

1. **Switch to Tesseract** (fastest OCR option for CPU):
   ```bash
   python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --scale-bar-nm -1 --ocr-backend tesseract
   ```

2. **Use manual scale bar text input** (no OCR needed):
   ```bash
   python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --scale-bar-nm 200
   ```

3. **If you have a GPU**, ensure PyTorch with CUDA is installed:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install easyocr
   ```

---

### Problem: "ModuleNotFoundError: No module named 'pytesseract'"

**Solution**: Install Tesseract OCR engine (system-level) + Python wrapper

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
pip install pytesseract
```

**macOS:**
```bash
brew install tesseract
pip install pytesseract
```

**Windows:**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to default location (e.g., `C:\Program Files\Tesseract-OCR`)
3. Add to PATH or set in Python:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```
4. Install Python wrapper: `pip install pytesseract`

---

### Problem: Slow performance with large batches

**Solutions:**

1. **Process in smaller batches**
2. **Use manual scale bar text** to skip OCR
3. **Consider downsampling** very high-resolution images (if scientifically appropriate)
4. **Use GPU** if available for EasyOCR

---

### Problem: Poor particle detection / segmentation

**Potential issues:**

1. **Low contrast images**: Adjust preprocessing parameters
2. **Overlapping particles**: May require manual separation or AI segmentation
3. **Non-uniform background**: Try adjusting CLAHE parameters
4. **Wrong minimum size**: Adjust `--min-size` parameter

---

### Problem: Images don't have scale bars

**Solution**: Use manual calibration mode with `--nm-per-pixel`:
```bash
python3 nanopsd.py --mode single --input image.tif --algo classical --min-size 3 --nm-per-pixel 2.5
```

**How to find nm-per-pixel:**
1. Check microscope metadata/settings
2. Calculate from magnification: `(pixel_size_µm × 1000) / magnification`
3. Use ImageJ/Fiji: Set scale manually and read calibration

**Example calculation:**
- Pixel size: 5 µm
- Magnification: 10,000x
- nm/pixel = (5 × 1000) / 10000 = **0.5 nm/pixel**

---

## Example Results

- **Raw STEM Image**
  ![STEM Raw](sample_image_1.jpg)

- **Scalebar Detection**
  ![Scalebar Detection](/docs/figures/scale_candidates.jpg)

- **Contour Overlay**
  ![Contour Overlay](/docs/figures/sample_image_1_true_contours.jpg)

- **Morphology Overlay**
  ![Morphology Overlay](/docs/figures/sample_image_1_morphology_overlay.jpg)

**Particle Size (Diameter) Histogram**
  ![Particle Size (Diameter) Histogram](/docs/figures/sample_image_1_diameter_histogram.jpg)

---

## Roadmap

- [ ] Integrate **AI-assisted segmentation** (U-Net, Mask R-CNN)
- [ ] Extend support for **TEM images with diffraction patterns**
- [ ] Jupyter Notebook integration for reproducible workflows
- [ ] Batch processing with parallel execution
- [ ] GUI interface for non-programmers

---

<!-- ## Contributing

Contributions are welcome! Please fork the repo and submit a pull request.

**Guidelines:**
- Document new features clearly
- Provide test images/examples
- Ensure PEP8 compliance
- Add unit tests where appropriate

--- -->

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

*Note: License will be updated after consultation with Prof. Davide*

---

## Citation

If you use NanoPSD in academic work, please cite:

```
Huq, M.F. (2025). NanoPSD: Automated Nanoparticle Size and Morphology Distribution Analysis
from Electron Microscopy Images. GitHub repository.
https://github.com/Huq2090/NanoPSD
```

---

## Contact & Support

- **GitHub Issues**: https://github.com/Huq2090/NanoPSD/issues
- **Author**: M.F. Huq
- **Institution**: University of Illinois at Urbana-Champaign

For questions, bug reports, or feature requests, please open an issue on GitHub.

---

## Acknowledgments

- Prof. Davide Curreli for project supervision
- [Other contributors or funding sources]

---

**Last Updated**: October 2025
