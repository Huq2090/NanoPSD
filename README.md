# NanoPSD
**Software Package for Analyzing Nanoparticle Shape Distribution**

NanoPSD is a production-ready Python package designed to extract **particle shape distributions (PSD)** of **Nanoparticles (NPs)** from **TEM/STEM images**.
It supports both **single-image** and **batch image** analysis, providing a modular and object-oriented pipeline for nanoparticle research and metrology.

---

## Features
- Supports nanoparticle detection for both contrast polarities:
  - dark particles on light background
  - bright particles on dark background (via `--bright-particles` flag)
- Automated **scale bar & text exclusion** from images
- **Manual calibration mode** for images without scale bars (direct nm/pixel input)
- **Particle segmentation** using classical methods (Otsu thresholding, preprocessing filters)
- **Size extraction & visualization** (histograms, plots, CSV export)
- **Flexible particle filtering** with `--min-size` and `--max-size` (removes noise and false detections)
- **Pipeline visualization** for papers and presentations (`--save-preprocessing-steps`, `--save-segmentation-steps`)
- **Morphology-based nanoparticle classification** (spherical, rod-like, aggregate)
- **Morphology filtering** via `--only-morphology` (analyze only spherical, rod-like, or aggregate)
- **Customizable morphology thresholds** via optional CLI flags
- **Publication-quality plots** with enhanced font sizes and statistics
- **Comprehensive shape analysis** (aspect ratio, circularity, solidity distributions)
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

### OCR Dependencies (Optional - only needed for automatic scale bar text detection)

**EasyOCR (Supports both CPU and GPU)**
```bash
pip install easyocr torch torchvision
# GPU (CUDA): Fast performance 
# CPU: Slower but functional 
```

**Skip OCR entirely (Recommended)**
Provide `--scale-bar-nm` parameter manually. This is the fastest and most reliable method.

---

## Project Structure
```bash
NanoPSD/
├── CITATION.cff               # Guideline for citation
├── LICENSE                    # GPL-3.0 license
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
│   ├── ocr.py                 # OCR for scale bar text (EasyOCR)
│   └── scale_bar.py           # Scale bar detection (hybrid)
│
├── docs/                      # Documentation & assets
│   └── figures/               # Documentation images (README assets)
│       ├── scale_candidates.png
│       ├── sample_image_1_true_contours.jpg
│       ├── sample_image_1_morphology_overlay.jpg
│       └── sample_image_1_diameter_histogram.png
│
├── notebooks/
│   └── PSD_Interactive_Analysis.ipynb # Jupyter notebook demo
│
└── outputs/                   # Generated results & reports
    ├── debug/                 # Debug intermediate images
    ├── figures/               # Plots, overlays, batch comparisons
    │   │
    │   ├── # Single Image Outputs:
    │   ├── {image}_diameter_histogram.png           # Size distribution with statistics
    │   ├── {image}_aspect_ratio_histogram.png       # Aspect ratio distribution
    │   ├── {image}_circularity_histogram.png        # Circularity distribution
    │   ├── {image}_solidity_histogram.png           # Solidity distribution
    │   ├── {image}_morphology_pie.png               # Morphology distribution pie chart
    │   ├── {image}_boxplot.png                      # Size distribution box plot
    │   ├── {image}_morphology_histograms.png        # 4-panel morphology breakdown
    │   ├── {image}_morphology_overlay.{ext}         # Color-coded morphology overlay
    │   ├── {image}_true_contours.{ext}              # True particle contours
    │   ├── {image}_circular_equivalent.{ext}        # Circular equivalent contours
    │   ├── {image}_elliptical_equivalent.{ext}      # Elliptical fit contours
    │   ├── {image}_all_contour_types.{ext}          # Combined contour comparison
    │    ── {image_name}_true_circular.{ext}         # True contour and circular equivalent combined
    │   ├── {image_name}_morphology_overlay.{ext}    # Morphology classification overlayed
    │   │
    │   └── # Batch Mode Outputs:
    │       ├── batch_boxplot_comparison.png          # Size distribution box plots
    │       ├── batch_morphology_stacked_bars.png     # Morphology counts by image
    │       ├── batch_morphology_pie_chart.png        # Overall morphology distribution
    │       └── batch_summary_table.png               # Statistics table visualization
    │
    ├── preprocessed/          # Preprocessed images
    ├── preprocessing_steps/   # Step-by-step preprocessing (--save-preprocessing-steps)
    │   ├── {image}_step1_original.png
    │   ├── {image}_step2_normalized.png
    │   ├── {image}_step3_clahe.png
    │   ├── {image}_step4_gaussian_blur.png
    │   ├── {image}_step5_otsu_threshold.png
    │   └── {image}_step6_inverted.png
    │
    ├── segmentation_steps/    # Step-by-step segmentation (--save-segmentation-steps)
    │   ├── {image}_step1_input_binary.png
    │   ├── {image}_step2_after_small_removal.png
    │   ├── {image}_step3_after_large_removal.png
    │   ├── {image}_step4_after_hole_filling.png
    │   └── {image}_step5_labeled_components.png
    │
    ├── results/               # CSV & LaTeX summaries
    │   │
    │   ├── # Single Image Outputs:
    │   ├── {image_name}_nanoparticle_data.csv          # Per-particle detailed data
    │   ├── {image_name}_summary.csv                    # Summary statistics
    │   ├── report.tex                                  # LaTeX summary
    │   │
    │   └── # Batch Mode Outputs:
    │       ├── batch_all_particles.csv    # Combined data from all images
    │       └── batch_summary.csv          # Per-image summary statistics
    │
    └── scale_bar_debug/       # Scale bar detection visualization (optional)
        ├── scale_candidates.png
        └── scale_bar_final.png
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
│   └── ocr.py              # Text recognition (EasyOCR)
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

---

## Morphology Classification

NanoPSD classifies nanoparticles into three morphology categories based on shape descriptors:

### Classification Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Aspect Ratio (AR)** | Ratio of major to minor axis of fitted ellipse | ≥ 1.0 |
| **Circularity (C)** | `4π × Area / Perimeter²` (1.0 = perfect circle) | 0.0 - 1.0 |
| **Solidity (S)** | `Area / Convex Hull Area` (1.0 = no concavity) | 0.0 - 1.0 |

### Default Classification Thresholds

| Morphology | Conditions (Priority: Aggregate > Spherical > Rod-like) |
|------------|----------------------------------------------------------|
| **Spherical** | AR < 1.5 AND C > 0.75 AND S > 0.90 |
| **Rod-like** | AR ≥ 1.8 AND S > 0.80 |
| **Aggregate** | S < 0.85 OR C < 0.60 (fallback for unclassified) |

### Customizing Thresholds

You can adjust classification thresholds using optional CLI flags:
```bash
# Customize aspect ratio thresholds (ascending order required)
python3 nanopsd.py --mode single --input sample.tif --scale-bar-nm 200 \
    --aspect-ratio 1.4 2.0

# Customize circularity thresholds
python3 nanopsd.py --mode single --input sample.tif --scale-bar-nm 200 \
    --circularity 0.55 0.80

# Customize solidity thresholds (3 values: rodlike_min, aggregate_max, spherical_min)
python3 nanopsd.py --mode single --input sample.tif --scale-bar-nm 200 \
    --solidity 0.78 0.82 0.92

# Combine multiple threshold customizations
python3 nanopsd.py --mode single --input sample.tif --scale-bar-nm 200 \
    --aspect-ratio 1.4 2.0 \
    --circularity 0.55 0.80 \
    --solidity 0.78 0.82 0.92
```

**⚠️ Important Rules:**
- Values must be in **strict ascending order** (no equal values allowed)
- Circularity and solidity must be between 0 and 1
- Aspect ratio must be positive (typically 1.0 - 10.0)
- If not specified, default values are used

### Output Files

For each image, NanoPSD generates:

**Histograms with statistics:**
- Diameter distribution (with mean, median, std dev)
- Aspect ratio distribution
- Circularity distribution
- Solidity distribution

**Morphology visualizations:**
- Pie chart showing morphology percentages
- Color-coded overlay (blue=spherical, cyan=rod-like, magenta=aggregate)
- 4-panel morphology breakdown by particle type


## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/lcpp-org/NanoPSD.git
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

**Install EasyOCR:**
```bash
pip install easyocr torch torchvision
```

---

---

## Particle Size Filtering

NanoPSD provides two filtering parameters to control which particles are analyzed:

### `--min-size` (Minimum Particle Size)
- **Default:** 3 pixels
- **Purpose:** Remove small noise objects
- **When to adjust:**
  - Increase (5-10) for noisy images
  - Decrease (1-2) to detect very small particles

**Example:**
```bash
python3 nanopsd.py --mode single --input image.tif --scale-bar-nm 200 --min-size 5
```

### `--max-size` (Maximum Particle Size)
- **Default:** None (no upper limit)
- **Purpose:** Remove large false detections (scale bars, artifacts, large aggregates)
- **When to use:**
  - Images with large artifacts
  - To exclude particles above a certain size
  - To remove incorrectly detected regions

**Example:**
```bash
python3 nanopsd.py --mode single --input image.tif --scale-bar-nm 200 --max-size 100
```

### Combined Filtering
**Example:**
```bash
# Only analyze particles between 5 and 100 pixels in diameter
python3 nanopsd.py --mode single --input image.tif --scale-bar-nm 200 --min-size 5 --max-size 100
```

### Scale Independence
Both parameters are **in pixels**, making them **scale-independent**:
- Works for nanometer, micrometer, or millimeter scale images
- Same pixel threshold works regardless of calibration
- No need to convert between units

---

---

## Pipeline Visualization for Papers & Presentations

NanoPSD can generate step-by-step visualization of the processing pipeline, ideal for:
- 📄 **Scientific papers** - Show methodology in Methods section
- 📊 **Presentations** - Explain algorithm to audiences
- 🐛 **Debugging** - Understand why segmentation succeeded/failed
- 🎓 **Teaching** - Educational demonstrations

### Preprocessing Steps Visualization

Saves 6 intermediate images showing the preprocessing pipeline:
```bash
python3 nanopsd.py --mode single --input image.tif \
                   --scale-bar-nm 200 --min-size 3 \
                   --save-preprocessing-steps
```

**Output:** `outputs/preprocessing_steps/`
1. `*_step1_original.png` - Original grayscale image
2. `*_step2_normalized.png` - Intensity normalization (0-255)
3. `*_step3_clahe.png` - CLAHE contrast enhancement
4. `*_step4_gaussian_blur.png` - Gaussian noise reduction
5. `*_step5_otsu_threshold.png` - Otsu global thresholding
6. `*_step6_inverted.png` - Final binary mask

### Segmentation Steps Visualization

Saves 5 intermediate images showing the segmentation pipeline:
```bash
python3 nanopsd.py --mode single --input image.tif \
                   --scale-bar-nm 200 --min-size 5 --max-size 100 \
                   --save-segmentation-steps
```

**Output:** `outputs/segmentation_steps/`
1. `*_step1_input_binary.png` - Input binary mask (from preprocessing)
2. `*_step2_after_small_removal.png` - After min-size filtering
3. `*_step3_after_large_removal.png` - After max-size filtering (if used)
4. `*_step4_after_hole_filling.png` - After morphological hole filling
5. `*_step5_labeled_components.png` - Color-coded particle labels

### Combined Visualization

Generate both preprocessing and segmentation steps:
```bash
python3 nanopsd.py --mode single --input image.tif \
                   --scale-bar-nm 200 --min-size 5 \
                   --save-preprocessing-steps \
                   --save-segmentation-steps
```

**Use Case Example:**
```markdown
Figure 2 in your paper: "Image preprocessing pipeline showing progressive
refinement from raw SEM image to binary mask. Generated using NanoPSD v1.0
with --save-preprocessing-steps flag."
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
# Basic usage
python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --scale-bar-nm 200

# With minimum and maximum size filter (remove small and large false detections)
python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --max-size 100 --scale-bar-nm 200

# Generate step-by-step images for methodology visualization
python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --max-size 100 --scale-bar-nm 200 --save-preprocessing-steps --save-segmentation-steps
```

#### Method 2: Automatic Detection (Requires OCR)

Use `--ocr-backend` flag to enable automatic scale bar detection:

```bash
# CPU-friendly (easyocr-cpu)
python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --ocr-backend easyocr-cpu

# GPU-accelerated (EasyOCR - requires CUDA (fallback to CPU, if GPU is not available))
python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --ocr-backend easyocr-auto

# With minimum and maximum size filter
python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 5 --max-size 150 --ocr-backend easyocr-auto

# With verification prompt
python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --ocr-backend easyocr-auto --verify-scale-bar

# With step-by-step images for methodology visualization
python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --max-size 100 --scale-bar-nm 200 --save-preprocessing-steps --save-segmentation-steps
```

**Important Notes on Automatic Detection:**
- Works best with **dark scale bars on light backgrounds**
- **White or light-colored scale bars often fail** - use manual calculation instead
- EasyOCR on CPU is relatively slow
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
- For mixed magnifications with scale bars, use `--ocr-backend` (auto-detection) instead

---

### Contrast Polarity Option

By default, NanoPSD assumes nanoparticles appear darker than the background (dark-on-light contrast), which is common in electron microscopy images.

If nanoparticles appear brighter than the background (light-on-dark contrast), use the `--bright-particles` flag:

```bash
nanopsd input_image.png --bright-particles
```

### Morphology Filtering

To analyze only a specific particle type, use `--only-morphology`:

```bash
# Only spherical particles
python3 nanopsd.py --mode single --input sample.tif --scale-bar-nm 200 --algo classical --min-size 3 --only-morphology spherical

# Only rod-like particles
python3 nanopsd.py --mode single --input sample.tif --scale-bar-nm 200 --algo classical --min-size 3 --only-morphology rod-like

# Only aggregates
python3 nanopsd.py --mode single --input sample.tif --scale-bar-nm 200 --algo classical --min-size 3 --only-morphology aggregate
```

When active, all outputs (contour overlays, morphology overlay, histograms, CSV, and statistics) will only include the selected particle type.


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
python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --ocr-backend easyocr-auto
````

### Batch Image Analysis

Process multiple images in a folder and generate both **individual outputs** for each image and **aggregate comparisons** across all images.

**Basic batch processing:**
```bash
# With manual scale (all images must have same scale)
python3 nanopsd.py --mode batch --input ./batch_images --algo classical --min-size 3 --scale-bar-nm 200

# With automatic scale detection (each image detected separately)
python3 nanopsd.py --mode batch --input ./batch_images --algo classical --min-size 3 --ocr-backend easyocr-auto

# For images without scale bars (known calibration)
python3 nanopsd.py --mode batch --input ./batch_images --algo classical --min-size 3 --nm-per-pixel 1.8
```
---

**Per-Image Outputs** (same as single mode):
- Individual histograms (`{image}_histogram.png`)
- Individual box plots (`{image}_boxplot.png`)
- Individual contour overlays (`{image}_contours.png`)
- Individual morphology overlays (`{image}_morphology_overlay.png`)
- Individual morphology charts (`{image}_morphology_*.png`)

**Aggregate Batch Outputs:**
- `batch_all_particles.csv` - Combined dataset from all images
- `batch_summary.csv` - Summary statistics per image
- `batch_boxplot_comparison.png` - Box plot comparison showing size distributions
- `batch_morphology_stacked_bars.png` - Morphology distribution by image (stacked bar chart)
- `batch_morphology_pie_chart.png` - Overall morphology distribution (pie chart)
- `batch_summary_table.png` - Statistical summary table

**Example batch folder:**
```bash
batch_images/
    batch_sample_1.tif
    batch_sample_2.tif
    batch_sample_3.tif
```

### OCR Backend Options

| Backend        | Hardware           | Speed                    | Recommended For          |
|----------------|--------------------|--------------------------|--------------------------|
| `easyocr-auto` | GPU (fallback CPU) | GPU: Fast, CPU: Moderate | **Most users (default)** |
| `easyocr-cpu`  | CPU only           | Moderate                 | Systems without GPU      |


**Performance Example:**
- EasyOCR (GPU): ~8-12 seconds per image
- EasyOCR (CPU): ~15-25 seconds per image

**Recommendations:**
- **Best practice**: Input scale bar size manually, skip OCR entirely
- **CPU-only systems**: Use `--ocr-backend easyocr-cpu`
- **GPU systems (CUDA)**: Use `--ocr-backend easyocr-auto`

---

## Command-Line Parameters

| Parameter           | Description                                     | Example                  | Required |
|---------------------|-------------------------------------------------|--------------------------|----------|
| `--mode`            | Processing mode: `single` or `batch`            | `--mode single`          | Yes      |
| `--input`           | Input image path or folder                      | `--input image.png`      | Yes      |
| `--algo`            | Segmentation algorithm: `classical`             | `--algo classical`       | Yes      |
| `--min-size`        | Minimum particle size (nm)                      | `--min-size 3`           | Yes      |
| `--scale-bar-nm`    | Scale bar size in nm. Use with images that have scale bars. | `--scale-bar-nm 200` | One of these* |
| `--nm-per-pixel`    | Direct calibration (nm/pixel). Use for images WITHOUT scale bars. | `--nm-per-pixel 2.5` | One of these* |
| `--ocr-backend`     | OCR engine: `easyocr-auto`, or `easyocr-cpu`   | `--ocr-backend easyocr-auto`| No       |
| `--verify-scale-bar`| Prompt user to verify detected scale            | `--verify-scale-bar`     | No       |
| `--aspect-ratio`    | Aspect ratio thresholds (2 values, ascending)    | `--aspect-ratio 1.5 1.8` | No       |
| `--circularity`     | Circularity thresholds (2 values, 0-1, ascending)| `--circularity 0.60 0.75`| No       |
| `--solidity`        | Solidity thresholds (3 values, 0-1, ascending)   | `--solidity 0.80 0.85 0.90`| No       |
| `--max-size`        | Maximum particle size (pixels)                   | `--max-size 200`         | No       |
| `--save-preprocessing-steps` | Save step-by-step preprocessing images | `--save-preprocessing-steps` | No  |
| `--save-segmentation-steps`  | Save step-by-step segmentation images  | `--save-segmentation-steps`  | No  |
|  `--bright-particles` | Detect bright nanoparticles on dark background | `--bright-particles` | No  |
| `--only-morphology` | Only report results for a specific morphology type | `--only-morphology spherical` | No |

\* **Must provide either `--scale-bar-nm` OR `--nm-per-pixel` (not both)**

<!-- | Parameter           | Description                                     | Example                  |
|---------------------|-------------------------------------------------|--------------------------|
| `--mode`            | Processing mode: `single` or `batch`            | `--mode single`          |
| `--input`           | Input image path or folder                      | `--input image.png`      |
| `--algo`            | Segmentation algorithm: `classical`             | `--algo classical`       |
| `--min-size`        | Minimum particle size (nm)                      | `--min-size 3`           |
| `--scale-bar-nm`    | Scale bar size in nm                            | `--scale-bar-nm 200`     |
| `--ocr-backend`     | OCR engine: `easyocr-auto`, or `easyocr-cpu`    | `--ocr-backend easyocr-auto`|
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

### Default Classification Thresholds

| Type | Description | Default Criteria |
|------|-------------|------------------|
| **Spherical** | Round, compact particles | Aspect ratio < 1.5 AND Circularity > 0.75 AND Solidity > 0.90 |
| **Rod-like** | Elongated particles | Aspect ratio ≥ 1.8 AND Solidity > 0.80 |
| **Aggregate** | Clustered or irregular particles | Solidity < 0.85 OR Circularity < 0.60 (fallback for unclassified) |

**Classification Priority:** Aggregate > Spherical > Rod-like

### Shape Metrics

The classification uses the following geometric measurements:

| Metric | Formula | Range | Description |
|--------|---------|-------|-------------|
| **Aspect Ratio (AR)** | Major axis / Minor axis | ≥ 1.0 | Elongation measure (1.0 = circular) |
| **Circularity (C)** | 4π × Area / Perimeter² | 0.0 - 1.0 | Shape roundness (1.0 = perfect circle) |
| **Solidity (S)** | Area / Convex Hull Area | 0.0 - 1.0 | Boundary smoothness (1.0 = no concavity) |
| **Extent** | Area / Bounding Box Area | 0.0 - 1.0 | Space filling (computed but not used in classification) |

**Note:** Extent is computed and saved in output CSV files for user analysis but is not used in the built-in classification algorithm.

### Customizing Classification Thresholds

You can override default thresholds using optional command-line flags:

**Syntax:**
```bash
--aspect-ratio [spherical_max] [rodlike_min]
--circularity [aggregate_max] [spherical_min]
--solidity [rodlike_min] [aggregate_max] [spherical_min]
```

**Important Rules:**
- All values must be in **strict ascending order** (no equal values allowed)
- Circularity and solidity must be between 0.0 and 1.0
- Aspect ratio must be positive (typically 1.0 - 10.0)
- Values cannot be equal (transition gaps are required)

**Examples:**
```bash
# Stricter spherical classification
python3 nanopsd.py --mode single --input sample.tif --scale-bar-nm 200 \
    --aspect-ratio 1.3 1.9 \
    --circularity 0.65 0.82 \
    --solidity 0.82 0.87 0.93

# More lenient rod-like detection
python3 nanopsd.py --mode single --input sample.tif --scale-bar-nm 200 \
    --aspect-ratio 1.6 2.2

# Custom thresholds for gold nanospheres
python3 nanopsd.py --mode single --input gold_np.tif --scale-bar-nm 200 \
    --circularity 0.70 0.85 \
    --solidity 0.85 0.88 0.95

# Batch processing with custom thresholds
python3 nanopsd.py --mode batch --input ./images/ --scale-bar-nm 200 \
    --aspect-ratio 1.4 2.0 \
    --circularity 0.55 0.80 \
    --solidity 0.78 0.82 0.92
```

### Visualization

Particles are color-coded in the morphology overlay:
- **Blue**: Spherical particles
- **Cyan**: Rod-like particles
- **Magenta**: Aggregate particles

### Output Files

#### Single Image Mode

NanoPSD generates the following outputs for each image:

**1. Detailed Particle Data** (`{image_name}_nanoparticle_data.csv`):
```csv
Diameter (nm),Diameter (pixels),Centroid_X,Centroid_Y,Aspect_Ratio,Circularity,Solidity,Extent,Morphology
42.5,20.8,523.4,312.1,1.2,0.85,0.92,0.78,spherical
37.8,18.5,601.2,445.8,2.3,0.65,0.87,0.71,rod-like
56.2,27.5,234.7,189.3,1.8,0.54,0.71,0.65,aggregate
...
```

**2. Summary Statistics** (`{image_name}_summary.csv`):
```csv
Image,Total_Particles,Mean_Diameter_nm,Std_Diameter_nm,Median_Diameter_nm,Min_Diameter_nm,Max_Diameter_nm,Spherical_Count,RodLike_Count,Aggregate_Count
sample_image.tif,147,42.35,12.78,39.21,18.45,89.32,89,32,26
```

**3. Visualizations** (in `outputs/figures/`):

*Size Distribution:*
- `{image}_diameter_histogram.png` - Size distribution with statistics (mean, median, std dev, min, max)
- `{image}_boxplot.png` - Box plot showing median, quartiles, and outliers

*Shape Analysis:*
- `{image}_aspect_ratio_histogram.png` - Aspect ratio distribution with statistics
- `{image}_circularity_histogram.png` - Circularity distribution with statistics
- `{image}_solidity_histogram.png` - Solidity distribution with statistics

*Morphology Classification:*
- `{image}_morphology_pie.png` - Morphology distribution pie chart (percentages and counts)
- `{image}_morphology_overlay.{ext}` - Color-coded particle contours (Blue=Spherical, Cyan=Rod-like, Magenta=Aggregate)
- `{image}_morphology_histograms.png` - 4-panel morphology breakdown by particle type

*Contour Overlays:*
- `{image}_true_contours.{ext}` - True detected contours overlay
- `{image}_circular_equivalent.{ext}` - Circular equivalent diameter overlay
- `{image}_elliptical_equivalent.{ext}` - Elliptical fit overlay
- `{image}_all_contour_types.{ext}` - Combined contour visualization (all three types)

**Note:** All histograms include publication-quality statistics boxes with mean (red), median (blue), standard deviation, min, max, and bin width.

**4. LaTeX Summary** (`report.tex`):
Statistical summary table formatted for direct inclusion in scientific documents.

---

#### Batch Mode

**1. Combined Particle Data** (`batch_all_particles.csv`):

All particles from all images with source tracking:
```csv
Image,Diameter (nm),Diameter (pixels),Centroid_X,Centroid_Y,Aspect_Ratio,Circularity,Solidity,Extent,Morphology
sample_1.tif,42.5,20.8,523.4,312.1,1.2,0.85,0.92,0.78,spherical
sample_1.tif,37.8,18.5,601.2,445.8,2.3,0.65,0.87,0.71,rod-like
sample_2.tif,56.2,27.5,234.7,189.3,1.8,0.54,0.71,0.65,aggregate
...
```

**2. Batch Summary** (`batch_summary.csv`):

Per-image statistics for all processed images:
```csv
Image,Total_Particles,Mean_Diameter_nm,Std_Diameter_nm,Median_Diameter_nm,Min_Diameter_nm,Max_Diameter_nm,Spherical_Count,RodLike_Count,Aggregate_Count
sample_1.tif,147,42.35,12.78,39.21,18.45,89.32,89,32,26
sample_2.tif,203,38.92,10.45,36.78,20.12,75.43,142,45,16
sample_3.tif,189,45.67,15.23,41.55,22.34,95.67,98,54,37
```

**Key Features:**
- Comprehensive statistics for each image in the batch
- Median diameter included for robust central tendency measure
- Morphology counts enable comparison of particle type distributions
- All values rounded to 2 decimal places for readability

**3. Comparative Visualizations** (in `outputs/figures/`):
- `batch_boxplot_comparison.png` - Box plot comparison showing median, quartiles, and outliers
- `batch_morphology_stacked_bars.png` - Morphology distribution by image (stacked bar chart with percentages)
- `batch_morphology_pie_chart.png` - Overall morphology distribution across all images (pie chart with counts)
- `batch_summary_table.png` - Summary statistics table visualization

---

### Statistical Metrics

All summary files (both single and batch mode) include:
- **Total Particles**: Count of detected particles after filtering
- **Mean Diameter**: Average particle size (nm)
- **Standard Deviation**: Spread of size distribution (nm)
- **Median Diameter**: 50th percentile particle size (nm) - robust to outliers
- **Min/Max Diameter**: Size range (nm)
- **Morphology Counts**: Number of spherical, rod-like, and aggregate particles

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
Saved summary statistics: outputs/results/sample_image_1_summary.csv

============================================================
MORPHOLOGY SUMMARY
============================================================
Spherical   :   37 (  9.8%)  Avg:   7.88 nm
Rod-like    :   13 (  3.4%)  Avg:   9.63 nm
Aggregate   :  327 ( 86.7%)  Avg:  13.61 nm
============================================================
2026-01-03 22:59:34,805 [INFO] Completed: sample_image_1.tif | Count=377
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
   python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --ocr-backend easyocr-auto --verify-scale-bar
   ```
   This will prompt you to confirm the detected scale bar.

---

### Problem: OCR taking hours to complete

**Cause**: You're using EasyOCR on a CPU-only system.

**Solutions:**

1. **Use manual scale bar text input** (no OCR needed):
   ```bash
   python3 nanopsd.py --mode single --input sample_image_1.tif --algo classical --min-size 3 --scale-bar-nm 200
   ```

2. **If you have a GPU**, ensure PyTorch with CUDA is installed:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install easyocr
   ```

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
5. **Large false detections**: Use `--max-size` to filter out artifacts
```bash
   # Example: Exclude particles larger than 200 pixels
   python3 nanopsd.py --mode single --input image.tif --scale-bar-nm 200 --min-size 3 --max-size 200
```

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

---

### Problem: Morphology classification seems incorrect

**Diagnosis:** Default thresholds may not suit your specific particle type.

**Solutions:**

1. **Check current defaults:**
   - Aspect ratio: spherical < 1.5, rod-like ≥ 1.8
   - Circularity: aggregate < 0.60, spherical > 0.75
   - Solidity: rod-like > 0.80, aggregate < 0.85, spherical > 0.90

2. **View distribution histograms to understand your particles:**
```bash
   # Run analysis first
   python3 nanopsd.py --mode single --input sample.tif --scale-bar-nm 200

   # Check these output files:
   # - {image}_aspect_ratio_histogram.png
   # - {image}_circularity_histogram.png
   # - {image}_solidity_histogram.png
```
   Use these histograms to determine appropriate thresholds for your particles.

3. **Adjust thresholds based on particle type:**
```bash
   # For gold nanospheres (stricter spherical)
   python3 nanopsd.py --mode single --input gold_np.tif --scale-bar-nm 200 \
       --circularity 0.70 0.85 \
       --solidity 0.85 0.88 0.95

   # For silver nanorods (more elongated)
   python3 nanopsd.py --mode single --input silver_nr.tif --scale-bar-nm 200 \
       --aspect-ratio 1.6 2.5

   # For aggregated particles (lower solidity)
   python3 nanopsd.py --mode single --input aggregates.tif --scale-bar-nm 200 \
       --solidity 0.75 0.80 0.88
```

---

### Problem: Threshold validation error

**Error message examples:**
```
ERROR: --aspect-ratio values must be in STRICT ascending order
ERROR: --circularity values must be between 0 and 1
ERROR: --solidity values must be in STRICT ascending order
```

**Common causes and fixes:**

1. **Values not in ascending order:**
```bash
   # WRONG - descending order
   --aspect-ratio 1.8 1.5

   # CORRECT - ascending order
   --aspect-ratio 1.5 1.8
```

2. **Equal values (not allowed):**
```bash
   # WRONG - equal values
   --aspect-ratio 1.5 1.5
   --circularity 0.75 0.75

   # CORRECT - must have gap between values
   --aspect-ratio 1.5 1.8
   --circularity 0.60 0.75
```
   **Why equal values are rejected:** Transition gaps between morphology types are scientifically standard practice. Equal values would create sharp boundaries with no transition zone.

3. **Out of range values:**
```bash
   # WRONG - circularity > 1.0
   --circularity 0.60 1.2

   # CORRECT - must be 0-1
   --circularity 0.60 0.80
```

4. **Solidity requires 3 values in ascending order:**
```bash
   # WRONG - only 2 values
   --solidity 0.80 0.90

   # WRONG - not ascending
   --solidity 0.90 0.85 0.80

   # CORRECT - 3 values, ascending
   --solidity 0.80 0.85 0.90
```

**Remember:** All three morphology flags are optional. If not provided, default values are used.

## Example Results

- **Raw STEM Image**
  ![STEM Raw](sample_image_1.jpg)

- **Scalebar Detection**
  ![Scalebar Detection](/docs/figures/scale_candidates.png)

- **Contour Overlay**
  ![Contour Overlay](/docs/figures/sample_image_1_true_contours.jpg)

- **Morphology Overlay**
  ![Morphology Overlay](/docs/figures/sample_image_1_morphology_overlay.jpg)

**Particle Size (Diameter) Histogram**
  ![Particle Size (Diameter) Histogram](/docs/figures/sample_image_1_diameter_histogram.png)

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

NanoPSD is free software released under the GNU General Public License
version 3 or later (GPL-3.0-or-later).

You are free to use, modify, and redistribute this software under the
terms of the GPL, provided that any distributed modifications remain
licensed under the same terms and that the corresponding source code
is made available.

See the LICENSE file for full details.

---

## Citation

If you use NanoPSD in academic work, please cite:

```
Huq, M.F. (2026). NanoPSD: A Software for Automatic Detection of Nano-Particle Shape Distribution in Electron
Microscopy Images. 
GitHub repository.
https://github.com/lcpp-org/NanoPSD
```

---

## Contact & Support

- **GitHub Issues**: https://github.com/lcpp-org/NanoPSD/issues
- **Author**: M.F. Huq
- **Institution**: University of Illinois at Urbana-Champaign

For questions, bug reports, or feature requests, please open an issue on GitHub.

---

## Acknowledgments

- Prof. Davide Curreli for project supervision
- [Other contributors or funding sources]

---

**Last Updated**: February 2026
