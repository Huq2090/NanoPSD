# NanoPSD
**Software Package for Analyzing Plasma-Synthesized Nanoparticle Size Distribution**

NanoPSD is a production-ready Python package designed to extract **particle size distributions (PSD)** of **Nanoparticles (NPs)** from **SEM/TEM images**.
It supports both **single-image** and **batch image** analysis, providing a modular and object-oriented pipeline for nanoparticle research and metrology.

---

## Features
- Automated **scale bar & text exclusion** from analysis
- **Particle segmentation** using classical methods (Otsu thresholding, preprocessing filters)
- **Size extraction & visualization** (histograms, plots, CSV export)
- Works with both **single images** and **batch folders**
- Modular, **object-oriented codebase** for easy extension
- Ready for future **AI/ML-based segmentation integration**

---

## NanoPSD Pipeline
The processing workflow follows these main steps:

1. **Input Acquisition** – SEM/TEM image(s) provided as single or batch mode.
2. **Preprocessing** – Contrast enhancement (CLAHE, filters) to improve particle visibility.
3. **Segmentation** – Classical thresholding (Otsu) to identify particle regions.
4. **Scale Bar & Text Exclusion** – Automatic masking of scale bar and annotation text.
5. **Particle Measurement** – Extract particle sizes and compute statistics.
6. **Visualization & Export** – Histograms, CSV tables, and segmented overlay images.

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
# Note: Very slow on CPU (hours vs seconds). Only use with CUDA GPU.
```

**Option 3: Skip OCR entirely (Recommended)**
Calculate scale manually and use `--scale-bar-nm` parameter. This is the fastest and most reliable method.

---

## Project Structure
```bash
NanoPSD/
├── README.md                  # Project overview & usage
├── requirements.txt           # Python dependencies
├── imglab_environment.yml     # Conda environment
├── main.py                    # Entry point (calls CLI & pipeline)
├── sample_image_1.png
├── sample_image_2.png
├── sample_image_3.png
├── sample_image_4.tif
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
    ├── figures/               # Plots, overlays
    ├── preprocessed/          # Preprocessed images
    ├── results/               # .tex & CSV summaries
    │   ├── nanoparticle_data.csv
    │   └── sample_image_*_summary.tex
    └── report.tex             # Example LaTeX report
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Huq2090/NanoPSD.git
cd NanoPSD
```

### 2. Create and activate Conda environment
```bash
conda create -n imglab python=3.10
conda activate imglab
```

### 3. Install dependencies
```bash
conda install -c conda-forge opencv numpy matplotlib scikit-image scipy pandas pillow
```

Or recreate directly from the environment file:
```bash
conda env create -f imglab_environment.yml
conda activate imglab
```

### 4. (Optional) Install OCR dependencies
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

#### Method 1: Manual Reading (✅ Recommended - Fastest & Most Reliable)

1. Open your image in ImageJ, Fiji, or any image viewer with measurement tools
2. Read the scale bar annotation (e.g., "200 nm")
4. Use this value with `--scale-bar-nm` parameter

**Example:**
```bash
python main.py --mode single --input sample_image_1.png --algo classical --min-size 3 --scale-bar-nm 200
```

#### Method 2: Automatic Detection (Requires OCR)

Use `--scale-bar-nm -1` to enable automatic scale bar detection:

```bash
# CPU-friendly (Tesseract)
python main.py --mode single --input sample_image_1.png --algo classical --min-size 3 --scale-bar-nm -1 --ocr-backend tesseract

# GPU-accelerated (EasyOCR - requires CUDA)
python main.py --mode single --input sample_image_1.png --algo classical --min-size 3 --scale-bar-nm -1 --ocr-backend easyocr

# With verification prompt
python main.py --mode single --input sample_image_1.png --algo classical --min-size 3 --scale-bar-nm -1 --ocr-backend tesseract --verify-scale-bar
```

**⚠️ Important Notes on Automatic Detection:**
- Works best with **dark scale bars on light backgrounds**
- **White or light-colored scale bars often fail** - use manual calculation instead
- EasyOCR on CPU is extremely slow (hours) - use Tesseract or manual calculation
- Manual scale bar size parameter input is always faster and more reliable

---

### Single Image Analysis

**Recommended (with manual scale):**
```bash
python main.py --mode single --input sample_image_1.png --algo classical --min-size 3 --scale-bar-nm 200
```

**With automatic scale detection:**
```bash
python main.py --mode single --input sample_image_1.png --algo classical --min-size 3 --scale-bar-nm -1 --ocr-backend tesseract
```

### Batch Image Analysis

Process multiple images in a folder:

```bash
# With manual scale (all images must have same scale)
python main.py --mode batch --input ./batch_images --algo classical --min-size 3 --scale-bar-nm 200

# With automatic scale detection (each image detected separately)
python main.py --mode batch --input ./batch_images --algo classical --min-size 3 --scale-bar-nm -1 --ocr-backend tesseract
```

---

### OCR Backend Options

| Backend | Hardware | Speed | Accuracy | Recommended For |
|---------|----------|-------|----------|-----------------|
| `tesseract` | CPU only | Fast | Good | **CPU systems (default)** |
| `easyocr` | GPU preferred | Very fast (GPU) / Very slow (CPU) | Excellent | CUDA GPU systems only |
| `auto` | Automatic | Varies | Varies | Tries easyocr, falls back to tesseract |

**Performance Example:**
- Tesseract: ~2-5 seconds per image
- EasyOCR (GPU): ~3-8 seconds per image
- EasyOCR (CPU): ~300-600 seconds per image ⚠️

**Recommendations:**
- 🎯 **Best practice**: Input scale bar size manually, skip OCR entirely
- 💻 **CPU-only systems**: Use `--ocr-backend tesseract`
- 🚀 **GPU systems (CUDA)**: Use `--ocr-backend easyocr`

---

## Command-Line Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--mode` | Processing mode: `single` or `batch` | `--mode single` |
| `--input` | Input image path or folder | `--input image.png` |
| `--algo` | Segmentation algorithm: `classical` | `--algo classical` |
| `--min-size` | Minimum particle size (nm) | `--min-size 3` |
| `--scale-bar-nm` | Scale bar size in nm or `-1` for auto-detection | `--scale-bar-nm 200` |
| `--ocr-backend` | OCR engine: `tesseract`, `easyocr`, or `auto` | `--ocr-backend tesseract` |
| `--verify-scale-bar` | Prompt user to verify detected scale | `--verify-scale-bar` |

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

## Troubleshooting

### Problem: Scale bar detection failed or incorrect scale detected

**Solutions:**

1. **✅ Use manual scale calculation** (most reliable):
   ```bash
   # Measure scale bar, calculate ratio, then:
   python main.py --mode single --input image.png --algo classical --min-size 3 --scale-bar-nm 0.75
   ```

2. **White or light-colored scale bars fail detection**:
   - These are difficult for automatic detection
   - **Solution**: Always use manual calculation for light scale bars

3. **Verify detected scale**:
   ```bash
   python main.py --mode single --input image.png --algo classical --min-size 3 --scale-bar-nm -1 --ocr-backend tesseract --verify-scale-bar
   ```
   This will prompt you to confirm the detected scale value.

---

### Problem: OCR taking hours to complete

**Cause**: You're using EasyOCR on a CPU-only system.

**Solutions:**

1. **Switch to Tesseract** (fastest OCR option for CPU):
   ```bash
   python main.py --mode single --input image.png --algo classical --min-size 3 --scale-bar-nm -1 --ocr-backend tesseract
   ```

2. **Use manual scale calculation** (no OCR needed):
   ```bash
   python main.py --mode single --input image.png --algo classical --min-size 3 --scale-bar-nm 200
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
2. **Use manual scale** to skip OCR
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

## Example Results

*(You can add sample figures here)*

- **Raw SEM Image**
  ![SEM Raw](sample_image_1.png)

- **Segmented Overlay**
  *(example segmented image output)*

- **Particle Size Histogram**
  *(example histogram plot)*

---

## Roadmap

- [ ] Integrate **AI-assisted segmentation** (U-Net, Mask R-CNN)
- [ ] Extend support for **TEM images with diffraction patterns**
- [ ] Advanced morphology analysis (aspect ratio, circularity, shape factor)
- [ ] Jupyter Notebook integration for reproducible workflows
- [ ] Batch processing with parallel execution
- [ ] GUI interface for non-programmers

---

## Contributing

Contributions are welcome! Please fork the repo and submit a pull request.

**Guidelines:**
- Document new features clearly
- Provide test images/examples
- Ensure PEP8 compliance
- Add unit tests where appropriate

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

*Note: License will be updated after consultation with Prof. Davide*

---

## Citation

If you use NanoPSD in academic work, please cite:

```
Huq, M.F. (2025). NanoPSD: Automated Nanoparticle Size Distribution Analysis
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
