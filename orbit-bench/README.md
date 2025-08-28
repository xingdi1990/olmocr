# Orbit-bench

Orbit-bench is a multilingual OCR benchmark dataset containing 2,960 unit test cases across 5 language splits (English, French, Japanese, Chinese, and arXiv math papers).
This benchmark evaluates the ability of OCR systems to accurately convert PDF documents to markdown format while preserving critical textual and structural information across different languages and writing systems.

Quick links:
- 📃 [Paper](https://olmocr.allenai.org/papers/olmocr.pdf)
- 🛠️ [Code](https://github.com/allenai/olmocr)
- 🎮 [Demo](https://olmocr.allenai.org/)

## Table 1. Language Distribution

| **Language** | **Test Cases** | **Type** | **Description** |
|--------------|----------------|----------|-----------------|
| English (en) | 669 | Text Present | Financial reports and business documents |
| French (fr) | 669 | Text Present | Financial reports and business documents |
| Japanese (ja) | 714 | Text Present | Business documents with mixed scripts |
| Chinese (zh) | 708 | Text Present | Business documents with Chinese characters |
| arXiv Math | 200 | Math | Mathematical papers with LaTeX equations |
| **Total** | **2,960** | Mixed | Multilingual OCR benchmark |


## Evaluation Criteria
- Text Presence: Checks if a short text segment (1–3 sentences) is correctly identified in the OCR output. Supports fuzzy matching and positional constraints (e.g., must appear in the first/last N characters). Case-sensitive by default.
- Text Absence: Ensures specified text (e.g., headers, footers, page numbers) is excluded. Supports fuzzy matching and positional constraints. Not case-sensitive.
- Natural Reading Order: Verifies the relative order of two text spans (e.g., headline before paragraph). Soft matching enabled; case-sensitive by default.
- Table Accuracy: Confirms that specific cell values exist in tables with correct neighboring relationships (e.g., value above/below another). Supports Markdown and HTML, though complex structures require HTML.
- Math Formula Accuracy: Detects the presence of a target equation by matching symbol layout (e.g., $\int$ to the left of $x$). Based on rendered bounding boxes and relative positioning.

## Implementation

```
conda create -n benchmark python=3.11 -y
conda activate benchmark

git clone --branch v0.1.71-branch https://github.com/xingdi1990/olmocr.git olmocr
cd olmocr

# Install olmocr and the requirements needed to run the benchmark
pip install -e .[bench]

# Configure playwright headless browser to run the math rendering tests
playwright install chromium

# You will need to install the [gpu] subset of olmocr dependencies to run gpu inference
# For actually converting the files with your own GPU
pip install olmocr[gpu]  --extra-index-url https://download.pytorch.org/whl/cu128

python -m olmocr.bench.convert olmocr_pipeline --dir ./orbit-bench/bench_data

```

### 📊 Benchmark Results by Test Type

| **Model**     | absent | baseline | math  | order | present | table | overall     |
|---------------|:------:|:--------:|:-----:|:-----:|:-------:|:-----:|:-----------:|
| Magic-PDF     | 83.1   | 55.2     | 77.5  | 42.2  | 62.3    | 11.2  | 58.6 ± 1.7  |
| Marker v1.6.2| 80.8   | 57.4     | 3.5   | 36.1  | 60.8    | 14.3  | 45.0 ± 1.4  |
| Marker v1.8.3| 80.8   | 59.5     | **83.5** | 40.9 | **70.7** | 14.8 | **61.6 ± 1.5** |
| MinerU v2.1.0| **84.9** | **62.7** | 76.0 | 41.1  | 62.2    | 6.0   | 59.2 ± 1.7  |
| DOTsOCR      | 48.9   | 59.7     | 4.5   | 27.7  | 67.1    | **26.0** | FAILED   |
| olmOCR       | 2.1    | 1.8      | 3.0   | 1.4   | 1.9     | 0.7   | FAILED      |

### 📊 Benchmark Results by Language

| **Model**          | ArXiv Math | English | French | Japanese | Chinese | Overall     |
|-------------------|:----------:|:-------:|:------:|:--------:|:-------:|:-----------:|
| Magic-PDF         | 77.5       | 63.0    | 65.1   | 43.9     | 47.0    | 58.6 ± 1.7  |
| Marker v1.6.2     | 3.5        | 56.0    | 59.0   | 49.0     | 45.3    | 45.0 ± 1.4  |
| Marker v1.8.3     | **83.5**   | 60.7    | **69.6** | 49.9   | 46.5    | **61.6 ± 1.5** |
| MinerU v2.1.0     | 76.0       | **63.7**| 63.0   | 45.2     | 44.4    | 59.2 ± 1.7  |
| DOTsOCR   | 0.0        | 0.0     | 0.0    | 0.0      | 0.0     | FAILED      |
| olmOCR    | 0.0        | 0.0     | 0.0    | 0.0      | 0.0     | FAILED      |



