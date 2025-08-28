# Downloading and running the benchmark

## language distributions:
Total files processed: 19654
Files with detected languages: 19213
Files with unknown language (excluded): 441
Total files copied: 30

Language Distribution (before sampling, excluding unknown):
  en: 13967 files (72.7%)
  zh: 3187 files (16.6%)
  ja: 1114 files (5.8%)
  fr: 836 files (4.4%)
  la: 17 files (0.1%)
  bn: 14 files (0.1%)
  ...

We choose en, zh, ja, fr in line-44  in "orbit_scripts/filter_language.py"

```python
langid.set_languages(['en', 'zh', 'ja', 'fr'])
```


| Document Source | Text Present | Text Absent | Reading Order | Table | Math | Total | verified |
|----------------|---------------|-------------|---------------|-------|------|-------|----------|
| en             | 199           | 178         | 199           | 98     | 9   | 683   |    345 / 471 |
| zh             | 197           | 194         | 199           | 120    | 7   | 717   | |
| ja             | 197           | 200         | 197           | 121    | 10  | 725   | 234 / 397 |
| fr             | 200           | 185         | 199           | 86     | 14  | 684   | 272 / 385 |
| all            | 793           | 757         | 796           | 425    | 40  | 2809  | |

To run a benchmark, first install the bench requirements
```bash
conda create -n benchmark python=3.11 -y
conda activate benchmark

git clone https://github.com/protagolabs/protago_olmocr.git olmocr
cd olmocr

# Install olmocr and the requirements needed to run the benchmark
pip install -e .[bench]

# Configure playwright headless browser to run the math rendering tests
playwright install chromium



# Convert your documents
```bash
# You will need to install the [gpu] subset of olmocr dependencies to run gpu inference
# For actually converting the files with your own GPU
pip install olmocr[gpu]  --extra-index-url https://download.pytorch.org/whl/cu128

# Recommended: Install flash infer for faster inference on GPU
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
```

# Check the installed version
```bash
pip show olmocr
```

# convert using the same engine as olmOCR pipeline.py uses, see the olmocr/bench/runners directory for options
```bash
python -m olmocr.bench.convert olmocr_pipeline --dir ./olmOCR-bench/bench_data
```
# or use convert_all.sh to run OCR with many common frameworks all at once, API keys will be required
```bash
./olmocr/bench/scripts/convert_all.sh
```

Now run the benchmark

```python
python -m olmocr.bench.benchmark --dir ./olmOCR-bench/bench_data
```

## Results - verified


<table>
  <thead>
    <tr>
      <th align="left"><strong>Model</strong></th>
      <th align="center">absent</th>
      <th align="center">baseline</th>
      <th align="center">math</th>
      <th align="center">order</th>
      <th align="center">present</th>
      <th align="center">table</th>
      <th align="center">overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">MinerU v2.1.10 (auto)</td>
      <td align="center"> 94.1 </td>
      <td align="center">62.7</td>
      <td align="center">75.0</td>
      <td align="center"> - </td>
      <td align="center">83.8</td>
      <td align="center">16.7</td>
      <td align="center">78.6 ± 4.2</td>
    </tr>
    <tr>
      <td align="left">Marker v1.8.4</td>
      <td align="center"> 91.3 </td>
      <td align="center">59.5</td>
      <td align="center">91.7</td>
      <td align="center"> - </td>
      <td align="center">88.7</td>
      <td align="center">19.3</td>
      <td align="center">81.5 ± 2.9 </td>
    </tr>
  </tbody>
</table>

## Results - all


<table>
  <thead>
    <tr>
      <th align="left"><strong>Model</strong></th>
      <th align="center">absent</th>
      <th align="center">baseline</th>
      <th align="center">math</th>
      <th align="center">order</th>
      <th align="center">present</th>
      <th align="center">table</th>
      <th align="center">overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">Marker v1.8.4</td>
      <td align="center">80.8%</td>
      <td align="center">59.5%</td>
      <td align="center">72.3%</td>
      <td align="center">40.9%</td>
      <td align="center">70.7%</td>
      <td align="center">15.2%</td>
      <td align="center">61.2% ± 1.6%</td>
    </tr>
    <tr>
      <td align="left">MinerU v2.1.10 (auto)</td>
      <td align="center">84.9%</td>
      <td align="center">62.7%</td>
      <td align="center">66.7%</td>
      <td align="center">41.1%</td>
      <td align="center">62.2%</td>
      <td align="center">6.0%</td>
      <td align="center">58.8% ± 1.6%</td>
    </tr>
    <tr>
      <td align="left">MinerU v1.3.10 (auto)</td>
      <td align="center">83.1%</td>
      <td align="center">55.2%</td>
      <td align="center">69.3%</td>
      <td align="center">42.2%</td>
      <td align="center">62.3%</td>
      <td align="center">11.2%</td>
      <td align="center">58.3% ± 1.6%</td>
    </tr>
    <tr>
      <td align="left">Marker v1.6.2</td>
      <td align="center">80.8%</td>
      <td align="center">57.4%</td>
      <td align="center">3.0%</td>
      <td align="center">36.1%</td>
      <td align="center">60.8%</td>
      <td align="center">14.3%</td>
      <td align="center">44.6% ± 1.4%</td>
    </tr>
    <tr>
      <td align="left">olmocr_pipeline</td>
      <td align="center">21.1%</td>
      <td align="center">59.9%</td>
      <td align="center">19.0%</td>
      <td align="center">52.0%</td>
      <td align="center">71.6%</td>
      <td align="center">2.1%</td>
      <td align="center">41.2% ± 1.7%</td>
    </tr>
    <tr>
      <td align="left">dotsocr_pipeline</td>
      <td align="center">48.9%</td>
      <td align="center">59.7%</td>
      <td align="center">3.9%</td>
      <td align="center">27.7%</td>
      <td align="center">67.1%</td>
      <td align="center">26.0%</td>
      <td align="center">FAILED (errors)</td>
    </tr>
  </tbody>
</table>

## Previewing the benchmark questions

We have an internal data annotation tool that can be used to review the questions in the benchmark, and make edits.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/dd24fd88-a642-4379-b5a1-9911717bf5b1" />


```bash
python -m olmocr.bench.review_app --port 5000 --debug ./olmOCR-bench/bench_data/multi_column.jsonl --force
```