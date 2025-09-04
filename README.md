<div align="center">
  <!-- <img src="https://github.com/allenai/OLMo/assets/8812459/774ac485-a535-4768-8f7c-db7be20f5cc3" width="300"/> -->
<img src="https://github.com/user-attachments/assets/d70c8644-3e64-4230-98c3-c52fddaeccb6" alt="olmOCR Logo" width="300"/>
<hr/>
</div>
<p align="center">
  <a href="https://github.com/allenai/OLMo/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/OLMo">
  </a>
  <a href="https://github.com/allenai/olmocr/releases">
    <img alt="GitHub release" src="https://img.shields.io/github/release/allenai/olmocr.svg">
  </a>
  <a href="https://olmocr.allenai.org/papers/olmocr.pdf">
    <img alt="Tech Report" src="https://img.shields.io/badge/Paper-olmOCR-blue">
  </a>
  <a href="https://olmocr.allenai.org">
    <img alt="Demo" src="https://img.shields.io/badge/Ai2-Demo-F0529C">
  </a>
  <a href="https://discord.gg/sZq3jTNVNG">
    <img alt="Discord" src="https://img.shields.io/badge/Discord%20-%20blue?style=flat&logo=discord&label=Ai2&color=%235B65E9">
  </a>
</p>

A toolkit for converting PDFs and other image-based document formats into clean, readable, plain text format.

Try the online demo: [https://olmocr.allenai.org/](https://olmocr.allenai.org/)

Features:
 - Convert PDF, PNG, and JPEG based documents into clean Markdown
 - Support for equations, tables, handwriting, and complex formatting
 - Automatically removes headers and footers
 - Convert into text with a natural reading order, even in the presence of
   figures, multi-column layouts, and insets
 - Efficient, less than $200 USD per million pages converted
 - (Based on a 7B parameter VLM, so it requires a GPU)

### News
 - August 13, 2025 - v0.3.0 - [New model release](https://huggingface.co/allenai/olmOCR-7B-0825-FP8), fixes auto-rotation detection, and hallucinations on blank documents.
 - July 24, 2025 - v0.2.1 - [New model release](https://huggingface.co/allenai/olmOCR-7B-0725-FP8), scores 3 points higher on [olmOCR-Bench](https://github.com/allenai/olmocr/tree/main/olmocr/bench), also runs significantly faster because it's default FP8, and needs much fewer retries per document.
 - July 23, 2025 - v0.2.0 - New cleaned up [trainer code](https://github.com/allenai/olmocr/tree/main/olmocr/train), makes it much simpler to train olmOCR models yourself.
 - June 17, 2025 - v0.1.75 - Switch from sglang to vllm based inference pipeline, updated docker image to CUDA 12.8.
 - May 23, 2025 - v0.1.70 - Official docker support and images are now available! [See Docker usage](#using-docker)
 - May 19, 2025 - v0.1.68 - [olmOCR-Bench](https://github.com/allenai/olmocr/tree/main/olmocr/bench) launch, scoring 77.4. Launch includes 2 point performance boost in olmOCR pipeline due to bug fixes with prompts.
 - Mar 17, 2025 - v0.1.60 - Performance improvements due to better temperature selection in sampling.
 - Feb 25, 2025 - v0.1.58 -  Initial public launch and demo.

### Benchmark

**olmOCR-Bench**:
We also ship a comprehensive benchmark suite covering over 7,000 test cases across 1,400 documents to help measure performance of OCR systems. 

| Model                            | ArXiv | Old Scans Math | Tables | Old Scans | Headers and Footers | Multi column | Long tiny text | Base | Overall        |
| :------------------------------- | :---: | :------------: | :----: | :-------: | :-----------------: | :----------: | :------------: | :--: | :------------: |
| Marker v1.8.3 (base, force_ocr)  | 75.2  |      60.9      |  52.8  |   28.3    |        85.1         |     72.5     |      84.8      | 98.9 | 69.8 ± 1.0    |
| Marker v1.7.5 (base, force_ocr)  | 76.0  |      57.9      |  57.6  |   27.8    |        84.9         |     72.9     |      84.6      | 99.1 | 70.1 ± 1.1    |
| MinerU v2.1.10                   | 75.7  |      48.9      |  57.6  |   19.2    |     **97.5**        |     58.9     |      42.1      | 97.6 | 62.2 ± 1.0    |
| MinerU v1.3.10                   | 75.4  |      47.4      |  60.9  |   17.3    |        96.6         |     59.0     |      39.1      | 96.6 | 61.5 ± 1.1    |
| Mistral OCR API                  | 77.2  |      67.5      |  60.6  |   29.3    |        93.6         |     71.3     |      77.1      | 99.4 | 72.0 ± 1.1    |
| olmOCR v0.1.75 (Anchored)        | 74.9  |      71.2      |  71.0  |   42.2    |        94.5         |     78.3     |      73.3      | 98.3 | 75.5 ± 1.0    |
| olmOCR v0.2.0                    | 78.8  |      77.5      |  71.9  | **45.4**  |        94.2         |     78.6     |      81.4      | **99.8** | 78.5 ± 1.1 |
| olmOCR v0.3.0                    | 78.6  |   **79.9**     |  72.9  |   43.9    |        95.1         |     77.3     |      81.2      | 98.9 | 78.5 ± 1.1    |
| dotsocr_nohf *                   | **82.6** |      68.3      |  **87.1**  |   42.6    |        90.3         |   **82.7**   |   **82.1**    | 99.2 | 79.4 ± 1.0    |
| -------------------------------- | ------ | -------------- | ------ | -------- | ------------------- | ------------ | -------------- | ---- | ------------- |
| dots.ocr                         | 82.1  |      64.2      | 88.3|   40.9    |        94.1         |     82.4     |      81.2      | 99.5 | 79.1 ± 1.0    |



| Model             | absent | baseline |   math   |   order  |  present |   table  |   overall  |
| :---------------- | :----: | :------: | :------: | :------: | :------: | :------: | :--------: |
| marker v1.7.5     |  83.8  |   99.0   |   73.7   |   64.4   |   60.1   |   57.0   | 70.2 ± 1.1 |
| marker v1.8.3     |  84.0  |   98.9   |   73.2   |   63.9   |   60.5   |   52.7   | 69.8 ± 1.1 |
| magic-pdf v1.3.10 |  96.4  |   96.4   |   71.6   |   49.8   |   26.9   |   60.5   | 61.5 ± 1.1 |
| mineru v2.1.10    |  97.2  |   97.6   |   72.1   |   49.8   |   29.7   |   57.5   | 62.2 ± 1.1 |
| olmOCR v0.3.0     |  96.4  |   98.9   |   78.7   |   68.9   |   65.6   |   71.3   | 78.3 ± 1.0 |
| dotsocr\_nohf *    |  90.2  | **99.2** | **80.7** | **73.8** | **65.5** | **87.1** |  **80.2 ± 1.0**  |

* dotsocr\_nohf fails on missing markdown on 9 files, the results are reported with those ignores.

**orbit-Bench** 

## Results - all

### Language Breakdown

| Model             |   en   |   fr   |   ja   |   zh   | arxiv_math | baseline |    overall    |
| :---------------- | :----: | :----: | :----: | :----: | :--------: | :------: | :-----------: |
| Marker v1.8.3     | 61.0  | 69.6  | **49.9** | 46.5  | **83.5** | 59.5   | 61.7 ± 1.6 |
| Marker v1.7.5     | 60.4  | 69.3  | **49.9** | 46.1  | 77.0     | 59.6   | 60.4 ± 1.7 |
| MinerU v2.1.10    | 63.7  | 63.0  | 45.2  | 44.4  | 76.0     | **62.7** | 59.2 ± 1.7 |
| MinerU v1.3.10    | 63.0  | 65.1  | 43.9  | 47.0  | 77.5     | 55.2   | 58.6 ± 1.7 |
| olmocr v0.3.0     | **72.2** | **73.0** | 48.8  | **58.7** | 78.0     | 61.7   | 65.4 ± 1.6 |
| dotsocr_nohf      | 70.0  | 72.7  | **52.7** | 56.1  | **84.0** | 59.8   | **65.9 ± 1.5** |
<!-- | Marker v1.6.2     | 56.0  | 59.0  | 49.0  | 45.3  |  3.5     | 57.4   | 45.0 ± 1.5 | -->

---

### Category Breakdown

| Model             | absent | baseline |  math  | order | present | table |   overall    |
| :---------------- | :----: | :------: | :----: | :---: | :-----: | :---: | :----------: |
| Marker v1.8.3     | 80.8  |  59.5   | **83.5** | 40.9 |  70.7  | 15.2 | 61.7 ± 1.6 |
| Marker v1.7.5     | 81.4  |  59.6   | 77.0  | 40.6 |  70.2  | 13.8 | 60.4 ± 1.7 |
| MinerU v2.1.10    | 84.9  | **62.7** | 76.0  | 41.1 |  62.2  |  6.0 | 59.2 ± 1.7 |
| MinerU v1.3.10    | 83.1  |  55.2   | 77.5  | 42.2 |  62.3  | 11.2 | 58.6 ± 1.7 |
| olmocr v0.3.0     | **91.3** |  61.7   | 78.0  | **43.2** | **72.6** | 30.5 | 65.4 ± 1.6 |
| dotsocr_nohf      | 85.5  |  59.8   | **84.0** | 42.1 |  72.1  | **42.4** | **65.9 ± 1.5** |
<!-- | Marker v1.6.2     | 80.8  |  57.4   |  3.5  | 36.1 |  60.8  | 14.3 | 45.0 ± 1.5 | --> 




### Installation

Requirements:
 - Recent NVIDIA GPU (tested on RTX 4090, L40S, A100, H100) with at least 15 GB of GPU RAM
 - 30GB of free disk space

You will need to install poppler-utils and additional fonts for rendering PDF images.

Install dependencies (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools
```

Set up a conda environment and install olmocr. The requirements for running olmOCR
are difficult to install in an existing python environment, so please do make a clean python environment to install into.
```bash
conda create -n olmocr python=3.11
conda activate olmocr

# For CPU-only operations, ex running the benchmark
pip install olmocr[bench]

# For actually converting the files with your own GPU
pip install olmocr[gpu]  --extra-index-url https://download.pytorch.org/whl/cu128

# Recommended: Install flash infer for faster inference on GPU
pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl
```

### Local Usage Example

For quick testing, try the [web demo](https://olmocr.allen.ai/). To run locally, a GPU is required, as inference is powered by [sglang](https://github.com/sgl-project/sglang) under the hood.

Convert a Single PDF:
```bash
# Download a sample PDF
curl -o olmocr-sample.pdf https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf

# Convert it to markdown
python -m olmocr.pipeline ./localworkspace --markdown --pdfs olmocr-sample.pdf
```

Convert an Image file:
```bash
python -m olmocr.pipeline ./localworkspace --markdown --pdfs random_page.png
```

Convert Multiple PDFs:
```bash
python -m olmocr.pipeline ./localworkspace --markdown --pdfs tests/gnarly_pdfs/*.pdf
```

With the addition of the `--markdown` flag, results will be stored as markdown files inside of `./localworkspace/markdown/`. 

### Using External vLLM Server

If you have a vLLM server already running elsewhere (or any inference platform implementing the relevant subset of the OpenAI API), you can point olmOCR to use it instead of spawning a local instance:

```bash
# Use external vLLM server instead of local one
python -m olmocr.pipeline ./localworkspace --server http://remote-server:8000 --markdown --pdfs tests/gnarly_pdfs/*.pdf
```

The served model name should be `olmocr`. An example vLLM launch command would be:
```bash
vllm serve allenai/olmOCR-7B-0825-FP8 --served-model-name olmocr --max-model-len 16384
```

#### Viewing Results

The `./localworkspace/` workspace folder will then have both [Dolma](https://github.com/allenai/dolma) and markdown files (if using `--markdown`).


```bash
cat localworkspace/markdown/olmocr-sample.md 
```

```
olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models
...
```

### Multi-node / Cluster Usage

If you want to convert millions of PDFs, using multiple nodes running in parallel, then olmOCR supports
reading your PDFs from AWS S3, and coordinating work using an AWS S3 output bucket.

For example, you can start this command on your first worker node, and it will set up
a simple work queue in your AWS bucket and start converting PDFs.

```bash
python -m olmocr.pipeline s3://my_s3_bucket/pdfworkspaces/exampleworkspace --pdfs s3://my_s3_bucket/jakep/gnarly_pdfs/*.pdf
```

Now on any subsequent nodes, just run this and they will start grabbing items from the same workspace queue.
```bash
python -m olmocr.pipeline s3://my_s3_bucket/pdfworkspaces/exampleworkspace
```

If you are at Ai2 and want to linearize millions of PDFs efficiently using [beaker](https://www.beaker.org), just add the `--beaker`
flag. This will prepare the workspace on your local machine, and then launch N GPU workers in the cluster to start
converting PDFs.

For example:
```bash
python -m olmocr.pipeline s3://my_s3_bucket/pdfworkspaces/exampleworkspace --pdfs s3://my_s3_bucket/jakep/gnarly_pdfs/*.pdf --beaker --beaker_gpus 4
```

### Using Docker

Pull the Docker image.
```bash
docker pull alleninstituteforai/olmocr:latest
```

To run the container interactively:
```bash
docker run -it --gpus all --name olmocr_container alleninstituteforai/olmocr:latest /bin/bash
```

If you want to access your local files inside the container, use volume mounting:
```bash
docker run -it --gpus all \
  -v /path/to/your/local/files:/local_files \
  --name olmocr_container \
  alleninstituteforai/olmocr:latest /bin/bash
```

All dependencies are already installed. Once you’re inside the container, you can run olmOCR commands. For example:

```bash
curl -o olmocr-sample.pdf https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf

python -m olmocr.pipeline ./localworkspace --markdown --pdfs olmocr-sample.pdf
```
> You can also visit our Docker repository on [Docker Hub](https://hub.docker.com/r/alleninstituteforai/olmocr).

### Full documentation for the pipeline

```bash
python -m olmocr.pipeline --help
usage: pipeline.py [-h] [--pdfs [PDFS ...]] [--model MODEL] [--workspace_profile WORKSPACE_PROFILE] [--pdf_profile PDF_PROFILE] [--pages_per_group PAGES_PER_GROUP] [--max_page_retries MAX_PAGE_RETRIES] [--max_page_error_rate MAX_PAGE_ERROR_RATE] [--workers WORKERS]
                   [--apply_filter] [--stats] [--markdown] [--target_longest_image_dim TARGET_LONGEST_IMAGE_DIM] [--target_anchor_text_len TARGET_ANCHOR_TEXT_LEN] [--guided_decoding] [--gpu-memory-utilization GPU_MEMORY_UTILIZATION] [--max_model_len MAX_MODEL_LEN]
                   [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--data-parallel-size DATA_PARALLEL_SIZE] [--port PORT] [--server SERVER] [--beaker] [--beaker_workspace BEAKER_WORKSPACE] [--beaker_cluster BEAKER_CLUSTER] [--beaker_gpus BEAKER_GPUS] [--beaker_priority BEAKER_PRIORITY]
                   workspace

Manager for running millions of PDFs through a batch inference pipeline

positional arguments:
  workspace             The filesystem path where work will be stored, can be a local folder, or an s3 path if coordinating work with many workers, s3://bucket/prefix/

options:
  -h, --help            show this help message and exit
  --pdfs [PDFS ...]     Path to add pdfs stored in s3 to the workspace, can be a glob path s3://bucket/prefix/*.pdf or path to file containing list of pdf paths
  --model MODEL         Path where the model is located, allenai/olmOCR-7B-0725-FP8 is the default, can be local, s3, or hugging face.
  --workspace_profile WORKSPACE_PROFILE
                        S3 configuration profile for accessing the workspace
  --pdf_profile PDF_PROFILE
                        S3 configuration profile for accessing the raw pdf documents
  --pages_per_group PAGES_PER_GROUP
                        Aiming for this many pdf pages per work item group
  --max_page_retries MAX_PAGE_RETRIES
                        Max number of times we will retry rendering a page
  --max_page_error_rate MAX_PAGE_ERROR_RATE
                        Rate of allowable failed pages in a document, 1/250 by default
  --workers WORKERS     Number of workers to run at a time
  --apply_filter        Apply basic filtering to English pdfs which are not forms, and not likely seo spam
  --stats               Instead of running any job, reports some statistics about the current workspace
  --markdown            Also write natural text to markdown files preserving the folder structure of the input pdfs
  --target_longest_image_dim TARGET_LONGEST_IMAGE_DIM
                        Dimension on longest side to use for rendering the pdf pages
  --target_anchor_text_len TARGET_ANCHOR_TEXT_LEN
                        Maximum amount of anchor text to use (characters), not used for new models
  --guided_decoding     Enable guided decoding for model YAML type outputs

VLLM arguments:
  --gpu-memory-utilization GPU_MEMORY_UTILIZATION
                        Fraction of VRAM vLLM may pre-allocate for KV-cache (passed through to vllm serve).
  --max_model_len MAX_MODEL_LEN
                        Upper bound (tokens) vLLM will allocate KV-cache for, lower if VLLM won't start
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
                        Tensor parallel size for vLLM
  --data-parallel-size DATA_PARALLEL_SIZE, -dp DATA_PARALLEL_SIZE
                        Data parallel size for vLLM
  --port PORT           Port to use for the VLLM server
  --server SERVER       URL of external vLLM (or other compatible provider)
                        server (e.g., http://hostname:port). If provided,
                        skips spawning local vLLM instance

beaker/cluster execution:
  --beaker              Submit this job to beaker instead of running locally
  --beaker_workspace BEAKER_WORKSPACE
                        Beaker workspace to submit to
  --beaker_cluster BEAKER_CLUSTER
                        Beaker clusters you want to run on
  --beaker_gpus BEAKER_GPUS
                        Number of gpu replicas to run
  --beaker_priority BEAKER_PRIORITY
                        Beaker priority level for the job
```

## Code overview

There are some nice reusable pieces of the code that may be useful for your own projects:
 - A prompting strategy to get really good natural text parsing using ChatGPT 4o - [buildsilver.py](https://github.com/allenai/olmocr/blob/main/olmocr/data/buildsilver.py)
 - An side-by-side eval toolkit for comparing different pipeline versions - [runeval.py](https://github.com/allenai/olmocr/blob/main/olmocr/eval/runeval.py)
 - Basic filtering by language and SEO spam removal - [filter.py](https://github.com/allenai/olmocr/blob/main/olmocr/filter/filter.py)
 - Finetuning code for Qwen2-VL and Molmo-O - [train.py](https://github.com/allenai/olmocr/blob/main/olmocr/train/train.py)
 - Processing millions of PDFs through a finetuned model using Sglang - [pipeline.py](https://github.com/allenai/olmocr/blob/main/olmocr/pipeline.py)
 - Viewing [Dolma docs](https://github.com/allenai/dolma) created from PDFs - [dolmaviewer.py](https://github.com/allenai/olmocr/blob/main/olmocr/viewer/dolmaviewer.py)



## Team

<!-- start team -->

**olmOCR** is developed and maintained by the AllenNLP team, backed by [the Allen Institute for Artificial Intelligence (AI2)](https://allenai.org/).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
To learn more about who specifically contributed to this codebase, see [our contributors](https://github.com/allenai/olmocr/graphs/contributors) page.

<!-- end team -->

## License

<!-- start license -->

**olmOCR** is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
A full copy of the license can be found [on GitHub](https://github.com/allenai/olmocr/blob/main/LICENSE).

<!-- end license -->

## Citing

```bibtex
@misc{olmocr,
      title={{olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models}},
      author={Jake Poznanski and Jon Borchardt and Jason Dunkelberger and Regan Huff and Daniel Lin and Aman Rangapur and Christopher Wilhelm and Kyle Lo and Luca Soldaini},
      year={2025},
      eprint={2502.18443},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.18443},
}
```
