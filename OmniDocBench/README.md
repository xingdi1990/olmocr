### Get the OmniDocBench dataset (start here)

```shell
pip install huggingface_hub

# you may need to login with your HF_TOKENS by huggingface-cli login
# huggingface-cli download --repo-type dataset opendatalab/OmniDocBench --revision v1_0 --local-dir ./bench_data_v1
huggingface-cli download --repo-type dataset opendatalab/OmniDocBench --local-dir ./bench_data
```

### Get the OmniDOcBench code


```shell
git clone https://github.com/opendatalab/OmniDocBench.git
cd OmniDocBench
conda create -n omnidocbench python=3.10
conda activate omnidocbench
pip install -r requirements.txt
```

### Evaluate the model

```shell
python pdf_validation.py --config <config_path>
```
