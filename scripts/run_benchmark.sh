#!/bin/bash

set -e

# Get version from version.py
VERSION=$(python -c 'import olmocr.version; print(olmocr.version.VERSION)')
echo "OlmOCR version: $VERSION"

# Get first 10 characters of git hash
GIT_HASH=$(git rev-parse HEAD | cut -c1-10)
echo "Git hash: $GIT_HASH"

# Create full image tag
IMAGE_TAG="olmocr-benchmark-${VERSION}-${GIT_HASH}"
echo "Building Docker image with tag: $IMAGE_TAG"

# Build the Docker image
echo "Building Docker image..."
docker build --platform linux/amd64 -f ./Dockerfile -t $IMAGE_TAG .

# Push image to beaker
echo "Pushing image to Beaker..."
beaker image create --workspace ai2/oe-data-pdf --name $IMAGE_TAG $IMAGE_TAG

# Create Python script to run beaker experiment
cat << 'EOF' > /tmp/run_benchmark_experiment.py
import sys
from beaker import Beaker, ExperimentSpec, TaskSpec, TaskResources, ImageSource, Priority, Constraints

# Get image tag from command line
image_tag = sys.argv[1]

# Initialize Beaker client
b = Beaker.from_env(default_workspace="ai2/oe-data-pdf")

# Create experiment spec
experiment_spec = ExperimentSpec(
    description="OlmOCR Benchmark Run",
    budget="ai2/oe-data",
    tasks=[
        TaskSpec(
            name="olmocr-benchmark",
            image=ImageSource(beaker=f"ai2/oe-data-pdf/{image_tag}"),
            command=[
                "bash", "-c",
                " && ".join([
                    "huggingface-cli download --repo-type dataset --resume-download allenai/olmOCR-bench --local-dir ./olmOCR-bench",
                    "python -m olmocr.pipeline ./localworkspace --markdown --pdfs './olmOCR-bench/bench_data/pdfs/**/*.pdf'",
                    "python olmocr/bench/scripts/workspace_to_bench.py localworkspace/ olmOCR-bench/bench_data/markdown_output --bench-path ./olmOCR-bench/",
                    "python -m olmocr.bench.benchmark --dir ./olmOCR-bench/bench_data"
                ])
            ],
            resources=TaskResources(gpu_count=1),
            #constraints=Constraint(cluster=["ai2/pluto-cirrascale", "ai2/jupiter-cirrascale"]),
            priority=Priority.normal,
        )
    ],
)

# Create the experiment
experiment = b.experiment.create(spec=experiment_spec, workspace="ai2/oe-data-pdf")
print(f"Created experiment: {experiment.id}")
print(f"View at: https://beaker.org/ex/{experiment.id}")
EOF

# Run the Python script to create the experiment
echo "Creating Beaker experiment..."
python /tmp/run_benchmark_experiment.py $IMAGE_TAG

# Clean up temporary file
rm /tmp/run_benchmark_experiment.py

echo "Benchmark experiment submitted successfully!"