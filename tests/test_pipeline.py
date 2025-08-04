import asyncio
import base64
import json
import tempfile
from io import BytesIO
from unittest import mock

import pytest
from PIL import Image

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.pipeline import build_page_query, process_page
from olmocr.prompts import PageResponse


@pytest.mark.asyncio
async def test_build_page_query_rotation():
    """Test that build_page_query correctly rotates images when requested."""

    # Create a simple test image with an asymmetric pattern to verify rotation
    test_image = Image.new("RGB", (100, 200), color="white")
    # Add a red square in top-left to track rotation
    for x in range(20):
        for y in range(20):
            test_image.putpixel((x, y), (255, 0, 0))

    # Save as PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        test_image.save(tmp_pdf.name, format="PDF")
        pdf_path = tmp_pdf.name

    # Don't mock render_pdf_to_base64png - use the real function
    # Test no rotation (0 degrees)
    result = await build_page_query(pdf_path, 1, 1288, image_rotation=0)

    # Extract and decode the image from the result
    image_data = result["messages"][0]["content"][0]["image_url"]["url"]
    base64_data = image_data.split(",")[1]
    decoded_image = Image.open(BytesIO(base64.b64decode(base64_data)))

    # The image is scaled to fit within 1288 pixels on the longest side
    # Original was 100x200, so it gets scaled up
    width, height = decoded_image.size
    assert height == 1288  # Height is the longest dimension
    assert width == 644  # Width scales proportionally (100/200 * 1288)

    # Check that red pixels are in the top-left region (accounting for scaling)
    # The red square should be roughly in the top-left 20% of the image
    red_found = False
    for x in range(width // 5):
        for y in range(height // 10):
            if decoded_image.getpixel((x, y))[0] > 200:  # Red channel high
                red_found = True
                break
        if red_found:
            break
    assert red_found

    # Test 90 degree rotation
    result = await build_page_query(pdf_path, 1, 1288, image_rotation=90)
    image_data = result["messages"][0]["content"][0]["image_url"]["url"]
    base64_data = image_data.split(",")[1]
    decoded_image = Image.open(BytesIO(base64.b64decode(base64_data)))

    # After 90 degree rotation, dimensions should be swapped
    width, height = decoded_image.size
    assert width == 1288  # Width is now the longest dimension
    assert height == 644  # Height scales proportionally

    # Red square should now be in top-right region
    red_found = False
    for x in range(width - width // 10, width):
        for y in range(height // 5):
            if decoded_image.getpixel((x, y))[0] > 200:
                red_found = True
                break
        if red_found:
            break
    assert red_found

    # Test 180 degree rotation
    result = await build_page_query(pdf_path, 1, 1288, image_rotation=180)
    image_data = result["messages"][0]["content"][0]["image_url"]["url"]
    base64_data = image_data.split(",")[1]
    decoded_image = Image.open(BytesIO(base64.b64decode(base64_data)))

    # Same dimensions as 0 rotation
    width, height = decoded_image.size
    assert height == 1288
    assert width == 644

    # Red square should be in bottom-right region
    red_found = False
    for x in range(width - width // 5, width):
        for y in range(height - height // 10, height):
            if decoded_image.getpixel((x, y))[0] > 200:
                red_found = True
                break
        if red_found:
            break
    assert red_found

    # Test 270 degree rotation
    result = await build_page_query(pdf_path, 1, 1288, image_rotation=270)
    image_data = result["messages"][0]["content"][0]["image_url"]["url"]
    base64_data = image_data.split(",")[1]
    decoded_image = Image.open(BytesIO(base64.b64decode(base64_data)))

    # Dimensions should be swapped like 90 degree
    width, height = decoded_image.size
    assert width == 1288
    assert height == 644

    # Red square should be in bottom-left region
    red_found = False
    for x in range(width // 10):
        for y in range(height - height // 5, height):
            if decoded_image.getpixel((x, y))[0] > 200:
                red_found = True
                break
        if red_found:
            break
    assert red_found


@pytest.mark.asyncio
async def test_process_page_rotation_correction():
    """Test that process_page correctly handles rotation correction from server response."""

    # Create mock args
    mock_args = mock.Mock()
    mock_args.max_page_retries = 3
    mock_args.target_longest_image_dim = 1288
    mock_args.target_anchor_text_len = 1000
    mock_args.guided_decoding = False

    # Create a test PDF
    test_image = Image.new("RGB", (100, 200), color="white")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        test_image.save(tmp_pdf.name, format="PDF")
        pdf_path = tmp_pdf.name

    # Mock responses from the server
    # First response: indicates rotation is invalid, needs 90 degree correction
    first_response = {
        "choices": [
            {
                "message": {
                    "content": """---
primary_language: en
is_rotation_valid: false
rotation_correction: 90
is_table: false
is_diagram: false
---
This text appears to be rotated"""
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050},
    }

    # Second response: after rotation, indicates valid
    second_response = {
        "choices": [
            {
                "message": {
                    "content": """---
primary_language: en
is_rotation_valid: true
rotation_correction: 0
is_table: false
is_diagram: false
---
This is the correctly oriented text content of the page."""
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 60, "total_tokens": 1060},
    }

    # Track the rotation values passed to build_page_query
    rotation_values = []

    async def mock_build_page_query(local_pdf_path, page, target_dim, image_rotation=0):
        rotation_values.append(image_rotation)
        return {"model": "olmocr", "messages": [{"role": "user", "content": []}], "max_tokens": 4500, "temperature": 0.1}

    # Mock the server response
    response_iter = iter([(200, json.dumps(first_response).encode()), (200, json.dumps(second_response).encode())])

    with (
        mock.patch("olmocr.pipeline.build_page_query", side_effect=mock_build_page_query),
        mock.patch("olmocr.pipeline.apost", side_effect=lambda url, json_data: next(response_iter)),
        mock.patch("olmocr.metrics.MetricsKeeper.add_metrics"),
        mock.patch("olmocr.metrics.WorkerTracker.track_work"),
    ):

        result = await process_page(mock_args, 0, "s3://bucket/test.pdf", pdf_path, 1)

        # Verify rotation values
        assert len(rotation_values) == 2
        assert rotation_values[0] == 0  # First attempt with no rotation
        assert rotation_values[1] == 90  # Second attempt with 90 degree rotation

        # Verify final result
        assert result.page_num == 1
        assert result.response.is_rotation_valid == True
        assert result.response.rotation_correction == 0
        assert result.response.natural_text == "This is the correctly oriented text content of the page."
        assert result.is_fallback == False


@pytest.mark.asyncio
async def test_process_page_rotation_multiple_corrections():
    """Test handling of multiple rotation corrections before finding the right orientation."""

    mock_args = mock.Mock()
    mock_args.max_page_retries = 4
    mock_args.target_longest_image_dim = 1288
    mock_args.target_anchor_text_len = 1000
    mock_args.guided_decoding = False

    # Create a test PDF
    test_image = Image.new("RGB", (100, 200), color="white")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        test_image.save(tmp_pdf.name, format="PDF")
        pdf_path = tmp_pdf.name

    # Mock responses - try 0, then 90, then 180 before success
    responses = [
        {  # First try - needs 90 degree rotation
            "choices": [
                {
                    "message": {
                        "content": """---
primary_language: en
is_rotation_valid: false
rotation_correction: 90
is_table: false
is_diagram: false
---
Text appears rotated"""
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050},
        },
        {  # Second try - still wrong, needs 180 degree total
            "choices": [
                {
                    "message": {
                        "content": """---
primary_language: en
is_rotation_valid: false
rotation_correction: 180
is_table: false
is_diagram: false
---
Still not right orientation"""
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050},
        },
        {  # Third try - finally correct
            "choices": [
                {
                    "message": {
                        "content": """---
primary_language: en
is_rotation_valid: true
rotation_correction: 0
is_table: false
is_diagram: false
---
Perfect! This is the correct orientation."""
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 60, "total_tokens": 1060},
        },
    ]

    rotation_values = []

    async def mock_build_page_query(local_pdf_path, page, target_dim, image_rotation=0):
        rotation_values.append(image_rotation)
        return {"model": "olmocr", "messages": [{"role": "user", "content": []}], "max_tokens": 4500, "temperature": 0.1}

    response_iter = iter((200, json.dumps(resp).encode()) for resp in responses)

    with (
        mock.patch("olmocr.pipeline.build_page_query", side_effect=mock_build_page_query),
        mock.patch("olmocr.pipeline.apost", side_effect=lambda url, json_data: next(response_iter)),
        mock.patch("olmocr.metrics.MetricsKeeper.add_metrics"),
        mock.patch("olmocr.metrics.WorkerTracker.track_work"),
    ):

        result = await process_page(mock_args, 0, "s3://bucket/test.pdf", pdf_path, 1)

        # Verify rotation sequence
        assert rotation_values == [0, 90, 180]

        # Verify successful result
        assert result.response.is_rotation_valid == True
        assert result.response.natural_text == "Perfect! This is the correct orientation."
        assert not result.is_fallback


@pytest.mark.asyncio
async def test_rotated_pdf_correction():
    """Test rotation correction on edgar-rotated90.pdf - a PDF that's already rotated 90 degrees."""

    import os
    from pathlib import Path

    # Path to the pre-rotated PDF
    pdf_path = "tests/gnarly_pdfs/edgar-rotated90.pdf"
    output_dir = "tests/rotation_output"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nTesting rotation correction on pre-rotated PDF: {pdf_path}")

    mock_args = mock.Mock()
    mock_args.max_page_retries = 4
    mock_args.target_longest_image_dim = 1288
    mock_args.target_anchor_text_len = 1000
    mock_args.guided_decoding = False

    # Track images at each rotation attempt
    captured_images = []
    rotation_values = []

    async def mock_build_page_query_capture(local_pdf_path, page, target_dim, image_rotation=0):
        rotation_values.append(image_rotation)
        # Actually build the query to get the real image
        real_result = await build_page_query(local_pdf_path, page, target_dim, image_rotation)

        # Capture the image
        image_data = real_result["messages"][0]["content"][0]["image_url"]["url"]
        base64_data = image_data.split(",")[1]
        decoded_image = Image.open(BytesIO(base64.b64decode(base64_data)))
        captured_images.append((image_rotation, decoded_image))

        return real_result

    # Mock responses that simulate the server detecting the rotation
    responses = [
        {  # First try - detects 90 degree rotation already in the PDF
            "choices": [
                {
                    "message": {
                        "content": """---
primary_language: en
is_rotation_valid: false
rotation_correction: 270
is_table: false
is_diagram: false
---
90 degree rotation detected in the document"""
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050},
        },
        {  # Second try - after applying 270 degree correction (to undo the 90)
            "choices": [
                {
                    "message": {
                        "content": """---
primary_language: en
is_rotation_valid: true
rotation_correction: 0
is_table: false
is_diagram: false
---
Document is now correctly oriented after rotation correction."""
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 60, "total_tokens": 1060},
        },
    ]

    response_iter = iter((200, json.dumps(resp).encode()) for resp in responses)

    with (
        mock.patch("olmocr.pipeline.build_page_query", side_effect=mock_build_page_query_capture),
        mock.patch("olmocr.pipeline.apost", side_effect=lambda url, json_data: next(response_iter)),
        mock.patch("olmocr.metrics.MetricsKeeper.add_metrics"),
        mock.patch("olmocr.metrics.WorkerTracker.track_work"),
    ):
        result = await process_page(mock_args, 0, pdf_path, pdf_path, 1)

        # Save the captured images from the rotation correction process
        for i, (rotation, image) in enumerate(captured_images):
            output_path = os.path.join(output_dir, f"edgar_rotated90_attempt{i}_rotation{rotation}.png")
            image.save(output_path)
            print(f"\nRotation correction attempt {i}: {rotation} degrees")
            print(f"Saved to: {output_path}")
            print(f"Dimensions: {image.size}")

    print(f"\nImages saved to: {output_dir}")
    print("The first image should show the PDF rotated 90 degrees (sideways)")
    print("The second image should show it correctly oriented after 270 degree correction")

    # Verify the test results
    assert len(rotation_values) == 2
    assert rotation_values[0] == 0  # First attempt with no additional rotation
    assert rotation_values[1] == 270  # Second attempt with 270 degree rotation to correct
    assert result.response.is_rotation_valid == True


if __name__ == "__main__":
    asyncio.run(test_build_page_query_rotation())
    asyncio.run(test_process_page_rotation_correction())
    asyncio.run(test_process_page_rotation_multiple_corrections())
    asyncio.run(test_rotated_pdf_correction())
