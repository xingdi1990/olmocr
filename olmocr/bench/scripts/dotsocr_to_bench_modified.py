"""
Convert OCR MD files (containing JSON data) to clean Markdown files
Usage:
    python dotsocr_to_bench_modified.py input_dir ./markdown_output --bench-path ../olmOCR-bench/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_md_files(input_dir):
    """Load all MD files from the input directory that contain JSON data."""
    md_files = list(Path(input_dir).glob("*.md"))
    if not md_files:
        print(f"No MD files found in {input_dir}")
        return []

    print(f"Found {len(md_files)} MD files: {[f.name for f in md_files]}")
    return md_files


def parse_ocr_md_entries(md_files):
    """Parse OCR MD files containing JSON data and extract text content."""
    all_entries = []
    
    for md_file in md_files:
        print(f"Processing {md_file.name}...")
        
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                continue
            
            # Parse the JSON array
            try:
                ocr_data = json.loads(content)
                if isinstance(ocr_data, list):
                    # Extract text from OCR results, filtering out empty text and non-text categories
                    texts = []
                    for item in ocr_data:
                        if isinstance(item, dict) and "text" in item and item["text"].strip():
                            # Skip categories that typically don't contain meaningful text
                            category = item.get("category", "")
                            if category not in ["Picture"]:
                                texts.append(item["text"])
                    
                    # Combine all text pieces with proper spacing
                    full_text = "\n\n".join(texts)
                    
                    # Extract filename info for output structure
                    filename_stem = md_file.stem  # e.g., "2502.15977_pg21_pg1_repeat1"
                    
                    # Parse filename to extract subdir and pdf info
                    # Assuming format: {pdf_name}_{page_info}
                    if "_pg" in filename_stem:
                        pdf_name = filename_stem.split("_pg")[0]
                    else:
                        pdf_name = filename_stem
                    
                    # For the subdir, we'll use the parent directory name
                    subdir = md_file.parent.name
                    
                    all_entries.append({
                        "text": full_text,
                        "pdf_name": pdf_name,
                        "subdir": subdir,
                        "original_filename": filename_stem
                    })
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing {md_file.name}: {e}")
                continue

    print(f"Loaded {len(all_entries)} entries from MD files")
    return all_entries


def create_markdown_files(entries, output_dir):
    """Create markdown files from OCR entries."""
    output_path = Path(output_dir)
    created_files = 0
    blank_files = 0
    
    for entry in entries:
        subdir = entry["subdir"]
        pdf_name = entry["pdf_name"]
        text = entry["text"]
        
        # Create subdirectory
        subdir_path = output_path / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        
        # Create markdown filename
        md_filename = f"{pdf_name}_pg1_repeat1.md"
        md_filepath = subdir_path / md_filename
        
        # Write content to file
        with open(md_filepath, "w", encoding="utf-8") as f:
            f.write(text)
        
        if not text.strip():
            blank_files += 1
        
        created_files += 1
        print(f"Created: {subdir}/{md_filename}")
    
    print(f"Created {created_files} markdown files from OCR data")
    print(f"{blank_files} of those had empty content")
    return created_files


def main():
    parser = argparse.ArgumentParser(description="Convert OCR MD files to Markdown")
    parser.add_argument("input_dir", help="Input directory containing MD files with JSON data")
    parser.add_argument("output_dir", nargs="?", default="./markdown_output", 
                       help="Output directory for markdown files (default: ./markdown_output)")
    parser.add_argument("--bench-path", default="../olmOCR-bench", 
                       help="Path to olmOCR-bench directory (default: ../olmOCR-bench)")

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)

    md_files = load_md_files(input_dir)
    if not md_files:
        sys.exit(1)

    entries = parse_ocr_md_entries(md_files)
    if not entries:
        print("No entries found in MD files")
        sys.exit(1)

    created_files = create_markdown_files(entries, output_dir)

    print("\nSummary:")
    print(f"Created {created_files} markdown files from OCR data")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
