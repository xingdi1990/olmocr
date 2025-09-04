#!/usr/bin/env python3
"""
DotsOCR to Benchmark Conversion Script

This script converts DotsOCR model output to ideal markdown content.
Based on the reference implementation in dots_ocr/parser.py lines 175-234.
"""

import json
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import base64
from io import BytesIO

try:
    from PIL import Image
except ImportError:
    print("Warning: PIL not available. Image processing features will be limited.")
    Image = None


class OutputCleaner:
    """Cleans model output when JSON parsing fails."""
    
    def clean_model_output(self, response: str) -> str:
        """
        Clean raw model output and convert to readable format.
        This is a fallback when JSON parsing fails.
        """
        # Basic cleaning - remove common artifacts
        cleaned = response.strip()
        
        # Remove any JSON-like artifacts that might be malformed
        if cleaned.startswith('{') or cleaned.startswith('['):
            # Try to extract text content from malformed JSON
            lines = cleaned.split('\n')
            text_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('{') and not line.startswith('[') and not line.startswith('}') and not line.startswith(']'):
                    # Clean up quotes and common artifacts
                    line = line.strip('"\'')
                    if line:
                        text_lines.append(line)
            
            if text_lines:
                return '\n\n'.join(text_lines)
        
        return cleaned


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Basic text cleaning
    text = text.strip()
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text


def get_formula_in_markdown(text: str) -> str:
    """
    Convert LaTeX formula to proper markdown format.
    """
    if not text:
        return ""
    
    # If it's already wrapped in $$ for display math, keep it
    if text.startswith('$$') and text.endswith('$$'):
        return text
    
    # If it's wrapped in $ for inline math, keep it
    if text.startswith('$') and text.endswith('$'):
        return text
    
    # Otherwise, wrap in $$ for display math
    return f"$${text}$$"


def PIL_image_to_base64(image) -> str:
    """Convert PIL Image to base64 string."""
    if Image is None:
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def layout_json_to_markdown(image_path: Optional[str], cells: List[Dict], text_key: str = 'text', no_page_hf: bool = False) -> str:
    """
    Convert layout JSON format to Markdown.
    
    Args:
        image_path: Path to the source image (optional)
        cells: List of layout cell dictionaries
        text_key: The key for the text field in the cell dictionary
        no_page_hf: If True, skips page headers and footers
        
    Returns:
        str: The text in Markdown format
    """
    text_items = []
    
    # Load image if available
    image = None
    if image_path and Image and os.path.exists(image_path):
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
    
    for i, cell in enumerate(cells):
        # Handle different cell formats
        if 'bbox' in cell:
            x1, y1, x2, y2 = [int(coord) for coord in cell['bbox']]
        else:
            x1 = y1 = x2 = y2 = 0
            
        text = cell.get(text_key, "")
        category = cell.get('category', 'Text')
        
        # Skip page headers and footers if requested
        if no_page_hf and category in ['Page-header', 'Page-footer']:
            continue
        
        # Handle different content types
        if category == 'Picture':
            if image is not None:
                try:
                    image_crop = image.crop((x1, y1, x2, y2))
                    image_base64 = PIL_image_to_base64(image_crop)
                    text_items.append(f"![]({image_base64})")
                except Exception as e:
                    print(f"Warning: Could not process image region: {e}")
                    text_items.append("![Image]")
            else:
                text_items.append("![Image]")
                
        elif category == 'Formula':
            formula_md = get_formula_in_markdown(text)
            text_items.append(formula_md)
            
        elif category == 'Table':
            # Tables should already be in HTML or markdown format
            cleaned_text = clean_text(text)
            if cleaned_text:
                text_items.append(cleaned_text)
                
        else:
            # Regular text content
            cleaned_text = clean_text(text)
            if cleaned_text:
                text_items.append(cleaned_text)
    
    markdown_text = '\n\n'.join(text_items)
    return markdown_text


def post_process_output(response: str, image_path: Optional[str] = None) -> Tuple[Any, bool]:
    """
    Post-process model output, attempting JSON parsing first.
    
    Args:
        response: Raw model output
        image_path: Optional path to source image
        
    Returns:
        Tuple of (processed_data, is_filtered)
        - processed_data: Either parsed JSON cells or cleaned text
        - is_filtered: True if JSON parsing failed and fallback was used
    """
    json_load_failed = False
    
    try:
        # Try to parse as JSON
        cells = json.loads(response)
        
        # Validate that it's a list of dictionaries (layout format)
        if isinstance(cells, list) and all(isinstance(cell, dict) for cell in cells):
            return cells, False
        else:
            # If it's not the expected format, treat as failed
            json_load_failed = True
            
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"JSON parsing failed: {e}")
        json_load_failed = True
    
    if json_load_failed:
        # Use fallback cleaning
        cleaner = OutputCleaner()
        cleaned_response = cleaner.clean_model_output(response)
        return cleaned_response, True
    
    return response, False


def convert_output_to_markdown(model_output: str, 
                              image_path: Optional[str] = None,
                              text_key: str = 'text',
                              no_page_hf: bool = False) -> str:
    """
    Main conversion function to convert model output to markdown.
    
    Args:
        model_output: Raw output from the model
        image_path: Optional path to the source image
        text_key: Key for text field in JSON cells
        no_page_hf: Whether to exclude page headers/footers
        
    Returns:
        str: Converted markdown content
    """
    # Post-process the output
    processed_data, is_filtered = post_process_output(model_output, image_path)
    
    if is_filtered:
        # If JSON parsing failed, return cleaned text directly
        return processed_data
    else:
        # Convert layout JSON to markdown
        return layout_json_to_markdown(image_path, processed_data, text_key, no_page_hf)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Convert DotsOCR model output to markdown')
    parser.add_argument('input_file', help='Input file containing model output')
    parser.add_argument('-o', '--output', help='Output markdown file (default: stdout)')
    parser.add_argument('-i', '--image', help='Path to source image file')
    parser.add_argument('--text-key', default='text', help='Key for text field in JSON (default: text)')
    parser.add_argument('--no-page-hf', action='store_true', help='Exclude page headers and footers')
    parser.add_argument('--both-versions', action='store_true', help='Generate both regular and no-header-footer versions')
    
    args = parser.parse_args()
    
    # Read input file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            model_output = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    # Convert to markdown
    try:
        if args.both_versions:
            # Generate both versions
            md_content = convert_output_to_markdown(
                model_output, args.image, args.text_key, no_page_hf=False
            )
            md_content_no_hf = convert_output_to_markdown(
                model_output, args.image, args.text_key, no_page_hf=True
            )
            
            if args.output:
                # Save regular version
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                
                # Save no-header-footer version
                base_name = os.path.splitext(args.output)[0]
                ext = os.path.splitext(args.output)[1] or '.md'
                nohf_output = f"{base_name}_nohf{ext}"
                with open(nohf_output, 'w', encoding='utf-8') as f:
                    f.write(md_content_no_hf)
                
                print(f"Markdown content saved to: {args.output}")
                print(f"No header/footer version saved to: {nohf_output}")
            else:
                print("=== Regular Version ===")
                print(md_content)
                print("\n=== No Header/Footer Version ===")
                print(md_content_no_hf)
        else:
            # Generate single version
            md_content = convert_output_to_markdown(
                model_output, args.image, args.text_key, args.no_page_hf
            )
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                print(f"Markdown content saved to: {args.output}")
            else:
                print(md_content)
                
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
