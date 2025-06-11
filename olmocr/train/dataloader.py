from os import PathLike
from pathlib import Path
from typing import Dict, Any, Optional, Type
import base64
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from pypdf import PdfReader
from tqdm import tqdm
from dataclasses import dataclass, fields

from olmocr.data.renderpdf import render_pdf_to_base64png

@dataclass(frozen=True)
class StandardFrontMatter:
    primary_language: Optional[str]
    is_rotation_valid: bool
    rotation_correction: int
    is_table: bool
    is_diagram: bool

    def __post_init__(self):
        # Validate rotation_correction is one of the allowed values
        if self.rotation_correction not in {0, 90, 180, 270}:
            raise ValueError("rotation_correction must be one of [0, 90, 180, 270].")

        # Type checks
        if not isinstance(self.primary_language, (str, type(None))):
            raise TypeError("primary_language must be of type Optional[str].")
        if not isinstance(self.is_rotation_valid, bool):
            raise TypeError("is_rotation_valid must be of type bool.")
        if not isinstance(self.rotation_correction, int):
            raise TypeError("rotation_correction must be of type int.")
        if not isinstance(self.is_table, bool):
            raise TypeError("is_table must be of type bool.")
        if not isinstance(self.is_diagram, bool):
            raise TypeError("is_diagram must be of type bool.")
        

class MarkdownPDFDocumentDataset(Dataset):
    def __init__(self, root_dir: str | PathLike, target_longest_image_dim: int, image_transform=None, front_matter_class=None):
        """
        Initialize the dataset by finding all markdown files with corresponding PDFs.
        
        Args:
            root_dir: Path to the root folder containing processed markdown and PDF files
            target_longest_image_dim: Target dimension for the longest side of the image
            image_transform: Optional transform to apply to the PDF images
            front_matter_class: Optional dataclass type to validate front matter against
        """
        self.root_dir = Path(root_dir)
        self.image_transform = image_transform
        self.target_longest_image_dim = target_longest_image_dim
        self.front_matter_class = front_matter_class
        self.samples = []
        
        # Find all markdown files recursively
        print(f"Scanning for markdown files in {self.root_dir}...")
        md_files = list(self.root_dir.rglob("*.md"))
        
        # Verify each markdown file has a corresponding PDF
        valid_count = 0
        invalid_pdfs = []
        
        print(f"Validating {len(md_files)} markdown-PDF pairs...")
        for md_path in tqdm(md_files, desc="Validating PDFs"):
            # Look for PDF with same stem (filename without extension)
            pdf_path = md_path.with_suffix('.pdf')
            
            if pdf_path.exists() or pdf_path.is_symlink():
                # Resolve symlink if it is one
                if pdf_path.is_symlink():
                    pdf_path = pdf_path.resolve()
                    
                # Verify the resolved path exists
                if pdf_path.exists():
                    # Validate PDF - check it loads and has exactly one page
                    try:
                        reader = PdfReader(str(pdf_path))
                        num_pages = len(reader.pages)
                        
                        if num_pages != 1:
                            invalid_pdfs.append((pdf_path, f"Expected 1 page, found {num_pages}"))
                            continue
                            
                        self.samples.append({
                            'markdown_path': md_path,
                            'pdf_path': pdf_path
                        })
                        valid_count += 1
                        
                    except Exception as e:
                        invalid_pdfs.append((pdf_path, f"Failed to load: {str(e)}"))
        
        print(f"Found {valid_count} valid markdown-PDF pairs")
        
        if invalid_pdfs:
            print(f"\nWarning: {len(invalid_pdfs)} invalid PDFs found:")
            for pdf_path, reason in invalid_pdfs[:5]:  # Show first 5
                print(f"  - {pdf_path.name}: {reason}")
            if len(invalid_pdfs) > 5:
                print(f"  ... and {len(invalid_pdfs) - 5} more")
    
    def _extract_front_matter_and_text(self, markdown_content: str) -> tuple[str, str]:
        """Extract raw front matter string and text from markdown content."""
        if markdown_content.startswith('---\n'):
            parts = markdown_content.split('---\n', 2)
            if len(parts) >= 3:
                return parts[1].strip(), parts[2].strip()
        
        return '', markdown_content
    
    def _parse_front_matter_string(self, front_matter_str: str) -> Dict[str, Any]:
        """Parse front matter string into a dictionary."""
        front_matter = {}
        
        if not front_matter_str:
            return front_matter
            
        for line in front_matter_str.split('\n'):
            if ': ' in line:
                key, value = line.split(': ', 1)
                # Simple type inference
                if value.lower() == 'true':
                    front_matter[key] = True
                elif value.lower() == 'false':
                    front_matter[key] = False
                elif value.isdigit():
                    front_matter[key] = int(value)
                else:
                    front_matter[key] = value
        
        return front_matter
    
    def _parse_front_matter(self, front_matter_dict: Dict[str, Any]) -> Any:
        """Parse front matter dictionary into dataclass instance if front_matter_class is specified."""
        if not self.front_matter_class:
            return front_matter_dict
            
        # Get field names and types from the dataclass
        field_info = {f.name: f.type for f in fields(self.front_matter_class)}
        
        # Validate and convert values
        kwargs = {}
        for field_name, field_type in field_info.items():
            if field_name not in front_matter_dict:
                raise ValueError(f"Missing required field '{field_name}' in front matter")
                
            value = front_matter_dict[field_name]
            
            # Handle type conversions
            if field_type == int and isinstance(value, str):
                kwargs[field_name] = int(value)
            elif field_type == bool and isinstance(value, str):
                kwargs[field_name] = value.lower() == 'true'
            elif field_type == Optional[str]:
                kwargs[field_name] = value if value else None
            else:
                kwargs[field_name] = value
                
        # Check for extra fields
        extra_fields = set(front_matter_dict.keys()) - set(field_info.keys())
        if extra_fields:
            raise ValueError(f"Unexpected fields in front matter: {extra_fields}")
            
        return self.front_matter_class(**kwargs)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Returns:
            dict containing:
                - 'image': PIL Image of the rendered PDF page
                - 'pdf_path': Path to the PDF file
                - 'text': Text content without front matter
                - 'front_matter': Dict with parsed front matter
        """
        sample = self.samples[idx]
        
        # Read and parse markdown file
        markdown_content = sample['markdown_path'].read_text(encoding='utf-8')
        front_matter_str, text = self._extract_front_matter_and_text(markdown_content)
        front_matter = self._parse_front_matter_string(front_matter_str)
        
        # Render PDF to image
        base64_png = render_pdf_to_base64png(str(sample['pdf_path']), page_num=1, target_longest_image_dim=self.target_longest_image_dim)
        png_bytes = base64.b64decode(base64_png)
        image = Image.open(BytesIO(png_bytes))
        
        # Apply transform if provided
        if self.image_transform:
            image = self.image_transform(image)
        
        # Parse front matter to dataclass if specified
        try:
            parsed_front_matter = self._parse_front_matter(front_matter)
        except Exception as e:
            raise ValueError(f"Error parsing front matter for {sample['markdown_path']}: {e}")
        
        return {
            'image': image,
            'pdf_path': str(sample['pdf_path']),
            'text': text,
            'front_matter': parsed_front_matter
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MarkdownPDFDocumentDataset")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/home/ubuntu/olmOCR-mix-0225/processed_00_documents_eval_s2pdf/",
        help="Root directory containing processed markdown and PDF files"
    )
    
    args = parser.parse_args()
    
    # Test dataset initialization
    print(f"\nTesting dataset with root directory: {args.root_dir}")
    dataset = MarkdownPDFDocumentDataset(args.root_dir, target_longest_image_dim=1024, front_matter_class=StandardFrontMatter, image_transform=None)
    
    print(f"\nDataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # Show first few samples
        print("\nFirst 5 samples:")
        for i in range(min(5, len(dataset))):
            sample = dataset.samples[i]
            print(f"  {i}: MD: {sample['markdown_path'].name}, PDF: {sample['pdf_path'].name}")
            
        # Test __getitem__
        print("\nTesting __getitem__ on first sample:")
        first_sample = dataset[0]
        print(f"Image type: {type(first_sample['image'])}")
        print(f"Image size: {first_sample['image'].size}")
        print(f"PDF Path: {first_sample['pdf_path']}")
        print(f"Front Matter: {first_sample['front_matter']}")
        print(f"Text: {first_sample['text']}...")
