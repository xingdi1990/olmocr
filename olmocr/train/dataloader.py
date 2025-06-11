from os import PathLike
from pathlib import Path
from typing import Dict, Any, Optional, Type, List, Callable, TypeAlias
import base64
from io import BytesIO
from functools import reduce
import logging
import yaml
from PIL import Image
from torch.utils.data import Dataset
from pypdf import PdfReader
from tqdm import tqdm
from dataclasses import dataclass, fields
from abc import ABC, abstractmethod

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts.prompts import PageResponse, build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

# Type alias for samples
Sample: TypeAlias = Dict[str, Any]

# Configure logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""
    
    @abstractmethod
    def __call__(self, sample: Sample) -> Sample:
        """Process a sample and return the modified sample."""
        ...


class BaseMarkdownPDFDataset(Dataset):
    """Base dataset class that loads and verifies markdown-PDF pairs."""
    
    def __init__(self, root_dir: str | PathLike, pipeline_steps: Optional[List[PipelineStep]] = None):
        """
        Initialize the dataset by finding all markdown files with corresponding PDFs.
        
        Args:
            root_dir: Path to the root folder containing processed markdown and PDF files
            pipeline_steps: Optional list of pipeline steps to apply to each sample
        """
        self.root_dir = Path(root_dir)
        self.pipeline_steps = pipeline_steps or []
        self.samples = []
        
        # Find all markdown files recursively
        logger.info(f"Scanning for markdown files in {self.root_dir}...")
        md_files = list(self.root_dir.rglob("*.md"))
        
        # Verify each markdown file has a corresponding PDF
        valid_count = 0
        invalid_pdfs = []
        
        logger.info(f"Validating {len(md_files)} markdown-PDF pairs...")
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
        
        logger.info(f"Found {valid_count} valid markdown-PDF pairs")
        
        if invalid_pdfs:
            logger.warning(f"{len(invalid_pdfs)} invalid PDFs found:")
            for pdf_path, reason in invalid_pdfs[:5]:  # Show first 5
                logger.warning(f"  - {pdf_path.name}: {reason}")
            if len(invalid_pdfs) > 5:
                logger.warning(f"  ... and {len(invalid_pdfs) - 5} more")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Returns:
            dict containing at minimum:
                - 'markdown_path': Path to the markdown file
                - 'pdf_path': Path to the PDF file
                
            Additional fields will be added by pipeline steps.
        """
        # Start with basic sample info
        sample = self.samples[idx].copy()
        
        # Apply pipeline steps using reduce
        return reduce(lambda s, f: f(s), self.pipeline_steps, sample)
    

@dataclass(frozen=True, slots=True)
class FrontMatterParser(PipelineStep):
    """Pipeline step that parses YAML front matter from markdown content."""
    front_matter_class: Optional[Type] = None
    
    def _extract_front_matter_and_text(self, markdown_content: str) -> tuple[Dict[str, Any], str]:
        """Extract YAML front matter and text from markdown content."""
        if markdown_content.startswith('---\n'):
            try:
                # Find the closing --- delimiter
                end_index = markdown_content.find('\n---\n', 4)
                if end_index != -1:
                    front_matter_str = markdown_content[4:end_index]
                    text = markdown_content[end_index + 5:].strip()
                    
                    # Parse YAML
                    front_matter = yaml.safe_load(front_matter_str) or {}
                    return front_matter, text
            except yaml.YAMLError as e:
                logger.warning(f"Failed to parse YAML front matter: {e}")
        
        return {}, markdown_content.strip()
    
    def _parse_front_matter(self, front_matter_dict: Dict[str, Any], text: str) -> Any:
        """Parse front matter dictionary into dataclass instance if front_matter_class is specified."""
        if not self.front_matter_class:
            return front_matter_dict
            
        # Get field names and types from the dataclass
        field_info = {f.name: f.type for f in fields(self.front_matter_class)}
        
        # Validate and convert values
        kwargs = {}
        for field_name, field_type in field_info.items():
            # Special handling for natural_text field in PageResponse
            if field_name == 'natural_text' and self.front_matter_class == PageResponse:
                kwargs[field_name] = text if text else None
                continue
                
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
                
        # Check for extra fields (excluding natural_text if it's PageResponse)
        expected_fields = set(field_info.keys())
        if self.front_matter_class == PageResponse:
            expected_fields.discard('natural_text')
        extra_fields = set(front_matter_dict.keys()) - expected_fields
        if extra_fields:
            raise ValueError(f"Unexpected fields in front matter: {extra_fields}")
            
        return self.front_matter_class(**kwargs)
    
    def __call__(self, sample: Sample) -> Sample:
        """Parse front matter from markdown content."""
        # Read markdown content if not already loaded
        if 'markdown_content' not in sample:
            sample['markdown_content'] = sample['markdown_path'].read_text(encoding='utf-8')
        
        # Extract and parse front matter
        front_matter, text = self._extract_front_matter_and_text(sample['markdown_content'])
        
        # Parse front matter to dataclass if specified
        try:
            page_data = self._parse_front_matter(front_matter, text)
        except Exception as e:
            raise ValueError(f"Error parsing front matter for {sample['markdown_path']}: {e}")
        
        # Only add page_data field
        sample['page_data'] = page_data
        
        return sample


@dataclass(frozen=True, slots=True)
class PDFRenderer(PipelineStep):
    """Pipeline step that renders PDF to image."""
    target_longest_image_dim: int
    image_transform: Optional[Callable] = None
    
    def __call__(self, sample: Sample) -> Sample:
        """Render PDF to image."""
        # Render PDF to image
        base64_png = render_pdf_to_base64png(
            str(sample['pdf_path']), 
            page_num=1, 
            target_longest_image_dim=self.target_longest_image_dim
        )
        png_bytes = base64.b64decode(base64_png)
        image = Image.open(BytesIO(png_bytes))
        
        # Apply transform if provided
        if self.image_transform:
            image = self.image_transform(image)
        
        # Update sample
        sample['image'] = image
        
        return sample
    

@dataclass(frozen=True, slots=True)
class StaticLengthDocumentAnchoring(PipelineStep):
    target_anchor_text_len: int

    """Pipeline step that runs document anchoring on the PDF and puts in the data to be used by later prompting stages"""
    def __call__(self, sample: Sample) -> Sample:
        anchor_text = get_anchor_text(sample["pdf_path"], page=1, pdf_engine="pdfreport", target_length=self.target_anchor_text_len)
        sample["anchor_text"] = anchor_text
        return sample
    

@dataclass(frozen=True, slots=True)
class FinetuningPrompt(PipelineStep):
    """Applies the standard fine tuning prompt"""
    def __call__(self, sample: Sample) -> Sample:
        sample["instruction_prompt"] = build_finetuning_prompt(sample["anchor_text"])
        return sample
    

@dataclass(frozen=True, slots=True)
class FrontMatterOutputFormat(PipelineStep):
    """Takes the output and applies the standard yaml formatting to it"""
    def __call__(self, sample: Sample) -> Sample:
        page_data = sample["page_data"]
        assert type(page_data) == PageResponse
        
        sample["output"] = f"""---
primary_language: {page_data.primary_language}
is_rotation_valid: {page_data.is_rotation_valid}
rotation_correction: {page_data.rotation_correction}
is_table: {page_data.is_table}
is_diagram: {page_data.is_diagram}
---
{page_data.natural_text}
""".strip()
        
        return sample
    

@dataclass(frozen=True, slots=True)
class InstructMessages(PipelineStep):
    """Creates instruction-following messages format for training."""
    def __call__(self, sample: Sample) -> Sample:
        # Convert PIL image to base64 string
        if 'image' in sample:
            buffered = BytesIO()
            sample['image'].save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        else:
            raise ValueError("Image not found in sample. Make sure PDFRenderer runs before InstructMessages.")
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": base64_image},
                    {"type": "text", "text": sample["instruction_prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": sample["output"]
            }
        ]
        sample["messages"] = messages

        return sample
    

class MarkdownPDFDocumentDataset(BaseMarkdownPDFDataset):
    """Dataset that includes front matter parsing and PDF rendering by default."""
    
    def __init__(self, root_dir: str | PathLike, target_longest_image_dim: int, image_transform=None, front_matter_class=None):
        """
        Initialize the dataset with default pipeline steps.
        
        Args:
            root_dir: Path to the root folder containing processed markdown and PDF files
            target_longest_image_dim: Target dimension for the longest side of the image
            image_transform: Optional transform to apply to the PDF images
            front_matter_class: Optional dataclass type to validate front matter against
        """
        # Create default pipeline steps
        pipeline_steps = [
            FrontMatterParser(front_matter_class),
            PDFRenderer(target_longest_image_dim, image_transform),
            StaticLengthDocumentAnchoring(target_anchor_text_len=6000),
            FinetuningPrompt(),
            FrontMatterOutputFormat(),
            InstructMessages(),
        ]
        
        # Initialize base class with pipeline
        super().__init__(root_dir, pipeline_steps)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Test MarkdownPDFDocumentDataset")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/home/ubuntu/olmOCR-mix-0225/processed_00_documents_eval_s2pdf/",
        help="Root directory containing processed markdown and PDF files"
    )
    
    args = parser.parse_args()
    
    # Quick test to ensure dataset loads
    print(f"\n=== Testing dataset loading ===")
    base_dataset = BaseMarkdownPDFDataset(args.root_dir)
    print(f"Found {len(base_dataset)} markdown-PDF pairs")
    
    # Test the convenience dataset class
    print(f"\n=== Testing MarkdownPDFDocumentDataset (convenience class) ===")
    dataset = MarkdownPDFDocumentDataset(
        args.root_dir, 
        target_longest_image_dim=1024, 
        front_matter_class=PageResponse, 
        image_transform=None
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # Show first few samples
        print("\nFirst 5 samples:")
        for i in range(min(5, len(dataset))):
            sample = dataset.samples[i]
            print(f"  {i}: MD: {sample['markdown_path'].name}, PDF: {sample['pdf_path'].name}")
            
        # Test __getitem__
        print("\nTesting __getitem__ on first sample:")
        first_sample = dataset[0]
        
        # Pretty print the message structure
        print("\n=== Message Structure ===")
        messages = first_sample['messages']
        
        for i, msg in enumerate(messages):
            print(f"\nMessage {i + 1}:")
            print(f"  role: {msg['role']}")
            
            if msg['role'] == 'user':
                print("  content:")
                for j, content_item in enumerate(msg['content']):
                    if content_item['type'] == 'image':
                        # Show that there's an image without the base64 data
                        image_data = content_item['image']
                        print(f"    [{j}] type: image")
                        print(f"        image: <base64 PNG data, {len(image_data)} chars>")
                    elif content_item['type'] == 'text':
                        text_preview = content_item['text'][:200].replace('\n', '\n            ')
                        print(f"    [{j}] type: text")
                        print(f"        text: {text_preview}...")
            else:
                # Assistant message
                content_preview = msg['content'][:300].replace('\n', '\n        ')
                print(f"  content: {content_preview}...")
        
        print("\n=== Sample Metadata ===")
        print(f"PDF: {Path(first_sample['pdf_path']).name}")
        print(f"Image size: {first_sample['image'].size}")
        print(f"Page data: {first_sample['page_data']}")
    else:
        raise AssertionError("Expected some data to be created at this point")