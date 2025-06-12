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
import numpy as np

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
        
        sample["response"] = f"""---
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
class InstructUserMessages(PipelineStep):
    """Creates instruction-following messages format for training."""
    def __call__(self, sample: Sample) -> Sample:
        # Prepare messages
        messages = {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample["instruction_prompt"]},
                ],
            }

        sample["user_messages"] = messages

        return sample
    

@dataclass(frozen=True, slots=True)
class Tokenizer(PipelineStep):
    """Tokenizes messages and creates training labels with proper masking."""
    processor: Any  # The model processor (e.g., AutoProcessor)
    masking_index: int = -100
    
    def __call__(self, sample: Sample) -> Sample:
        """Tokenize messages and create labels for training."""
        if np is None:
            raise ImportError("numpy is required for Tokenizer step")
            
        # Extract user message and response
        user_messages = sample["user_messages"]
        response = sample["response"]

        # Apply chat template to user message only with generation prompt
        # user_messages is a single dict, so wrap it in a list
        text = self.processor.apply_chat_template(
            [user_messages], 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        main_image = user_messages["content"][0]["image"]
        
        # Process inputs using processor
        inputs = self.processor(
            text=[text],
            images=[main_image],
            padding=True,
            return_tensors="np",
        )
        
        # Get labels by tokenizing the output text
        labels = self.processor(text=[response], padding=True, return_tensors="np")
        
        # Append <|im_end|>\n to the labels
        im_end_tokens = self.processor.tokenizer("<|im_end|>\n", add_special_tokens=False)["input_ids"]
        im_end_tokens = np.array(im_end_tokens, dtype=inputs.input_ids.dtype)
        
        # Handle the case where labels['input_ids'] is empty
        if labels["input_ids"].shape[1] == 0:
            labels_input_ids_0 = np.array([], dtype=inputs.input_ids.dtype)
        else:
            labels_input_ids_0 = labels["input_ids"][0].astype(inputs.input_ids.dtype)
        
        labels["input_ids"] = np.concatenate([labels_input_ids_0, im_end_tokens])
        labels["input_ids"] = np.expand_dims(labels["input_ids"], axis=0)
        
        # Concatenate input_ids and labels
        input_ids = np.concatenate([inputs.input_ids[0], labels.input_ids[0]], axis=0)
        
        # All columns will participate in attention fully
        attention_mask = np.ones_like(input_ids)
        
        # Create labels, masking the input portion with -100
        labels_full = np.full_like(input_ids, fill_value=self.masking_index)
        labels_full[len(inputs.input_ids[0]):] = labels.input_ids[0]
        
        # Return as dict, including pixel_values
        sample["input_ids"] = input_ids
        sample["attention_mask"] = attention_mask
        sample["labels"] = labels_full
        sample["pixel_values"] = inputs.pixel_values
        
        if hasattr(inputs, 'image_grid_thw'):
            sample["image_grid_thw"] = inputs.image_grid_thw[0]
            
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
            InstructUserMessages(),
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
        # TODO
        
        print("\n=== Sample Metadata ===")
        print(f"PDF: {Path(first_sample['pdf_path']).name}")
        print(f"Image size: {first_sample['image'].size}")
        print(f"Page data: {first_sample['page_data']}")
        
        # Test with actual Qwen2.5-VL tokenization
        print("\n\n=== Testing with Qwen2.5-VL-7B-Instruct Tokenization ===")
        
        try:
            from transformers import AutoProcessor
            
            print("Loading Qwen2.5-VL processor...")
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            
            # Create pipeline with real tokenizer
            tokenized_dataset = BaseMarkdownPDFDataset(
                args.root_dir,
                pipeline_steps=[
                    FrontMatterParser(front_matter_class=PageResponse),
                    PDFRenderer(target_longest_image_dim=512),
                    StaticLengthDocumentAnchoring(target_anchor_text_len=1000),
                    FinetuningPrompt(),
                    FrontMatterOutputFormat(),
                    InstructUserMessages(),
                    Tokenizer(processor),
                ]
            )
            
            if len(tokenized_dataset) > 0:
                print("\nProcessing first sample with Qwen2.5-VL...")
                tokenized_sample = tokenized_dataset[0]
                
                print("\nTokenized output:")
                print(f"  Keys: {list(tokenized_sample.keys())}")
                print(f"  Input IDs shape: {tokenized_sample['input_ids'].shape}")
                print(f"  Labels shape: {tokenized_sample['labels'].shape}")
                print(f"  Attention mask shape: {tokenized_sample['attention_mask'].shape}")
                
                if 'pixel_values' in tokenized_sample:
                    print(f"  Pixel values shape: {tokenized_sample['pixel_values'].shape}")
                if 'image_grid_thw' in tokenized_sample:
                    print(f"  Image grid THW: {tokenized_sample['image_grid_thw']}")
                
                # Show label masking
                print(f"\nLabel masking analysis:")
                labels = tokenized_sample['labels']
                masked_count = np.sum(labels == -100)
                total_count = len(labels)
                print(f"  Total tokens: {total_count}")
                print(f"  Masked tokens: {masked_count} ({masked_count/total_count*100:.1f}%)")
                print(f"  Unmasked tokens: {total_count - masked_count} ({(total_count - masked_count)/total_count*100:.1f}%)")
                
                # Find the transition point
                transition_idx = None
                for i in range(len(labels) - 1):
                    if labels[i] == -100 and labels[i + 1] != -100:
                        transition_idx = i + 1
                        break
                
                if transition_idx:
                    print(f"  Transition from masked to unmasked at position: {transition_idx}")
                
                # Print all tokens
                input_ids = tokenized_sample['input_ids']
                print(f"\nAll tokens ({len(input_ids)} total):")
                print("Format: [index] Token (repr) | Label | Token ID")
                print("-" * 80)
                
                for i in range(len(input_ids)):
                    token = processor.tokenizer.decode([input_ids[i]])
                    token_repr = repr(token)
                    label = labels[i] if i < len(labels) else "N/A"
                    token_id = input_ids[i]
                    
                    # Mark special positions
                    marker = ""
                    if transition_idx and i == transition_idx:
                        marker = " <-- TRANSITION (first unmasked)"
                    elif i == 0:
                        marker = " <-- START"
                    elif label != -100 and i > 0 and labels[i-1] == -100:
                        marker = " <-- response begins"
                    
                    print(f"[{i:4d}] {token_repr:20s} | {str(label):6s} | {token_id:6d}{marker}")
                
        except ImportError as e:
            print(f"\nCould not import transformers: {e}")
            print("Install with: pip install transformers")
        except Exception as e:
            print(f"\nError during tokenization test: {e}")
            import traceback
            traceback.print_exc()
        
    else:
        raise AssertionError("Expected some data to be created at this point")