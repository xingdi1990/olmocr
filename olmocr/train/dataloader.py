from os import PathLike
from pathlib import Path
from typing import Dict, Any
import base64
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset

from olmocr.data.renderpdf import render_pdf_to_base64png


class MarkdownPDFDocumentDataset(Dataset):
    def __init__(self, root_dir: str | PathLike, transform=None):
        """
        Initialize the dataset by finding all markdown files with corresponding PDFs.
        
        Args:
            root_dir: Path to the root folder containing processed markdown and PDF files
            transform: Optional transform to apply to the PDF images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        # Find all markdown files recursively
        print(f"Scanning for markdown files in {self.root_dir}...")
        md_files = list(self.root_dir.rglob("*.md"))
        
        # Verify each markdown file has a corresponding PDF
        for md_path in md_files:
            # Look for PDF with same stem (filename without extension)
            pdf_path = md_path.with_suffix('.pdf')
            
            if pdf_path.exists() or pdf_path.is_symlink():
                # Resolve symlink if it is one
                if pdf_path.is_symlink():
                    pdf_path = pdf_path.resolve()
                    
                # Verify the resolved path exists
                if pdf_path.exists():
                    self.samples.append({
                        'markdown_path': md_path,
                        'pdf_path': pdf_path
                    })
        
        print(f"Found {len(self.samples)} valid markdown-PDF pairs")
    
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
        
        # Read markdown file
        markdown_content = sample['markdown_path'].read_text(encoding='utf-8')
        
        # Parse front matter and extract text
        front_matter = {}
        text = markdown_content
        
        if markdown_content.startswith('---\n'):
            # Find the closing --- for front matter
            parts = markdown_content.split('---\n', 2)
            if len(parts) >= 3:
                # Parse front matter
                front_matter_text = parts[1]
                for line in front_matter_text.strip().split('\n'):
                    if ': ' in line:
                        key, value = line.split(': ', 1)
                        # Try to parse boolean values
                        if value.lower() == 'true':
                            front_matter[key] = True
                        elif value.lower() == 'false':
                            front_matter[key] = False
                        else:
                            front_matter[key] = value
                
                # Get text without front matter
                text = parts[2].strip()
        
        # Render PDF to image
        base64_png = render_pdf_to_base64png(str(sample['pdf_path']), page_num=1)
        png_bytes = base64.b64decode(base64_png)
        image = Image.open(BytesIO(png_bytes))
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'pdf_path': str(sample['pdf_path']),
            'text': text,
            'front_matter': front_matter
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
    dataset = MarkdownPDFDocumentDataset(args.root_dir)
    
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
        print(f"Text preview (first 200 chars): {first_sample['text'][:200]}...")
        
        # Test with transforms
        print("\nTesting with torchvision transforms:")
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        
        dataset_with_transform = MarkdownPDFDocumentDataset(args.root_dir, transform=transform)
        transformed_sample = dataset_with_transform[0]
        print(f"Transformed image type: {type(transformed_sample['image'])}")
        print(f"Transformed image shape: {transformed_sample['image'].shape}")