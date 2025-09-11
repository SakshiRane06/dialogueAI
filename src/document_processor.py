"""
Document Processor Module

Handles extraction and cleaning of text from PDF and TXT files.
"""

import os
from pathlib import Path
from typing import List, Optional, Union
import fitz  # PyMuPDF
import pdfplumber
from rich.console import Console
from rich.progress import track

console = Console()


class DocumentProcessor:
    """
    Processes documents (PDF, TXT) and extracts clean text content.
    """
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.docx']
    
    def process_document(self, file_path: Union[str, Path]) -> str:
        """
        Process a document and return cleaned text content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Cleaned text content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        console.print(f"📄 Processing {file_path.name}...", style="blue")
        
        if file_ext == '.pdf':
            return self._process_pdf(file_path)
        elif file_ext == '.txt':
            return self._process_txt(file_path)
        elif file_ext == '.docx':
            return self._process_docx(file_path)
    
    def _process_pdf(self, file_path: Path) -> str:
        """
        Extract text from PDF using PyMuPDF with fallback to pdfplumber.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            # Primary method: PyMuPDF (faster)
            return self._extract_with_pymupdf(file_path)
        except Exception as e:
            console.print(f"⚠️  PyMuPDF failed, trying pdfplumber: {e}", style="yellow")
            # Fallback method: pdfplumber (better for complex layouts)
            return self._extract_with_pdfplumber(file_path)
    
    def _extract_with_pymupdf(self, file_path: Path) -> str:
        """Extract text using PyMuPDF."""
        text_content = []
        
        with fitz.open(file_path) as doc:
            for page_num in track(range(len(doc)), description="Extracting pages..."):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():  # Only add non-empty pages
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}")
        
        return "\n\n".join(text_content)
    
    def _extract_with_pdfplumber(self, file_path: Path) -> str:
        """Extract text using pdfplumber (better for tables/complex layouts)."""
        text_content = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num in track(range(len(pdf.pages)), description="Extracting pages..."):
                page = pdf.pages[page_num]
                text = page.extract_text()
                
                if text and text.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}")
        
        return "\n\n".join(text_content)
    
    def _process_txt(self, file_path: Path) -> str:
        """
        Process plain text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            File content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except UnicodeDecodeError:
            # Fallback to other encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                    console.print(f"📝 Used {encoding} encoding", style="yellow")
                    return content
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode text file: {file_path}")
    
    def _process_docx(self, file_path: Path) -> str:
        """
        Process DOCX file (requires python-docx).
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Extracted text content
        """
        try:
            from docx import Document
            
            doc = Document(file_path)
            text_content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            return "\n\n".join(text_content)
        
        except ImportError:
            raise ImportError("python-docx is required for DOCX files. Install with: pip install python-docx")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Keep non-empty lines
                cleaned_lines.append(line)
        
        # Join with single newlines and normalize spacing
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove multiple consecutive newlines
        import re
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        return cleaned_text
    
    def get_document_info(self, file_path: Union[str, Path]) -> dict:
        """
        Get basic information about the document.
        
        Args:
            file_path: Path to document
            
        Returns:
            Dictionary with document metadata
        """
        file_path = Path(file_path)
        
        info = {
            'filename': file_path.name,
            'size_bytes': file_path.stat().st_size,
            'format': file_path.suffix.lower(),
            'size_mb': round(file_path.stat().st_size / (1024 * 1024), 2)
        }
        
        # For PDFs, get page count
        if file_path.suffix.lower() == '.pdf':
            try:
                with fitz.open(file_path) as doc:
                    info['pages'] = len(doc)
            except:
                info['pages'] = 'Unknown'
        
        return info


# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Test with a sample file (replace with your own)
    test_file = Path("data/sample.pdf")
    
    if test_file.exists():
        try:
            content = processor.process_document(test_file)
            cleaned_content = processor.clean_text(content)
            
            print(f"Document length: {len(cleaned_content)} characters")
            print(f"First 500 characters:\n{cleaned_content[:500]}...")
            
            # Get document info
            info = processor.get_document_info(test_file)
            print(f"Document info: {info}")
            
        except Exception as e:
            print(f"Error processing document: {e}")
    else:
        print("No test file found. Place a PDF in data/sample.pdf to test.")
