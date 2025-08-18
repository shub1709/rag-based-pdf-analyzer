import os
import re
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd
import camelot
import cv2
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of document content"""
    content: str
    metadata: Dict[str, Any]
    chunk_type: str  # 'text', 'table', 'image'
    page_number: int
    section: Optional[str] = None

class PDFProcessor:
    """Handles PDF processing and content extraction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.extract_images = config.get('pdf_processing', {}).get('extract_images', True)
        self.extract_tables = config.get('pdf_processing', {}).get('extract_tables', True)
        self.ocr_enabled = config.get('pdf_processing', {}).get('ocr_enabled', True)
    
    def process_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Process PDF and extract all content
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        try:
            # Extract text content
            text_chunks = self._extract_text(pdf_path)
            chunks.extend(text_chunks)
            
            # Extract tables
            if self.extract_tables:
                table_chunks = self._extract_tables(pdf_path)
                chunks.extend(table_chunks)
            
            # Extract images and perform OCR
            if self.extract_images and self.ocr_enabled:
                image_chunks = self._extract_images_with_ocr(pdf_path)
                chunks.extend(image_chunks)
            
            # Identify sections
            chunks = self._identify_sections(chunks)
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
        
        return chunks
    
    def _extract_text(self, pdf_path: str) -> List[DocumentChunk]:
        """Extract text content from PDF"""
        chunks = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    
                    if text:
                        # Clean the text
                        text = self._clean_text(text)
                        
                        # Split into paragraphs
                        paragraphs = self._split_into_paragraphs(text)
                        
                        for para in paragraphs:
                            if len(para.strip()) > 50:  # Filter out very short paragraphs
                                chunk = DocumentChunk(
                                    content=para,
                                    metadata={
                                        'source': os.path.basename(pdf_path),
                                        'page': page_num,
                                        'total_pages': len(pdf.pages)
                                    },
                                    chunk_type='text',
                                    page_number=page_num
                                )
                                chunks.append(chunk)
        
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            # Fallback to PyPDF2
            chunks.extend(self._extract_text_pypdf2(pdf_path))
        
        return chunks
    
    def _extract_text_pypdf2(self, pdf_path: str) -> List[DocumentChunk]:
        """Fallback text extraction using PyPDF2"""
        chunks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text:
                        text = self._clean_text(text)
                        paragraphs = self._split_into_paragraphs(text)
                        
                        for para in paragraphs:
                            if len(para.strip()) > 50:
                                chunk = DocumentChunk(
                                    content=para,
                                    metadata={
                                        'source': os.path.basename(pdf_path),
                                        'page': page_num + 1
                                    },
                                    chunk_type='text',
                                    page_number=page_num + 1
                                )
                                chunks.append(chunk)
        
        except Exception as e:
            logger.error(f"Error with PyPDF2 extraction: {str(e)}")
        
        return chunks
    
    def _extract_tables(self, pdf_path: str) -> List[DocumentChunk]:
        """Extract tables from PDF"""
        chunks = []
        
        try:
            # Use camelot for table extraction
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
            
            for i, table in enumerate(tables):
                df = table.df
                
                if not df.empty:
                    # Convert table to markdown format
                    table_text = self._table_to_markdown(df)
                    
                    chunk = DocumentChunk(
                        content=table_text,
                        metadata={
                            'source': os.path.basename(pdf_path),
                            'table_index': i,
                            'page': table.page
                        },
                        chunk_type='table',
                        page_number=table.page
                    )
                    chunks.append(chunk)
        
        except Exception as e:
            logger.warning(f"Error extracting tables with camelot: {str(e)}")
            # Fallback to pdfplumber tables
            chunks.extend(self._extract_tables_pdfplumber(pdf_path))
        
        return chunks
    
    def _extract_tables_pdfplumber(self, pdf_path: str) -> List[DocumentChunk]:
        """Fallback table extraction using pdfplumber"""
        chunks = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    
                    for i, table in enumerate(tables):
                        if table:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            table_text = self._table_to_markdown(df)
                            
                            chunk = DocumentChunk(
                                content=table_text,
                                metadata={
                                    'source': os.path.basename(pdf_path),
                                    'table_index': i,
                                    'page': page_num
                                },
                                chunk_type='table',
                                page_number=page_num
                            )
                            chunks.append(chunk)
        
        except Exception as e:
            logger.error(f"Error extracting tables with pdfplumber: {str(e)}")
        
        return chunks
    
    def _extract_images_with_ocr(self, pdf_path: str) -> List[DocumentChunk]:
        """Extract images from PDF and perform OCR"""
        chunks = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract images from page
                    if hasattr(page, 'images'):
                        for img_index, img in enumerate(page.images):
                            try:
                                # Perform OCR on image
                                ocr_text = self._perform_ocr_on_image(img)
                                
                                if ocr_text and len(ocr_text.strip()) > 20:
                                    chunk = DocumentChunk(
                                        content=f"[Image Content]: {ocr_text}",
                                        metadata={
                                            'source': os.path.basename(pdf_path),
                                            'page': page_num,
                                            'image_index': img_index
                                        },
                                        chunk_type='image',
                                        page_number=page_num
                                    )
                                    chunks.append(chunk)
                            
                            except Exception as e:
                                logger.warning(f"Error processing image: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error extracting images: {str(e)}")
        
        return chunks
    
    def _perform_ocr_on_image(self, image_obj) -> str:
        """Perform OCR on an image object"""
        try:
            # Convert image object to PIL Image
            # This is a simplified version - actual implementation would need proper image extraction
            # For now, return empty string
            return ""
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
        return text.strip()
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines or periods followed by capital letters
        paragraphs = re.split(r'\n\n|\. (?=[A-Z])', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _table_to_markdown(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to markdown table format"""
        return df.to_markdown(index=False)
    
    def _identify_sections(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Identify sections in the document based on headings"""
        current_section = "Introduction"
        section_patterns = [
            r'^(?:Chapter|Section|\d+\.)\s+(.+)$',
            r'^([A-Z][A-Z\s]+)$',  # All caps headings
            r'^(\d+\.\d+\.?\s+.+)$'  # Numbered sections
        ]
        
        for chunk in chunks:
            if chunk.chunk_type == 'text':
                # Check if this chunk is a heading
                for pattern in section_patterns:
                    match = re.match(pattern, chunk.content[:100])
                    if match:
                        current_section = match.group(1)
                        break
                
                chunk.section = current_section
        
        return chunks