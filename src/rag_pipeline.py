"""Main RAG pipeline orchestration"""

import os
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass

from src.document_processor import PDFProcessor, DocumentChunk
from src.embeddings_manager import EmbeddingsManager
from src.llm_manager_gemini import LLMManager
from src.utils import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class RAGResult:
    """Represents the result of RAG processing"""
    summary: str
    key_insights: List[str]
    section_summaries: Dict[str, str]
    metadata: Dict[str, Any]

class RAGPipeline:
    """Main RAG pipeline for PDF processing and summarization"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Initialize components
        self.pdf_processor = PDFProcessor(self.config)
        self.embeddings_manager = EmbeddingsManager(self.config)
        self.llm_manager = LLMManager(self.config)
        
        logger.info("RAG Pipeline initialized successfully")
    
    def process_pdf(self, pdf_path: str, clear_existing: bool = True) -> RAGResult:
        """
        Process a PDF file through the complete RAG pipeline
        
        Args:
            pdf_path: Path to the PDF file
            clear_existing: Whether to clear existing vector store
            
        Returns:
            RAGResult object containing summaries and insights
        """
        try:
            logger.info(f"Starting PDF processing for: {pdf_path}")
            
            # Clear existing collection if requested
            if clear_existing:
                self.embeddings_manager.clear_collection()
            
            # Step 1: Extract content from PDF
            logger.info("Extracting content from PDF...")
            chunks = self.pdf_processor.process_pdf(pdf_path)
            logger.info(f"Extracted {len(chunks)} chunks from PDF")
            
            # Step 2: Create embeddings and store in vector database
            logger.info("Creating embeddings...")
            self.embeddings_manager.create_embeddings(chunks)
            
            # Step 3: Generate overall summary
            logger.info("Generating overall summary...")
            overall_summary = self._generate_overall_summary(chunks)
            
            # Step 4: Extract key insights
            logger.info("Extracting key insights...")
            key_insights = self._extract_key_insights(chunks)
            
            # Step 5: Generate section-wise summaries
            logger.info("Generating section summaries...")
            section_summaries = self._generate_section_summaries(chunks)
            
            # Prepare metadata
            metadata = {
                'file_name': os.path.basename(pdf_path),
                'total_chunks': len(chunks),
                'total_pages': max([c.page_number for c in chunks]) if chunks else 0,
                'sections_found': list(section_summaries.keys()),
                'vector_store_stats': self.embeddings_manager.get_collection_stats()
            }
            
            result = RAGResult(
                summary=overall_summary,
                key_insights=key_insights,
                section_summaries=section_summaries,
                metadata=metadata
            )
            
            logger.info("PDF processing completed successfully")
            return result
        
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            raise
    
    def _generate_overall_summary(self, chunks: List[DocumentChunk]) -> str:
        """Generate an overall summary of the document"""
        # Combine text from first few chunks and last few chunks for context
        text_chunks = [c for c in chunks if c.chunk_type == 'text']
        
        if not text_chunks:
            return "No text content found in the document."
        
        # Take first 3 and last 2 chunks for summary
        sample_chunks = text_chunks[:3] + text_chunks[-2:] if len(text_chunks) > 5 else text_chunks
        combined_text = "\n".join([c.content for c in sample_chunks])
        
        # Limit text length
        if len(combined_text) > 5000:
            combined_text = combined_text[:5000]
        
        summary_type = self.config.get('summarization', {}).get('summary_length', 'medium')
        return self.llm_manager.summarize(combined_text, summary_type)
    
    def _extract_key_insights(self, chunks: List[DocumentChunk]) -> List[str]:
        """Extract key insights from the document"""
        # Sample diverse chunks for insight extraction
        text_chunks = [c for c in chunks if c.chunk_type == 'text']
        
        if not text_chunks:
            return ["No insights could be extracted from the document."]
        
        # Sample chunks evenly across the document
        sample_size = min(5, len(text_chunks))
        step = len(text_chunks) // sample_size if sample_size > 0 else 1
        sampled_chunks = text_chunks[::step][:sample_size]
        
        combined_text = "\n".join([c.content for c in sampled_chunks])
        
        # Limit text length
        if len(combined_text) > 4000:
            combined_text = combined_text[:4000]
        
        return self.llm_manager.extract_key_insights(combined_text, num_insights=5)
    
    def _generate_section_summaries(self, chunks: List[DocumentChunk]) -> Dict[str, str]:
        """Generate summaries for each section"""
        section_summaries = {}
        
        # Group chunks by section
        sections = {}
        for chunk in chunks:
            if chunk.section:
                if chunk.section not in sections:
                    sections[chunk.section] = []
                sections[chunk.section].append(chunk)
        
        # Generate summary for each section
        for section_name, section_chunks in sections.items():
            # Combine text from section chunks
            section_text = "\n".join([c.content for c in section_chunks if c.chunk_type == 'text'][:3])
            
            if section_text:
                # Limit text length
                if len(section_text) > 3000:
                    section_text = section_text[:3000]
                
                summary = self.llm_manager.summarize(section_text, summary_type="short")
                section_summaries[section_name] = summary
        
        return section_summaries
    
    def answer_question(self, question: str, k: int = 5) -> str:
        """
        Answer a question using RAG
        
        Args:
            question: The question to answer
            k: Number of relevant chunks to retrieve
            
        Returns:
            Answer to the question
        """
        try:
            # Search for relevant chunks
            relevant_chunks = self.embeddings_manager.similarity_search(question, k=k)
            
            if not relevant_chunks:
                return "No relevant information found in the document to answer this question."
            
            # Combine relevant context
            context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
            
            # Generate answer
            answer = self.llm_manager.answer_question(question, context)
            
            # Add citations if configured
            if self.config.get('summarization', {}).get('include_citations', True):
                citations = []
                for chunk in relevant_chunks:
                    page = chunk['metadata'].get('page', 'Unknown')
                    citations.append(f"Page {page}")
                
                if citations:
                    answer += f"\n\nSources: {', '.join(set(citations))}"
            
            return answer
        
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"Error processing question: {str(e)}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current pipeline statistics"""
        return {
            'vector_store': self.embeddings_manager.get_collection_stats(),
            'llm_model': self.llm_manager.model_name,
            'embedding_model': self.embeddings_manager.model_name,
            'config': {
                'chunk_size': self.config.get('embeddings', {}).get('chunk_size'),
                'chunk_overlap': self.config.get('embeddings', {}).get('chunk_overlap')
            }
        }