import os
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np

from src.document_processor import DocumentChunk

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    """Manages embeddings generation and vector store operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('embeddings', {}).get('model_name', 'all-MiniLM-L6-v2')
        self.chunk_size = config.get('embeddings', {}).get('chunk_size', 1000)
        self.chunk_overlap = config.get('embeddings', {}).get('chunk_overlap', 200)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.model_name)
        
        # Initialize ChromaDB
        persist_dir = config.get('vector_store', {}).get('persist_directory', './data/vector_store')
        os.makedirs(persist_dir, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection_name = config.get('vector_store', {}).get('collection_name', 'pdf_documents')
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
    
    def create_embeddings(self, chunks: List[DocumentChunk]) -> None:
        """
        Create embeddings for document chunks and store in vector database
        
        Args:
            chunks: List of DocumentChunk objects
        """
        try:
            # Prepare texts and metadata
            texts = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                texts.append(chunk.content)
                
                metadata = {
                    **chunk.metadata,
                    'chunk_type': chunk.chunk_type,
                    'page_number': chunk.page_number
                }
                if chunk.section:
                    metadata['section'] = chunk.section
                
                metadatas.append(metadata)
                ids.append(f"chunk_{i}")
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embedding_model.encode(texts)
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully stored {len(chunks)} chunks in vector database")
        
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search in vector store
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of similar documents with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=filter_dict if filter_dict else None
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection"""
        try:
            # Get all IDs and delete them
            all_ids = self.collection.get()['ids']
            if all_ids:
                self.collection.delete(ids=all_ids)
            logger.info("Collection cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        try:
            collection_data = self.collection.get()
            return {
                'total_documents': len(collection_data['ids']),
                'collection_name': self.collection.name,
                'embedding_model': self.model_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}