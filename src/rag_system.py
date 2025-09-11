"""
RAG System Module

Implements Retrieval-Augmented Generation with document chunking, 
embeddings, and vector database functionality.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pickle
import numpy as np
from dataclasses import dataclass

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

# Alternative embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from rich.console import Console
from rich.progress import track
import tiktoken

console = Console()


@dataclass
class ChunkMetadata:
    """Metadata for document chunks."""
    source: str
    page_number: Optional[int] = None
    chunk_index: int = 0
    chunk_size: int = 0
    total_chunks: int = 0


class RAGSystem:
    """
    Retrieval-Augmented Generation system for document processing and retrieval.
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_openai: bool = True,
        vector_store_path: Optional[str] = None
    ):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model: Model name for embeddings
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            use_openai: Whether to use OpenAI embeddings
            vector_store_path: Path to save/load vector store
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_path = vector_store_path or "data/vector_store"
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embeddings
        self.use_openai = use_openai
        self.embedding_model = embedding_model
        self.embeddings = self._initialize_embeddings()
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.vector_store = None
        self.document_chunks = []
    
    def _initialize_embeddings(self):
        """Initialize embedding model based on configuration."""
        if self.use_openai:
            try:
                return OpenAIEmbeddings(model=self.embedding_model)
            except Exception as e:
                console.print(f"⚠️  OpenAI embeddings failed: {e}", style="yellow")
                console.print("Falling back to SentenceTransformers...", style="yellow")
                self.use_openai = False
        
        if not self.use_openai:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "SentenceTransformers not available. Install with: pip install sentence-transformers"
                )
            
            # Use a good general-purpose model
            model_name = "all-MiniLM-L6-v2"  # Fast and good quality
            console.print(f"🔤 Loading SentenceTransformers model: {model_name}", style="blue")
            
            return SentenceTransformersEmbeddings(model_name=model_name)
    
    def process_document(self, text: str, source: str) -> List[Document]:
        """
        Process document text into chunks with metadata.
        
        Args:
            text: Document text content
            source: Source identifier (filename, etc.)
            
        Returns:
            List of LangChain Document objects with chunks
        """
        console.print(f"📝 Chunking document: {source}", style="blue")
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create Document objects with metadata
        documents = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            # Count tokens in chunk
            token_count = len(self.tokenizer.encode(chunk))
            
            metadata = ChunkMetadata(
                source=source,
                chunk_index=i,
                chunk_size=len(chunk),
                total_chunks=total_chunks
            ).__dict__
            
            metadata['token_count'] = token_count
            
            doc = Document(
                page_content=chunk,
                metadata=metadata
            )
            documents.append(doc)
        
        console.print(f"✅ Created {len(documents)} chunks", style="green")
        return documents
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vector store from documents.
        
        Args:
            documents: List of LangChain documents
            
        Returns:
            FAISS vector store
        """
        console.print("🔍 Creating vector embeddings...", style="blue")
        
        try:
            # Create vector store
            vector_store = FAISS.from_documents(documents, self.embeddings)
            self.vector_store = vector_store
            self.document_chunks = documents
            
            console.print(f"✅ Vector store created with {len(documents)} embeddings", style="green")
            return vector_store
            
        except Exception as e:
            console.print(f"❌ Error creating vector store: {e}", style="red")
            raise
    
    def save_vector_store(self, path: Optional[str] = None) -> None:
        """
        Save vector store to disk.
        
        Args:
            path: Optional custom path to save
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save. Create one first.")
        
        save_path = Path(path or self.vector_store_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        console.print(f"💾 Saving vector store to {save_path}", style="blue")
        
        # Save FAISS index
        self.vector_store.save_local(str(save_path))
        
        # Save additional metadata
        metadata_path = save_path / "metadata.pkl"
        metadata = {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'embedding_model': self.embedding_model,
            'use_openai': self.use_openai,
            'total_chunks': len(self.document_chunks)
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        console.print("✅ Vector store saved successfully", style="green")
    
    def load_vector_store(self, path: Optional[str] = None) -> FAISS:
        """
        Load vector store from disk.
        
        Args:
            path: Optional custom path to load from
            
        Returns:
            Loaded FAISS vector store
        """
        load_path = Path(path or self.vector_store_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        
        console.print(f"📂 Loading vector store from {load_path}", style="blue")
        
        try:
            # Load FAISS index
            vector_store = FAISS.load_local(str(load_path), self.embeddings)
            self.vector_store = vector_store
            
            # Load metadata if available
            metadata_path = load_path / "metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                console.print(f"📊 Loaded {metadata['total_chunks']} chunks", style="green")
            
            console.print("✅ Vector store loaded successfully", style="green")
            return vector_store
            
        except Exception as e:
            console.print(f"❌ Error loading vector store: {e}", style="red")
            raise
    
    def retrieve_relevant_chunks(
        self, 
        query: str, 
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            k: Number of chunks to retrieve
            score_threshold: Minimum similarity score
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Create or load one first.")
        
        console.print(f"🔎 Retrieving top {k} chunks for query: '{query[:50]}...'", style="blue")
        
        try:
            # Get documents with scores
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, k=k
            )
            
            # Apply score threshold if provided
            if score_threshold is not None:
                docs_with_scores = [
                    (doc, score) for doc, score in docs_with_scores
                    if score >= score_threshold
                ]
            
            console.print(f"✅ Retrieved {len(docs_with_scores)} relevant chunks", style="green")
            return docs_with_scores
            
        except Exception as e:
            console.print(f"❌ Error retrieving chunks: {e}", style="red")
            raise
    
    def get_context_for_query(self, query: str, max_tokens: int = 3000) -> str:
        """
        Get combined context from relevant chunks for a query.
        
        Args:
            query: Search query
            max_tokens: Maximum tokens in context
            
        Returns:
            Combined context string
        """
        # Retrieve relevant chunks
        docs_with_scores = self.retrieve_relevant_chunks(query)
        
        context_parts = []
        token_count = 0
        
        for doc, score in docs_with_scores:
            chunk_tokens = len(self.tokenizer.encode(doc.page_content))
            
            if token_count + chunk_tokens > max_tokens:
                break
            
            # Add chunk with metadata
            source = doc.metadata.get('source', 'unknown')
            chunk_idx = doc.metadata.get('chunk_index', 0)
            
            context_part = f"[Source: {source}, Chunk {chunk_idx + 1}]\n{doc.page_content}"
            context_parts.append(context_part)
            token_count += chunk_tokens
        
        context = "\n\n---\n\n".join(context_parts)
        
        console.print(f"📝 Built context with {token_count} tokens from {len(context_parts)} chunks", style="green")
        return context
    
    def get_system_stats(self) -> Dict:
        """Get statistics about the RAG system."""
        if self.vector_store is None:
            return {"status": "No vector store loaded"}
        
        stats = {
            "total_chunks": len(self.document_chunks) if self.document_chunks else "Unknown",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "use_openai": self.use_openai,
            "vector_store_path": self.vector_store_path
        }
        
        return stats


class SentenceTransformersEmbeddings:
    """
    Wrapper for SentenceTransformers to work with LangChain.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()
    
    def __call__(self, text: str) -> List[float]:
        """Make the class callable for backward compatibility."""
        return self.embed_query(text)


# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem(use_openai=False)  # Use SentenceTransformers for testing
    
    # Example document text
    sample_text = """
    Machine Learning is a subset of artificial intelligence (AI) that provides systems 
    the ability to automatically learn and improve from experience without being explicitly programmed.
    
    Machine learning focuses on the development of computer programs that can access data 
    and use it learn for themselves. The process of learning begins with observations or data, 
    such as examples, direct experience, or instruction, in order to look for patterns in data 
    and make better decisions in the future based on the examples that we provide.
    """
    
    try:
        # Process document
        docs = rag.process_document(sample_text, "sample.txt")
        
        # Create vector store
        vector_store = rag.create_vector_store(docs)
        
        # Test retrieval
        query = "What is machine learning?"
        context = rag.get_context_for_query(query)
        
        print(f"Query: {query}")
        print(f"Context: {context[:500]}...")
        
        # Print stats
        stats = rag.get_system_stats()
        print(f"System stats: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install required dependencies and set up API keys.")
