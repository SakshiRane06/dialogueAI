"""
Test Demo for DialogueAI System

This demonstrates the core RAG functionality without requiring OpenAI API keys.
We'll use SentenceTransformers for embeddings and show how the system processes
documents, creates embeddings, and retrieves relevant context.
"""

import sys
import os
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, 'src')

from src.document_processor import DocumentProcessor
from src.rag_system import RAGSystem
from rich.console import Console

console = Console()

def main():
    console.print("🚀 [bold blue]DialogueAI Demo Test[/bold blue]", style="blue")
    console.print("Testing RAG system with SentenceTransformers (no API keys needed)\n")
    
    try:
        # 1. Test Document Processing
        console.print("📄 [bold]Step 1: Document Processing[/bold]")
        processor = DocumentProcessor()
        
        # Test if sample file exists
        sample_file = Path("data/sample.txt")
        if not sample_file.exists():
            console.print("❌ Sample file not found!", style="red")
            return
        
        # Process the document
        content = processor.process_document(sample_file)
        cleaned_content = processor.clean_text(content)
        
        console.print(f"✅ Document processed: {len(cleaned_content)} characters")
        console.print(f"Preview: {cleaned_content[:200]}...\n")
        
        # 2. Test RAG System
        console.print("🔍 [bold]Step 2: RAG System Setup[/bold]")
        
        # Use SentenceTransformers instead of OpenAI
        rag = RAGSystem(use_openai=False, chunk_size=500, chunk_overlap=100)
        
        # Process document into chunks
        docs = rag.process_document(cleaned_content, source="sample.txt")
        console.print(f"✅ Created {len(docs)} chunks")
        
        # Create vector store
        vector_store = rag.create_vector_store(docs)
        console.print("✅ Vector store created with embeddings")
        
        # 3. Test Retrieval
        console.print("\n🎯 [bold]Step 3: Testing Retrieval[/bold]")
        
        test_queries = [
            "What is RAG?",
            "How does retrieval work?", 
            "What are the benefits of RAG?",
            "What challenges does RAG face?"
        ]
        
        for query in test_queries:
            console.print(f"\n[yellow]Query:[/yellow] {query}")
            
            # Get relevant chunks
            docs_with_scores = rag.retrieve_relevant_chunks(query, k=2)
            
            for i, (doc, score) in enumerate(docs_with_scores):
                chunk_preview = doc.page_content[:150].replace('\n', ' ')
                console.print(f"  [green]Chunk {i+1}[/green] (score: {score:.3f}): {chunk_preview}...")
        
        # 4. Show System Stats
        console.print(f"\n📊 [bold]Step 4: System Statistics[/bold]")
        stats = rag.get_system_stats()
        for key, value in stats.items():
            console.print(f"  {key}: {value}")
        
        console.print(f"\n🎉 [bold green]Demo Complete![/bold green]")
        console.print("The RAG system is working correctly. To generate dialogues, you'll need:")
        console.print("1. Add OPENAI_API_KEY to .env file")
        console.print("2. Run: python -m src.cli --file data/sample.txt")
        
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    main()
