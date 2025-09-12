#!/usr/bin/env python3
"""
DialogueAI with Gemini API Demo

This script demonstrates how to use the GeminiDialogueAI class
to generate dialogues using Google's Gemini API instead of OpenAI.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console

# Add src to path for imports
sys.path.insert(0, 'src')

from src.gemini_dialogue_ai import GeminiDialogueAI

console = Console()

# Load environment variables
load_dotenv()

# Check if Google API key is set
google_api_key = os.getenv('GOOGLE_API_KEY')
if not google_api_key:
    console.print("❌ GOOGLE_API_KEY not found in .env file", style="red")
    console.print("Please add your Google AI API key to the .env file:", style="yellow")
    console.print("GOOGLE_API_KEY=your_api_key_here", style="yellow")
    sys.exit(1)

# Initialize GeminiDialogueAI
console.print("🚀 Initializing GeminiDialogueAI...", style="blue")
try:
    # Create GeminiDialogueAI instance
    dialogue_ai = GeminiDialogueAI(
        tone="conversational",
        level="intermediate",
        model_name="gemini-pro",
        temperature=0.7,
        chunk_size=1000,
        chunk_overlap=200,
        use_openai_embeddings=False  # Use SentenceTransformers for embeddings
    )
    console.print("✅ GeminiDialogueAI initialized successfully", style="green")
    
    # Process a sample document
    sample_file = "data/sample.txt"
    if not Path(sample_file).exists():
        console.print(f"❌ Sample file {sample_file} not found", style="red")
        console.print("Creating a simple sample file...", style="yellow")
        
        # Create a simple sample file
        os.makedirs("data", exist_ok=True)
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("""# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a powerful technique that combines the strengths of retrieval-based and generation-based approaches in natural language processing.

RAG operates in two main phases:
1. Retrieval Phase: When given a query or prompt, the system searches through a knowledge base to find relevant documents.
2. Generation Phase: The retrieved documents are then used as context for a language model to generate a response.

This approach has several advantages:
- It grounds the model's responses in specific documents, reducing hallucinations
- It allows the model to access knowledge beyond its training data
- It provides citations and references for generated content
- It can be updated with new information without retraining the entire model
""")
        console.print(f"✅ Created sample file: {sample_file}", style="green")
    
    # Generate dialogue
    console.print("🔍 Processing document and generating dialogue...", style="blue")
    dialogue = dialogue_ai.process_document(
        file_path=sample_file,
        tone="conversational",
        level="beginner"
    )
    
    # Display the generated dialogue
    console.print("\n🎭 Generated Dialogue:\n", style="bold green")
    console.print(dialogue)
    
    console.print("\n✨ Success! Dialogue generated using Google's Gemini API", style="bold green")
    
except Exception as e:
    console.print(f"❌ Error: {str(e)}", style="red")
    console.print("Make sure you have the correct Google API key and the required packages installed.", style="yellow")
    sys.exit(1)