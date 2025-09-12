#!/usr/bin/env python3
"""
DialogueAI with OpenAI API Demo (Web App Version)

This script demonstrates how to use the web app's OpenAI integration
to generate dialogues when Gemini API is not working.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console

# Add src to path for imports
sys.path.insert(0, 'src')

# Import the WebDialogueGenerator from web_app.py
from web_app import WebDialogueGenerator

console = Console()

# Load environment variables
load_dotenv()

# Check if OpenAI API key is set
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    console.print("❌ OPENAI_API_KEY not found in .env file", style="red")
    console.print("Please add your OpenAI API key to the .env file:", style="yellow")
    console.print("OPENAI_API_KEY=your_api_key_here", style="yellow")
    sys.exit(1)

# Initialize WebDialogueGenerator
console.print("🚀 Initializing WebDialogueGenerator...", style="blue")
try:
    # Create WebDialogueGenerator instance
    dialogue_gen = WebDialogueGenerator()
    console.print("✅ WebDialogueGenerator initialized successfully", style="green")
    
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
    
    # Process document
    console.print("🔍 Processing document...", style="blue")
    content, stats, success = dialogue_gen.process_document(sample_file)
    
    if not success:
        console.print(f"❌ Error processing document: {content}", style="red")
        sys.exit(1)
    
    console.print("📊 Document stats:", style="green")
    for key, value in stats.items():
        console.print(f"  - {key}: {value}", style="green")
    
    # Generate dialogue using OpenAI
    console.print("🗣️ Generating dialogue with OpenAI...", style="blue")
    user_goal = "Explain what RAG is in simple terms"
    tone = "conversational"
    difficulty = "beginner"
    
    dialogue, success = dialogue_gen.generate_dialogue(user_goal, tone, difficulty)
    
    if not success:
        console.print(f"❌ Error generating dialogue: {dialogue}", style="red")
        sys.exit(1)
    
    # Display the generated dialogue
    console.print("\n🎭 Generated Dialogue:\n", style="bold green")
    console.print(dialogue)
    
    console.print("\n✨ Success! Dialogue generated using OpenAI API", style="bold green")
    console.print("\n📝 Note: We're using OpenAI because the Gemini API integration is not working correctly with the current Python version.", style="yellow")
    console.print("The error 'type' object is not subscriptable' suggests a compatibility issue between the Google Generative AI package and Python 3.8.", style="yellow")
    
except Exception as e:
    console.print(f"❌ Error: {str(e)}", style="red")
    console.print("Make sure you have the correct API keys and the required packages installed.", style="yellow")
    sys.exit(1)