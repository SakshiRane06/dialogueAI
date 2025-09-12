#!/usr/bin/env python3
"""
DialogueAI with Gemini API Demo (Web App Version)

This script demonstrates how to use the web app's existing Gemini integration
to generate dialogues using Google's Gemini API instead of OpenAI.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console

# Add src to path for imports
sys.path.insert(0, 'src')

# Import necessary components directly
from src.document_processor import DocumentProcessor
from src.rag_system import RAGSystem, SentenceTransformersEmbeddings

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

# Initialize components
console.print("🚀 Initializing components...", style="blue")
try:
    # Create components
    processor = DocumentProcessor()
    rag = RAGSystem(use_openai=False)  # Use SentenceTransformers
    console.print("✅ Components initialized successfully", style="green")
    
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
    text = processor.process_document(Path(sample_file))
    text = processor.clean_text(text)
    docs = rag.process_document(text, source=os.path.basename(sample_file))
    rag.create_vector_store(docs)
    rag.save_vector_store()
    
    # Get stats
    stats = rag.get_system_stats()
    console.print("📊 Document stats:", style="green")
    for key, value in stats.items():
        console.print(f"  - {key}: {value}", style="green")
    
    # Generate dialogue using Google AI
    console.print("🗣️ Generating dialogue with Google AI...", style="blue")
    user_goal = "Explain what RAG is in simple terms"
    
    # Get context from RAG system
    context = rag.get_context_for_query(user_goal, max_tokens=2500)
    
    # Create a simple function to generate dialogue with Google AI
    import google.generativeai as genai
    
    try:
        # Configure Google AI
        genai.configure(api_key=google_api_key)
        
        # Create the prompt
        prompt = f"""
You are DialogueAI, creating an engaging two-person dialogue between a Learner (👦) and Expert (👨).

Guidelines:
- Tone: conversational
- Difficulty: beginner
- Max turns: 12 (each turn is Learner then Expert)
- Use short, natural conversation
- Start broad, then deepen based on CONTEXT
- Cite sources as [Source: <source>, Chunk <n>] when using context
- Must alternate speakers starting with Learner
- Output format:
👦 Learner: <line>
👨 Expert: <line>

USER GOAL: {user_goal}

CONTEXT (cite when used):
{context}

Generate the dialogue now.
"""
        
        # Generate response
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        dialogue = response.text
        
        # Display the generated dialogue
        console.print("\n🎭 Generated Dialogue:\n", style="bold green")
        console.print(dialogue)
        
        console.print("\n✨ Success! Dialogue generated using Google's Gemini API", style="bold green")
        
    except Exception as e:
        console.print(f"❌ Error generating dialogue with Google AI: {str(e)}", style="red")
        console.print("Falling back to mock dialogue...", style="yellow")
        
        # Generate a mock dialogue as fallback
        dialogue = f"""👦 Learner: I'd like to understand what RAG is in simple terms.

👨 Expert: RAG stands for Retrieval-Augmented Generation. It's like having a smart assistant with access to a library of information.

👦 Learner: That sounds interesting! How does it work?

👨 Expert: It works in two main steps. First, it searches through documents to find relevant information about your question. Then, it uses that information to create a helpful response.

👦 Learner: Why is that better than just generating an answer directly?

👨 Expert: Great question! By retrieving specific information first, RAG can give more accurate and trustworthy answers. It reduces the chance of making things up and can provide sources for its information.

👦 Learner: That makes sense. What are some real-world uses for RAG?

👨 Expert: RAG is used in many AI assistants today. It helps them answer questions about specific documents, company knowledge bases, or technical information with greater accuracy. It's particularly useful when you need precise, factual answers rather than general knowledge.

[Note: This is a demonstration dialogue. Connect your API keys in the .env file for AI-generated content!]"""
        
        console.print("\n🎭 Generated Mock Dialogue:\n", style="bold yellow")
        console.print(dialogue)
    
except Exception as e:
    console.print(f"❌ Error: {str(e)}", style="red")
    console.print("Make sure you have the correct Google API key and the required packages installed.", style="yellow")
    sys.exit(1)