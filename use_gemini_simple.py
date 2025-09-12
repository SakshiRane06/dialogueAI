#!/usr/bin/env python3
"""
Simple Gemini API Demo

This script demonstrates how to use Google's Gemini API directly
without relying on the web app's integration.
"""

import os
import sys
from dotenv import load_dotenv
from rich.console import Console

console = Console()

# Load environment variables
load_dotenv()

# Set Google API key directly
google_api_key = "AIzaSyCeTPHxuVgq4rxr1Fs3nlj-N6uny6WbL9A"

# Simple test with Gemini API
console.print("🚀 Testing Gemini API...", style="blue")
try:
    # Import Google AI Generative Language
    import google.ai.generativelanguage as glm
    from google.api_core import client_options
    
    # Create client options with API key
    client_options_obj = client_options.ClientOptions(
        api_key=google_api_key
    )
    
    # Create client
    client = glm.GenerativeServiceClient(client_options=client_options_obj)
    
    # Create a simple prompt
    prompt = """Create a short dialogue between a student and teacher about artificial intelligence.
    Format it as:
    Student: [question]
    Teacher: [answer]
    
    Keep it to 3 exchanges.
    """
    
    # Generate content with text-bison model
    console.print("\n🗣️ Generating dialogue with text-bison model...", style="blue")
    
    request = glm.GenerateTextRequest(
        model="models/text-bison-001",
        prompt=glm.TextPrompt(text=prompt),
        temperature=0.7,
        max_output_tokens=1024
    )
    
    response = client.generate_text(request)
    
    # Display the generated dialogue
    console.print("\n🎭 Generated Dialogue:\n", style="bold green")
    if response.candidates:
        console.print(response.candidates[0].output)
    
    console.print("\n✨ Success! Dialogue generated using Google's Gemini API", style="bold green")
    
except Exception as e:
    console.print(f"❌ Error: {str(e)}", style="red")
    console.print("Make sure you have the correct Google API key and the required packages installed.", style="yellow")
    console.print("Try running: pip install google-generativeai==0.3.0", style="yellow")
    sys.exit(1)