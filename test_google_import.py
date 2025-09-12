#!/usr/bin/env python3
"""
Test Google AI Import
"""

import sys
import os
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, 'src')

# Load environment variables
load_dotenv()

# Set Google API key directly
google_api_key = os.getenv('GOOGLE_API_KEY')
print(f"Google API Key: {'Set' if google_api_key else 'Not Set'}")

# Try to import Google AI
try:
    import google.generativeai as genai
    print('Google Generative AI module imported successfully')
    
    # Configure the API
    genai.configure(api_key=google_api_key)
    print('Google Generative AI configured successfully')
    
    # Try to list models
    try:
        models = genai.list_models()
        print(f"Found {len(list(models))} models")
        for model in models:
            print(f"  - {model.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
    
except Exception as e:
    print(f'Error importing Google Generative AI: {e}')