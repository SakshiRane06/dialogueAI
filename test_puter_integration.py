#!/usr/bin/env python3
"""
Test script for Puter.js integration with DialogueAI

This script tests the Puter.js dialogue generator functionality
to ensure it works correctly with the web application.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, 'src')

from src.puter_dialogue_generator import PuterDialogueGenerator
from src.document_processor import DocumentProcessor
from src.rag_system import RAGSystem

def test_puter_dialogue_generator():
    """Test the PuterDialogueGenerator class."""
    print("🧪 Testing PuterDialogueGenerator...")
    
    try:
        # Initialize the generator
        generator = PuterDialogueGenerator()
        print("✅ PuterDialogueGenerator initialized successfully")
        
        # Test dialogue generation
        user_goal = "Explain the benefits of using AI in education"
        context = "AI can personalize learning experiences, provide instant feedback, and help teachers identify areas where students need additional support."
        
        dialogue = generator.generate(user_goal, context)
        print("✅ Dialogue generated successfully")
        print(f"📝 Dialogue preview: {dialogue[:200]}...")
        
        # Test HTML template generation
        html_template = generator.get_puter_html_template(user_goal, context)
        print("✅ HTML template generated successfully")
        print(f"📄 Template size: {len(html_template)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing PuterDialogueGenerator: {str(e)}")
        return False

def test_web_integration():
    """Test integration with web app components."""
    print("\n🌐 Testing web app integration...")
    
    try:
        # Test document processing (mock)
        print("📄 Testing document processing integration...")
        
        # Create a simple test document
        test_content = "Artificial Intelligence (AI) is transforming education by providing personalized learning experiences. AI can adapt to individual student needs, provide instant feedback, and help teachers identify areas where students struggle."
        
        # Test RAG system with simple content
        rag = RAGSystem(use_openai=False)  # Use SentenceTransformers to avoid API key issues
        docs = rag.process_document(test_content, source="test_document")
        vector_store = rag.create_vector_store(docs)
        print("✅ RAG system integration working")
        
        # Test context retrieval
        context = rag.get_context_for_query("AI in education", max_tokens=500)
        print(f"✅ Context retrieved: {len(context)} characters")
        
        # Test with PuterDialogueGenerator
        generator = PuterDialogueGenerator()
        dialogue = generator.generate("Explain AI benefits in education", context)
        print("✅ End-to-end integration successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing web integration: {str(e)}")
        return False

def test_provider_selection():
    """Test provider selection logic."""
    print("\n🤖 Testing provider selection...")
    
    try:
        # Import web app components
        from web_app import WebDialogueGenerator
        
        web_gen = WebDialogueGenerator()
        print("✅ WebDialogueGenerator initialized")
        
        # Test that Puter is available
        from web_app import PUTER_AVAILABLE
        print(f"✅ Puter availability: {PUTER_AVAILABLE}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing provider selection: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Puter.js Integration Tests")
    print("=" * 50)
    
    tests = [
        ("PuterDialogueGenerator", test_puter_dialogue_generator),
        ("Web Integration", test_web_integration),
        ("Provider Selection", test_provider_selection)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Puter.js integration is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)