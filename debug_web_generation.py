#!/usr/bin/env python3
"""
Debug script to test web interface dialogue generation
"""

import sys
sys.path.insert(0, 'src')

from web_app import WebDialogueGenerator
import traceback

def test_web_generation():
    print("🧪 Testing Web Interface Dialogue Generation")
    print("=" * 50)
    
    try:
        # Initialize the web dialogue generator
        web_gen = WebDialogueGenerator()
        print("✅ WebDialogueGenerator initialized")
        
        # Test document processing
        print("\n📄 Testing document processing...")
        content, stats, success = web_gen.process_document("data/sample.txt")
        
        if success:
            print(f"✅ Document processed successfully")
            print(f"   - Characters: {stats.get('characters', 0)}")
            print(f"   - Chunks: {stats.get('chunks', 0)}")
            print(f"   - Embedding Model: {stats.get('embedding_model', 'N/A')}")
            print(f"   - Using OpenAI: {stats.get('using_openai', False)}")
            
            # Test dialogue generation
            print("\n🗣️ Testing dialogue generation...")
            user_goal = "Explain RAG like a podcast"
            tone = "conversational"
            difficulty = "intermediate"
            provider = "auto"
            
            print(f"   - Goal: {user_goal}")
            print(f"   - Tone: {tone}")
            print(f"   - Difficulty: {difficulty}")
            print(f"   - Provider: {provider}")
            
            dialogue, success = web_gen.generate_dialogue(user_goal, tone, difficulty, provider)
            
            if success:
                print("✅ Dialogue generated successfully!")
                print("\n📝 Dialogue Preview:")
                print("-" * 40)
                print(dialogue[:500] + "..." if len(dialogue) > 500 else dialogue)
                print("-" * 40)
                
                return True
            else:
                print(f"❌ Dialogue generation failed: {dialogue}")
                return False
                
        else:
            print(f"❌ Document processing failed: {content}")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("\n🔍 Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_web_generation()
    print(f"\n🎯 Test Result: {'✅ PASSED' if success else '❌ FAILED'}")
