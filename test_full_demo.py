"""
Complete DialogueAI Demo with Google API

This script demonstrates the full pipeline:
1. Process document
2. Create RAG embeddings 
3. Generate dialogue using your Google API key
"""

import sys
import os
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, 'src')

from src.document_processor import DocumentProcessor
from src.rag_system import RAGSystem
from rich.console import Console
import google.generativeai as genai

console = Console()


def create_simple_dialogue_with_google(user_goal: str, context: str, api_key: str) -> str:
    """
    Simple function to generate dialogue using Google AI directly.
    """
    try:
        # Configure Google AI
        genai.configure(api_key=api_key)
        
        # Create the prompt
        prompt = f"""
You are DialogueAI, tasked with crafting a clear, engaging two-person dialogue between a curious Learner (👦 Learner) and a knowledgeable Expert (👨 Expert). The dialogue must be grounded in the provided CONTEXT.

Guidelines:
- Tone: conversational
- Difficulty: intermediate
- Max turns: 16 (each turn is a pair: Learner then Expert)
- Prefer short, natural utterances
- Encourage progressive disclosure: start broad, then deepen based on CONTEXT
- Explicitly cite references using [Source: <source>, Chunk <n>] when a point is taken from context
- If the context lacks information, say so briefly and avoid fabricating details
- Output MUST strictly alternate speakers and start with the Learner
- Output format must be plain text like:
👦 Learner: <line>
👨 Expert: <line>
...

USER_GOAL:
{user_goal}

CONTEXT (use selectively and cite):
{context}

Produce the dialogue now.
"""
        
        # Try using the older API structure
        try:
            # For newer versions
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except:
            # For older versions - try direct API call
            import google.ai.generativelanguage as glm
            
            client = glm.GenerativeServiceClient()
            
            request = glm.GenerateTextRequest(
                model='models/text-bison-001',
                prompt=glm.TextPrompt(text=prompt),
                temperature=0.7,
                max_output_tokens=2000
            )
            
            response = client.generate_text(request)
            
            if response.candidates:
                return response.candidates[0].output
            else:
                return "Error: No response generated"
        
    except Exception as e:
        console.print(f"❌ Error with Google AI: {e}", style="red")
        # Return a mock dialogue as fallback
        return f"""
👦 Learner: I'd like to understand the topic from the document you processed.

👨 Expert: Based on the context provided, this appears to be about {user_goal.split()[-1] if user_goal else 'the document content'}.

👦 Learner: Can you break that down for me?

👨 Expert: Certainly! The main concepts are outlined in the retrieved context. However, I encountered a technical issue accessing the AI service to provide the full detailed dialogue.

👦 Learner: What should I know about the key points?

👨 Expert: The document covers important information that would benefit from a more detailed explanation. I recommend checking the RAG system output above to see the relevant chunks that were retrieved.

[Note: This is a fallback dialogue due to API connectivity issues. The RAG retrieval system worked correctly.]
"""


def main():
    console.print("🚀 [bold blue]Complete DialogueAI Demo[/bold blue]", style="blue")
    console.print("Testing full pipeline with Google API\n")
    
    # Your Google API key
    google_api_key = "AIzaSyDshzR9BOiW1wWn2yKRclyrrPpiOFt6QeM"
    
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
        
        # Use SentenceTransformers (no API key needed for embeddings)
        rag = RAGSystem(use_openai=False, chunk_size=800, chunk_overlap=100)
        
        # Process document into chunks
        docs = rag.process_document(cleaned_content, source="sample.txt")
        console.print(f"✅ Created {len(docs)} chunks")
        
        # Create vector store
        vector_store = rag.create_vector_store(docs)
        console.print("✅ Vector store created with embeddings")
        
        # 3. Test Retrieval and Context Building
        console.print(f"\n🎯 [bold]Step 3: Context Building[/bold]")
        
        user_goal = "Explain RAG like a conversational podcast"
        
        # Get context for dialogue generation
        context = rag.get_context_for_query(user_goal)
        console.print(f"✅ Retrieved context ({len(context)} chars)")
        
        # Show context preview
        console.print(f"Context preview: {context[:300]}...\n")
        
        # 4. Generate Dialogue
        console.print("🗣️ [bold]Step 4: Dialogue Generation[/bold]")
        
        dialogue = create_simple_dialogue_with_google(user_goal, context, google_api_key)
        
        # 5. Display Results
        console.print(f"\n🎉 [bold green]Generated Dialogue:[/bold green]")
        console.print("=" * 60)
        console.print(dialogue)
        console.print("=" * 60)
        
        # Save dialogue to file
        output_file = Path("outputs/generated_dialogue.txt")
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"DialogueAI Generated Conversation\n")
            f.write(f"User Goal: {user_goal}\n")
            f.write(f"Generated at: {os.getcwd()}\n\n")
            f.write(dialogue)
        
        console.print(f"\n💾 Dialogue saved to: {output_file}")
        console.print(f"\n✅ [bold green]Demo Complete![/bold green]")
        
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    main()
