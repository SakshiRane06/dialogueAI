"""
Simple CLI for DialogueAI

Usage examples:
  python -m src.cli --index data/sample.pdf
  python -m src.cli --ask "Explain RAG like a podcast" --file data/sample.pdf --tone fun --level beginner
"""
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from rich.console import Console

from .dialogue_ai import DialogueAI
from .puter_dialogue_generator import PuterDialogueGenerator, PuterDialogueConfig

console = Console()

def create_puter_dialogue(rag, user_goal, tone, level):
    """Generate dialogue using Puter.js instead of OpenAI"""
    # Get context from RAG system
    context = rag.get_context_for_query(user_goal, max_tokens=2500)
    
    # Create Puter.js dialogue generator
    config = PuterDialogueConfig(
        tone=tone,
        level=level,
        max_turns=12,
        temperature=0.7,
        model_name="gpt-4o-mini"
    )
    
    generator = PuterDialogueGenerator(config)
    dialogue = generator.generate(user_goal, context)
    return dialogue


def main():
    load_dotenv(override=False)

    parser = argparse.ArgumentParser(description="DialogueAI CLI")
    parser.add_argument("--file", type=str, help="Path to document to index/use", default=None)
    parser.add_argument("--index", action="store_true", help="Only index the provided document and exit")
    parser.add_argument("--ask", type=str, help="User goal or query to generate dialogue", default=None)
    parser.add_argument("--tone", type=str, help="Dialogue tone", default=os.getenv("DEFAULT_TONE", "conversational"))
    parser.add_argument("--level", type=str, help="Audience level", default=os.getenv("DEFAULT_LEVEL", "intermediate"))
    parser.add_argument("--top_k", type=int, help="Number of chunks to retrieve", default=int(os.getenv("TOP_K_RESULTS", "5")))

    args = parser.parse_args()

    # Check environment for embedding preference
    use_openai = os.getenv("USE_OPENAI", "true").lower() == "true"
    
    ai = DialogueAI(tone=args.tone, level=args.level, use_openai_embeddings=use_openai)

    if args.index:
        if not args.file:
            parser.error("--index requires --file <path>")
        ai.index_document(args.file)
        console.print("✅ Indexing complete", style="green")
        return

    if args.ask:
        # Index document first
        if args.file:
            ai.index_document(args.file)
        else:
            # Try loading existing vector store
            try:
                ai.rag.load_vector_store()
            except Exception as e:
                parser.error("No indexed data found. Provide --file <path> to index a document first.")
        
        # Generate dialogue using Puter.js
        dialogue = create_puter_dialogue(ai.rag, args.ask, args.tone, args.level)
        print(dialogue)
        return

    # Default behavior: if file provided, process document; else show help
    if args.file:
        # Index document
        ai.index_document(args.file)
        # Generate default dialogue
        default_goal = f"Explain the document like a podcast for a {args.level} audience in a {args.tone} tone."
        dialogue = create_puter_dialogue(ai.rag, default_goal, args.tone, args.level)
        print(dialogue)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

