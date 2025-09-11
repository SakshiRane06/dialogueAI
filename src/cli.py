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

console = Console()


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

    ai = DialogueAI(tone=args.tone, level=args.level)

    if args.index:
        if not args.file:
            parser.error("--index requires --file <path>")
        ai.index_document(args.file)
        console.print("✅ Indexing complete", style="green")
        return

    if args.ask:
        dialogue = ai.create_dialogue(file_path=args.file, user_goal=args.ask, top_k=args.top_k)
        print(dialogue)
        return

    # Default behavior: if file provided, process document; else show help
    if args.file:
        dialogue = ai.process_document(args.file, tone=args.tone, level=args.level)
        print(dialogue)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

