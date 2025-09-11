"""
DialogueAI with Google Gemini Support

Alternative orchestrator that uses Google's Gemini API instead of OpenAI.
"""

import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from dotenv import load_dotenv

from .document_processor import DocumentProcessor
from .rag_system import RAGSystem
from .gemini_dialogue_generator import GeminiDialogueGenerator, GeminiDialogueConfig

console = Console()


class GeminiDialogueAI:
    """DialogueAI system powered by Google Gemini instead of OpenAI."""
    
    def __init__(
        self,
        tone: str = "conversational",
        level: str = "intermediate",
        model_name: str = "gemini-pro",
        temperature: Optional[float] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_openai_embeddings: bool = False,  # Default to SentenceTransformers
    ):
        # Load environment variables
        load_dotenv(override=False)

        # Config
        temperature = temperature if temperature is not None else float(os.getenv("TEMPERATURE", 0.7))

        # Components
        self.processor = DocumentProcessor()
        self.rag = RAGSystem(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_openai=use_openai_embeddings,  # Use SentenceTransformers by default
        )
        self.generator = GeminiDialogueGenerator(
            GeminiDialogueConfig(
                tone=tone, 
                level=level, 
                max_turns=16, 
                temperature=temperature, 
                model_name=model_name
            )
        )

    def index_document(self, file_path: str) -> None:
        """Extract, clean, chunk, and index a document into the vector store."""
        file_path = Path(file_path)
        console.print(f"📚 Indexing document: {file_path}", style="blue")

        text = self.processor.process_document(file_path)
        text = self.processor.clean_text(text)
        docs = self.rag.process_document(text, source=file_path.name)
        self.rag.create_vector_store(docs)
        self.rag.save_vector_store()

    def create_dialogue(
        self,
        file_path: Optional[str],
        user_goal: str,
        top_k: int = None,
    ) -> str:
        """
        Create a dialogue grounded in the indexed document. If file_path
        is provided, it will be indexed on the fly.
        """
        if file_path:
            self.index_document(file_path)
        else:
            # Try loading existing vector store
            try:
                self.rag.load_vector_store()
            except Exception as e:
                raise RuntimeError("No indexed data found. Provide file_path to index.") from e

        # Build context
        if top_k is not None:
            # temporarily adjust retrieval size via direct call
            docs_scores = self.rag.retrieve_relevant_chunks(user_goal, k=top_k)
            context = "\n\n---\n\n".join([
                f"[Source: {d.metadata.get('source','unknown')}, Chunk {d.metadata.get('chunk_index',0)+1}]\n{d.page_content}"
                for d, _ in docs_scores
            ])
        else:
            context = self.rag.get_context_for_query(user_goal)

        # Generate dialogue using Gemini
        dialogue = self.generator.generate(user_goal=user_goal, context=context)
        return dialogue

    def process_document(self, file_path: str, tone: Optional[str] = None, level: Optional[str] = None) -> str:
        """Convenience method: explain the document like a podcast."""
        if tone:
            self.generator.config.tone = tone
        if level:
            self.generator.config.level = level
        goal = f"Explain the document like a podcast for a {self.generator.config.level} audience in a {self.generator.config.tone} tone."
        return self.create_dialogue(file_path=file_path, user_goal=goal)
