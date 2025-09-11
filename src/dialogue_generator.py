"""
Dialogue Generator Module

Converts retrieved context into a two-person dialogue (Learner and Expert)
using an LLM. Supports tone and level customization.
"""
from dataclasses import dataclass
from typing import Dict, Optional

from rich.console import Console
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

console = Console()


@dataclass
class DialogueConfig:
    tone: str = "conversational"  # e.g., casual, academic, fun, formal
    level: str = "intermediate"   # e.g., beginner, intermediate, advanced
    max_turns: int = 16           # max back-and-forth turns (Learner/Expert)
    temperature: float = 0.7
    chat_model: str = "gpt-4"


class DialogueGenerator:
    """Generates dialogues from context and a user goal/instruction."""

    def __init__(self, config: Optional[DialogueConfig] = None):
        self.config = config or DialogueConfig()
        self.llm = ChatOpenAI(model=self.config.chat_model, temperature=self.config.temperature)
        self.output_parser = StrOutputParser()

    def _build_prompt(self, user_goal: str, context: str) -> ChatPromptTemplate:
        """Create a structured prompt for high-quality, grounded dialogue."""
        system = SystemMessage(
            content=(
                "You are DialogueAI, tasked with crafting a clear, engaging two-person dialogue between "
                "a curious Learner (👦 Learner) and a knowledgeable Expert (👨 Expert). The dialogue must be grounded in the provided CONTEXT.\n\n"
                "Guidelines:\n"
                f"- Tone: {self.config.tone}.\n"
                f"- Difficulty: {self.config.level}.\n"
                f"- Max turns: {self.config.max_turns} (each turn is a pair: Learner then Expert).\n"
                "- Prefer short, natural utterances.\n"
                "- Encourage progressive disclosure: start broad, then deepen based on CONTEXT.\n"
                "- Explicitly cite references using [Source: <source>, Chunk <n>] when a point is taken from context.\n"
                "- If the context lacks information, say so briefly and avoid fabricating details.\n"
                "- Output MUST strictly alternate speakers and start with the Learner.\n"
                "- Output format must be plain text like:\n"
                "👦 Learner: <line>\n👨 Expert: <line>\n...\n"
            )
        )

        human = HumanMessage(
            content=(
                "USER_GOAL:\n{user_goal}\n\n"
                "CONTEXT (use selectively and cite):\n{context}\n\n"
                "Produce the dialogue now."
            )
        )

        return ChatPromptTemplate.from_messages([system, human])

    def generate(self, user_goal: str, context: str) -> str:
        prompt = self._build_prompt(user_goal=user_goal, context=context)
        chain = prompt | self.llm | self.output_parser
        console.print("🗣️ Generating dialogue...", style="blue")
        return chain.invoke({"user_goal": user_goal, "context": context})

