"""
Multi-Agent Dialogue Composition

Implements a lightweight, provider-agnostic pipeline to produce more
natural, NotebookLM-style dialogues:
- NotesAgent: distills context into notes and citations
- PlannerAgent: outlines a conversational arc with turn intents
- WriterAgent: composes dialogue from the plan and notes
- EditorAgent: polishes phrasing for natural flow and variation

The writer can use provider-backed LLMs (Google Gemini) if available,
otherwise falls back to a deterministic composition that favors
short, varied utterances and selective citation.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import os


@dataclass
class MultiAgentConfig:
    tone: str = "conversational"
    level: str = "intermediate"
    max_turns: int = 12  # pairs of Learner/Expert utterances
    style_mode: str = "notebook"  # reserved for future styles


class NotesAgent:
    def analyze(self, context: str) -> Dict[str, any]:
        import re
        sources = re.findall(r"\[Source: ([^,]+), Chunk (\d+)\]", context)
        sentences = [s.strip() for s in context.split('. ') if s.strip()]

        key_facts = []
        for s in sentences:
            l = s.lower()
            if any(k in l for k in [" is ", " are ", " means ", " refers to", " consists of", " includes"]):
                key_facts.append(s)

        examples = [s for s in sentences if any(k in s.lower() for k in ["for example", "such as", "like ", " including "])][:3]
        benefits = [s for s in sentences if any(k in s.lower() for k in ["benefit", "advantage", "helps", "improve", "enables"])][:3]
        challenges = [s for s in sentences if any(k in s.lower() for k in ["challenge", "limitation", "problem", "risk", "issue"])][:3]

        return {
            "sources": sources,
            "key_facts": key_facts[:5],
            "examples": examples,
            "benefits": benefits,
            "challenges": challenges,
            "preview": (context[:300] + "...") if len(context) > 300 else context,
        }


class PlannerAgent:
    def plan(self, user_goal: str, notes: Dict[str, any], cfg: MultiAgentConfig) -> List[Dict[str, str]]:
        """Create a turn-by-turn conversational plan."""
        turns = []
        # Opening: broad question aligned to goal
        turns.append({"learner": f"I want to understand {user_goal}. Where should we start?",
                      "expert": "Begin with a quick overview, then set expectations."})

        # Overview and key concepts
        turns.append({"learner": "Give me a quick overview first.",
                      "expert": "Provide a concise definition and a core framing."})
        turns.append({"learner": "Can you unpack the main concepts?",
                      "expert": "Explain 2–3 key ideas naturally, not as bullet points."})

        # Examples and benefits
        if notes.get("examples"):
            turns.append({"learner": "A concrete example would help.",
                          "expert": "Walk through one practical example."})
        if notes.get("benefits"):
            turns.append({"learner": "Why does this matter?",
                          "expert": "Highlight benefits briefly."})

        # Challenges
        if notes.get("challenges"):
            turns.append({"learner": "Any caveats or challenges?",
                          "expert": "Mention a key limitation and how to work with it."})

        # Wrap-up
        turns.append({"learner": "What should I explore next?",
                      "expert": "Suggest next steps and resources."})

        # Respect max_turns (pairs). We created ~5–7 pairs above.
        return turns[:cfg.max_turns]


class WriterAgent:
    def __init__(self, cfg: MultiAgentConfig):
        self.cfg = cfg

    def _compose_line(self, intent: str, notes: Dict[str, any]) -> str:
        # Deterministic, natural phrasing with mild variation
        intent_l = intent.lower()
        if "overview" in intent_l or "definition" in intent_l:
            base = "Let’s start simple. "
            if notes.get("key_facts"):
                fact = notes["key_facts"][0]
                cite = f" [Source: {notes['sources'][0][0]}, Chunk {notes['sources'][0][1]}]" if notes.get("sources") else ""
                return base + fact + cite
            return base + "Here’s the gist in plain terms."
        if "key ideas" in intent_l or "concepts" in intent_l:
            facts = notes.get("key_facts", [])
            if facts:
                return "Two things to keep in mind: " + facts[0]
            return "There are a few moving parts; I’ll keep it tight."
        if "example" in intent_l:
            exs = notes.get("examples", [])
            if exs:
                return exs[0]
            return "Think of a small project where this comes up."
        if "benefits" in intent_l:
            bens = notes.get("benefits", [])
            if bens:
                return bens[0]
            return "It’s useful when clarity and grounding matters."
        if "limitation" in intent_l or "challenge" in intent_l:
            ch = notes.get("challenges", [])
            if ch:
                return ch[0]
            return "You’ll want to watch out for edge cases."
        if "next steps" in intent_l:
            return "Skim the source, try a tiny build, and iterate."
        return intent

    def write(self, plan: List[Dict[str, str]], notes: Dict[str, any]) -> str:
        # Compose natural, short utterances
        parts: List[str] = []
        for step in plan:
            l = step.get("learner", "")
            e_intent = step.get("expert", "")
            parts.append(f"👦 Learner: {l}")
            e_line = self._compose_line(e_intent, notes)
            parts.append(f"👨 Expert: {e_line}")
        return "\n\n".join(parts)


class EditorAgent:
    def polish(self, dialogue: str, cfg: MultiAgentConfig) -> str:
        # Light edits: contractions, varied sentence lengths, soften stiffness
        replacements = {
            "Let us": "Let’s",
            "Do not": "Don’t",
            "It is": "It’s",
            "We are": "We’re",
            "You will": "You’ll",
        }
        for a, b in replacements.items():
            dialogue = dialogue.replace(a, b)
        return dialogue


class MultiAgentDialogue:
    def __init__(self, cfg: Optional[MultiAgentConfig] = None):
        self.cfg = cfg or MultiAgentConfig()
        self.notes_agent = NotesAgent()
        self.planner_agent = PlannerAgent()
        self.writer_agent = WriterAgent(self.cfg)
        self.editor_agent = EditorAgent()

    def generate(self, user_goal: str, context: str, provider: str = "auto") -> str:
        notes = self.notes_agent.analyze(context)
        plan = self.planner_agent.plan(user_goal, notes, self.cfg)

        # If Google AI is explicitly selected and available, use it for composition
        use_gemini = (provider == "google") and bool(os.getenv("GOOGLE_API_KEY"))
        if use_gemini:
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                model = genai.GenerativeModel('gemini-pro')
                prompt = (
                    f"You are DialogueAI. Compose a natural, engaging two-person dialogue (Learner/Expert) in a NotebookLM-like style.\n"
                    f"Tone: {self.cfg.tone}; Level: {self.cfg.level}; Keep utterances short (1–3 sentences), use contractions, vary rhythm.\n"
                    f"Follow this plan strictly, alternating speakers starting with Learner. Cite sources like [Source: <source>, Chunk <n>] only when a specific fact is drawn from notes.\n\n"
                    f"PLAN:\n{plan}\n\nNOTES:\n{notes}\n\nProduce only the dialogue in plain text with 👦 Learner / 👨 Expert lines."
                )
                response = model.generate_content(prompt)
                composed = response.text if hasattr(response, 'text') else ''
                if composed:
                    return self.editor_agent.polish(composed, self.cfg)
            except Exception:
                pass  # Fall back to deterministic writer

        # Deterministic composition fallback
        raw = self.writer_agent.write(plan, notes)
        return self.editor_agent.polish(raw, self.cfg)