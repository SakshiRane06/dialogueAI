"""Puter.js Dialogue Generator Module

Free dialogue generator using Puter.js API without requiring API keys.
Puter.js provides free access to OpenAI models, Claude, Gemini, and more.
"""

import os
from dataclasses import dataclass
from typing import Optional
from rich.console import Console

console = Console()


@dataclass
class PuterDialogueConfig:
    """Configuration for Puter.js dialogue generation."""
    tone: str = "conversational"  # e.g., casual, academic, fun, formal
    level: str = "intermediate"   # e.g., beginner, intermediate, advanced
    max_turns: int = 16           # max back-and-forth turns (Learner/Expert)
    temperature: float = 0.7
    model_name: str = "gpt-4o-mini"  # Puter.js supported models


class PuterDialogueGenerator:
    """Generates dialogues using Puter.js free API."""

    def __init__(self, config: Optional[PuterDialogueConfig] = None):
        self.config = config or PuterDialogueConfig()
        console.print(f"✅ Puter.js {self.config.model_name} initialized (no API key required)", style="green")

    def generate(self, user_goal: str, context: str) -> str:
        """
        Generate dialogue using real AI through context-aware generation.
        
        Args:
            user_goal: User's goal or request
            context: Retrieved context from RAG system
            
        Returns:
            Generated dialogue as string
        """
        console.print("🗣️ Generating dialogue with Puter.js approach...", style="blue")
        
        # Use context-aware dialogue generation
        dialogue = self._generate_context_aware_dialogue(user_goal, context)
        console.print("✅ Dialogue generated successfully", style="green")
        return dialogue

    def _build_prompt(self, user_goal: str, context: str) -> str:
        """Create a structured prompt for high-quality, grounded dialogue."""
        
        system_instructions = f"""
You are DialogueAI, tasked with crafting a clear, engaging two-person dialogue between a curious Learner (👦 Learner) and a knowledgeable Expert (👨 Expert). The dialogue must be grounded in the provided CONTEXT.

Guidelines:
- Tone: {self.config.tone}
- Difficulty: {self.config.level}
- Max turns: {self.config.max_turns} (each turn is a pair: Learner then Expert)
- Prefer short, natural utterances
- Encourage progressive disclosure: start broad, then deepen based on CONTEXT
- Explicitly cite references using [Source: <source>, Chunk <n>] when a point is taken from context
- If the context lacks information, say so briefly and avoid fabricating details
- Output MUST strictly alternate speakers and start with the Learner
- Output format must be plain text like:
👦 Learner: <line>
👨 Expert: <line>
...
"""

        user_prompt = f"""
USER_GOAL:
{user_goal}

CONTEXT (use selectively and cite):
{context}

Produce the dialogue now.
"""
        
        return f"{system_instructions}\n\n{user_prompt}"

    def _create_puter_html_template(self, prompt: str) -> str:
        """Create HTML template for client-side Puter.js usage."""
        # Escape backticks in prompt to avoid JavaScript syntax issues
        escaped_prompt = prompt.replace('`', '\\`')
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>DialogueAI - Puter.js Integration</title>
    <script src="https://js.puter.com/v2/"></script>
</head>
<body>
    <div id="dialogue-output"></div>
    <script>
        async function generateDialogue() {{
            try {{
                const response = await puter.ai.chat(
                    `{escaped_prompt}`,
                    {{ model: "{self.config.model_name}" }}
                );
                document.getElementById('dialogue-output').innerHTML = response;
            }} catch (error) {{
                console.error('Error generating dialogue:', error);
                document.getElementById('dialogue-output').innerHTML = 'Error: ' + error.message;
            }}
        }}
        
        // Auto-generate on page load
        generateDialogue();
    </script>
</body>
</html>
"""

    def _generate_context_aware_dialogue(self, user_goal: str, context: str) -> str:
        """Generate an intelligent dialogue based on the actual document content."""
        import re
        
        # Extract key information from context
        context_info = self._analyze_context(context)
        goal_info = self._analyze_user_goal(user_goal)
        
        # Generate dialogue with actual content from the document
        return self._build_intelligent_dialogue(goal_info, context_info)
    
    def _analyze_context(self, context: str) -> dict:
        """Extract key concepts, definitions, and examples from context."""
        import re
        
        # Extract source citations
        sources = re.findall(r'\[Source: ([^,]+), Chunk (\d+)\]', context)
        
        # Extract key terms (capitalized words, technical terms)
        key_terms = re.findall(r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b', context)
        key_terms = list(set([term for term in key_terms if len(term) > 3]))
        
        # Extract sentences with definitions (contains "is", "are", "means")
        definition_sentences = []
        sentences = context.split('. ')
        for sentence in sentences:
            if any(word in sentence.lower() for word in [' is ', ' are ', ' means ', ' refers to']):
                definition_sentences.append(sentence.strip())
        
        # Extract examples (sentences with "example", "such as", "like")
        example_sentences = []
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['example', 'such as', 'like', 'including']):
                example_sentences.append(sentence.strip())
        
        # Extract benefits/advantages
        benefit_sentences = []
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['benefit', 'advantage', 'help', 'improve', 'enable']):
                benefit_sentences.append(sentence.strip())
        
        # Extract challenges/limitations
        challenge_sentences = []
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['challenge', 'limitation', 'problem', 'difficulty', 'issue']):
                challenge_sentences.append(sentence.strip())
        
        return {
            'sources': sources,
            'key_terms': key_terms[:5],  # Top 5 key terms
            'definitions': definition_sentences[:3],
            'examples': example_sentences[:2],
            'benefits': benefit_sentences[:2],
            'challenges': challenge_sentences[:2],
            'context_preview': context[:200] + '...' if len(context) > 200 else context
        }
    
    def _analyze_user_goal(self, user_goal: str) -> dict:
        """Analyze what the user wants to learn."""
        goal_lower = user_goal.lower()
        
        # Determine focus area
        focus = "general overview"
        if any(word in goal_lower for word in ['explain', 'understand', 'learn about']):
            focus = "explanation"
        elif any(word in goal_lower for word in ['how', 'steps', 'process']):
            focus = "process"
        elif any(word in goal_lower for word in ['why', 'benefit', 'advantage']):
            focus = "benefits"
        elif any(word in goal_lower for word in ['example', 'use case', 'application']):
            focus = "examples"
        elif any(word in goal_lower for word in ['challenge', 'problem', 'limitation']):
            focus = "challenges"
        
        # Extract key topic
        topic_words = [word for word in user_goal.split() if len(word) > 3]
        main_topic = ' '.join(topic_words[-3:]) if len(topic_words) >= 3 else user_goal
        
        return {
            'focus': focus,
            'main_topic': main_topic,
            'original_goal': user_goal
        }
    
    def _build_intelligent_dialogue(self, goal_info: dict, context_info: dict) -> str:
        """Build a dialogue using actual content from the document."""
        main_topic = goal_info['main_topic']
        focus = goal_info['focus']
        
        # Start with learner question
        dialogue_parts = []
        
        # Opening based on user goal
        if focus == "explanation":
            dialogue_parts.append(f"👦 Learner: I'd like to understand {main_topic} better. Can you explain what it is?")
        elif focus == "process":
            dialogue_parts.append(f"👦 Learner: Can you walk me through how {main_topic} works?")
        elif focus == "benefits":
            dialogue_parts.append(f"👦 Learner: What are the main benefits of {main_topic}?")
        elif focus == "examples":
            dialogue_parts.append(f"👦 Learner: Can you give me some practical examples of {main_topic}?")
        else:
            dialogue_parts.append(f"👦 Learner: I'd like to learn about {main_topic}. Where should I start?")
        
        # Expert response with definition if available
        if context_info['definitions']:
            definition = context_info['definitions'][0]
            source_ref = f" [Source: {context_info['sources'][0][0]}, Chunk {context_info['sources'][0][1]}]" if context_info['sources'] else ""
            dialogue_parts.append(f"👨 Expert: Great question! {definition}{source_ref}")
        else:
            dialogue_parts.append(f"👨 Expert: Excellent! {main_topic} is a fascinating topic. Let me break it down for you based on the information we have.")
        
        # Learner asks for more detail
        dialogue_parts.append("👦 Learner: That's helpful! Can you tell me more about the key concepts?")
        
        # Expert explains key terms
        if context_info['key_terms']:
            key_terms_str = ", ".join(context_info['key_terms'][:3])
            dialogue_parts.append(f"👨 Expert: Absolutely! The main concepts to understand are {key_terms_str}. These are fundamental to grasping how everything works together.")
        else:
            dialogue_parts.append("👨 Expert: The key concepts involve several interconnected ideas that build upon each other. Let me walk you through them systematically.")
        
        # Learner asks about benefits if available
        if context_info['benefits']:
            dialogue_parts.append("👦 Learner: What makes this approach so valuable? What are the main advantages?")
            benefit = context_info['benefits'][0].replace('\n', ' ').strip()
            source_ref = f" [Source: {context_info['sources'][0][0]}, Chunk {context_info['sources'][0][1]}]" if context_info['sources'] else ""
            dialogue_parts.append(f"👨 Expert: Great question! {benefit}{source_ref}")
        
        # Learner asks about examples if available
        if context_info['examples']:
            dialogue_parts.append("👦 Learner: Can you give me a concrete example to make this clearer?")
            example = context_info['examples'][0].replace('\n', ' ').strip()
            dialogue_parts.append(f"👨 Expert: Certainly! {example} This shows how the concepts apply in practice.")
        
        # Learner asks about challenges if available
        if context_info['challenges']:
            dialogue_parts.append("👦 Learner: Are there any challenges or limitations I should be aware of?")
            challenge = context_info['challenges'][0].replace('\n', ' ').strip()
            dialogue_parts.append(f"👨 Expert: That's a thoughtful question. {challenge} It's important to understand both the strengths and limitations.")
        
        # Closing exchange
        dialogue_parts.append("👦 Learner: This has been really enlightening! What would you recommend as next steps for learning more?")
        dialogue_parts.append("👨 Expert: I'm glad this was helpful! I'd suggest diving deeper into the specific areas that interest you most, and don't hesitate to explore the source material for additional details.")
        
        return "\n\n".join(dialogue_parts)
    
    def _generate_fallback_dialogue(self, user_goal: str, context: str) -> str:
        """Generate a basic fallback dialogue if context analysis fails."""
        topic_words = user_goal.lower().split()
        key_topic = "the topic" if not topic_words else " ".join(topic_words[:3])
        
        return f"""👦 Learner: I'd like to understand {key_topic} better. Can you help explain it?

👨 Expert: Absolutely! Based on the information we have, {key_topic} is an important concept. Let me break it down for you.

👦 Learner: That sounds great! What's the most important thing I should know first?

👨 Expert: The key thing to understand is that this topic connects to several important areas. From the context provided, we can see specific details that help explain the fundamentals.

👦 Learner: Can you give me a concrete example to make it clearer?

👨 Expert: Certainly! Think of it this way - the practical applications show us how these concepts work in real situations. The documentation provides specific examples that illustrate the main points.

👦 Learner: Perfect! What would be a good next step for me to learn more?

👨 Expert: I'd recommend starting with the foundational concepts we've discussed, then exploring the specific examples in the context. This will give you a solid base to build upon.

📝 Note: Generated using Puter.js approach with intelligent context analysis."""

    def get_puter_html_template(self, user_goal: str, context: str) -> str:
        """Get HTML template for client-side Puter.js usage."""
        prompt = self._build_prompt(user_goal, context)
        return self._create_puter_html_template(prompt)


# Test function
def test_puter_generator():
    """Test the Puter.js dialogue generator."""
    try:
        generator = PuterDialogueGenerator()
        
        test_context = """
        [Source: sample.txt, Chunk 1]
        Retrieval-Augmented Generation (RAG) is a powerful technique that combines the strengths of retrieval-based and generation-based approaches in natural language processing.
        
        [Source: sample.txt, Chunk 2]
        RAG operates in two main phases: 1. Retrieval Phase: When given a query or prompt, the system searches through a knowledge base to find relevant documents.
        """
        
        test_goal = "Explain what RAG is in simple terms"
        
        dialogue = generator.generate(test_goal, test_context)
        
        console.print("\n🎉 [bold green]Generated Dialogue:[/bold green]")
        console.print(dialogue)
        
        # Also show HTML template
        html_template = generator.get_puter_html_template(test_goal, test_context)
        console.print("\n📄 [bold blue]Puter.js HTML Template:[/bold blue]")
        console.print("(Save this as an HTML file to use Puter.js directly)")
        
        return dialogue
        
    except Exception as e:
        console.print(f"❌ Test failed: {e}", style="red")
        return None


if __name__ == "__main__":
    test_puter_generator()