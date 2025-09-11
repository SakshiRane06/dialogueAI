"""
Agent AI Controller Module

Provides intelligent decision-making for dialogue generation,
including tone adaptation, content routing, and persona management.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from rich.console import Console
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

console = Console()


class ToneCategory(str, Enum):
    """Available dialogue tones."""
    CASUAL = "casual"
    ACADEMIC = "academic"  
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    ENTHUSIASTIC = "enthusiastic"
    FORMAL = "formal"


class LevelCategory(str, Enum):
    """Available difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ContentType(str, Enum):
    """Types of content the agent can identify."""
    RESEARCH_PAPER = "research_paper"
    TEXTBOOK = "textbook"
    ARTICLE = "article"
    MANUAL = "manual"
    WHITEPAPER = "whitepaper"
    TUTORIAL = "tutorial"
    OTHER = "other"


@dataclass
class PersonaConfig:
    """Configuration for dialogue personas."""
    learner_style: str = "curious_student"  # curious_student, skeptical_questioner, eager_beginner
    expert_style: str = "patient_teacher"   # patient_teacher, authoritative_expert, friendly_mentor


class ContentAnalysis(BaseModel):
    """Analysis of document content for agent decision-making."""
    content_type: ContentType = Field(description="Type of content detected")
    complexity_level: str = Field(description="Estimated complexity: beginner/intermediate/advanced")
    key_topics: List[str] = Field(description="Main topics covered")
    recommended_tone: ToneCategory = Field(description="Recommended dialogue tone")
    estimated_reading_time: int = Field(description="Estimated reading time in minutes")
    technical_density: str = Field(description="Technical density: low/medium/high")


class AgentController:
    """
    Intelligent agent that analyzes content and makes decisions about 
    dialogue generation parameters.
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model, temperature=0.3)  # Lower temp for more consistent analysis
        self.analysis_parser = PydanticOutputParser(pydantic_object=ContentAnalysis)
    
    def analyze_content(self, text: str, source_filename: str = "") -> ContentAnalysis:
        """
        Analyze document content to inform agent decisions.
        
        Args:
            text: Document text content
            source_filename: Optional filename for context
            
        Returns:
            ContentAnalysis with recommended parameters
        """
        console.print("🤖 Agent analyzing content...", style="blue")
        
        # Create analysis prompt
        system_msg = SystemMessage(content=(
            "You are a content analysis expert. Analyze the provided document text and determine "
            "the best parameters for creating an educational dialogue.\n\n"
            "Consider:\n"
            "- Content complexity and technical level\n"
            "- Target audience appropriateness\n"
            "- Key topics that would benefit from explanation\n"
            "- Optimal tone for engagement\n"
            "- Document type and structure\n\n"
            f"Format instructions:\n{self.analysis_parser.get_format_instructions()}"
        ))
        
        # Truncate text if too long (keep first 2000 chars for analysis)
        analysis_text = text[:2000] + ("..." if len(text) > 2000 else "")
        
        human_msg = HumanMessage(content=(
            f"Document filename: {source_filename}\n\n"
            f"Document content:\n{analysis_text}\n\n"
            "Provide your analysis:"
        ))
        
        prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])
        chain = prompt | self.llm | self.analysis_parser
        
        try:
            analysis = chain.invoke({"text": analysis_text, "filename": source_filename})
            console.print(f"✅ Analysis complete - Type: {analysis.content_type}, Level: {analysis.complexity_level}", style="green")
            return analysis
        except Exception as e:
            console.print(f"⚠️  Analysis failed, using defaults: {e}", style="yellow")
            # Return default analysis if parsing fails
            return ContentAnalysis(
                content_type=ContentType.OTHER,
                complexity_level="intermediate",
                key_topics=["General content"],
                recommended_tone=ToneCategory.CONVERSATIONAL,
                estimated_reading_time=10,
                technical_density="medium"
            )
    
    def select_personas(self, analysis: ContentAnalysis, user_level: str = "intermediate") -> PersonaConfig:
        """
        Select appropriate personas based on content analysis and user level.
        
        Args:
            analysis: Content analysis results
            user_level: User's knowledge level
            
        Returns:
            PersonaConfig with selected personas
        """
        console.print("🎭 Selecting dialogue personas...", style="blue")
        
        # Persona selection logic
        learner_style = "curious_student"  # Default
        expert_style = "patient_teacher"   # Default
        
        # Adjust based on content type
        if analysis.content_type == ContentType.RESEARCH_PAPER:
            if user_level == "beginner":
                learner_style = "eager_beginner"
                expert_style = "patient_teacher"
            else:
                learner_style = "skeptical_questioner"
                expert_style = "authoritative_expert"
        
        elif analysis.content_type == ContentType.TECHNICAL:
            learner_style = "curious_student"
            expert_style = "technical_mentor"
            
        elif analysis.content_type == ContentType.TUTORIAL:
            learner_style = "hands_on_learner"
            expert_style = "step_by_step_guide"
        
        # Adjust based on complexity
        if analysis.complexity_level == "advanced" and user_level == "beginner":
            expert_style = "patient_teacher"
            learner_style = "eager_beginner"
        
        config = PersonaConfig(learner_style=learner_style, expert_style=expert_style)
        console.print(f"✅ Personas: Learner={config.learner_style}, Expert={config.expert_style}", style="green")
        return config
    
    def adapt_tone(self, base_tone: str, analysis: ContentAnalysis, user_preferences: Optional[Dict] = None) -> str:
        """
        Adapt dialogue tone based on content analysis and user preferences.
        
        Args:
            base_tone: User's requested tone
            analysis: Content analysis
            user_preferences: Optional user preferences
            
        Returns:
            Adapted tone recommendation
        """
        console.print("🎨 Adapting dialogue tone...", style="blue")
        
        # Start with user's preference
        adapted_tone = base_tone
        
        # Override based on content analysis if mismatch
        if analysis.content_type == ContentType.RESEARCH_PAPER and base_tone == "casual":
            adapted_tone = "academic"
            console.print("📚 Switching to academic tone for research paper", style="yellow")
        
        elif analysis.technical_density == "high" and base_tone == "casual":
            adapted_tone = "technical"
            console.print("🔧 Switching to technical tone for complex content", style="yellow")
        
        elif analysis.content_type == ContentType.TUTORIAL and base_tone == "formal":
            adapted_tone = "conversational"
            console.print("💬 Switching to conversational tone for tutorial", style="yellow")
        
        console.print(f"✅ Final tone: {adapted_tone}", style="green")
        return adapted_tone
    
    def generate_dialogue_strategy(
        self, 
        analysis: ContentAnalysis, 
        user_goal: str,
        personas: PersonaConfig
    ) -> Dict[str, str]:
        """
        Generate a strategic approach for the dialogue based on analysis.
        
        Args:
            analysis: Content analysis
            user_goal: User's stated goal
            personas: Selected personas
            
        Returns:
            Strategy dictionary with approach details
        """
        console.print("🎯 Generating dialogue strategy...", style="blue")
        
        strategy = {
            "approach": "progressive_disclosure",  # Default
            "starting_point": "overview",
            "depth_level": analysis.complexity_level,
            "focus_areas": ", ".join(analysis.key_topics[:3]),  # Top 3 topics
            "learner_persona": personas.learner_style,
            "expert_persona": personas.expert_style,
            "estimated_turns": "12-16"
        }
        
        # Adjust strategy based on content type
        if analysis.content_type == ContentType.RESEARCH_PAPER:
            strategy["approach"] = "structured_analysis"
            strategy["starting_point"] = "research_question"
        
        elif analysis.content_type == ContentType.TUTORIAL:
            strategy["approach"] = "step_by_step"
            strategy["starting_point"] = "prerequisites"
        
        elif analysis.content_type == ContentType.MANUAL:
            strategy["approach"] = "problem_solution"
            strategy["starting_point"] = "use_cases"
        
        # Adjust based on technical density
        if analysis.technical_density == "high":
            strategy["approach"] = "concept_building"
            strategy["starting_point"] = "fundamentals"
        
        console.print(f"✅ Strategy: {strategy['approach']} starting with {strategy['starting_point']}", style="green")
        return strategy
    
    def recommend_follow_ups(self, dialogue: str, analysis: ContentAnalysis) -> List[str]:
        """
        Suggest follow-up questions or topics based on the generated dialogue.
        
        Args:
            dialogue: Generated dialogue text
            analysis: Content analysis
            
        Returns:
            List of recommended follow-up topics
        """
        console.print("🔍 Generating follow-up recommendations...", style="blue")
        
        follow_ups = []
        
        # Based on content type
        if analysis.content_type == ContentType.RESEARCH_PAPER:
            follow_ups.extend([
                "What are the practical applications of this research?",
                "What questions does this research leave unanswered?",
                "How does this compare to other approaches?"
            ])
        
        elif analysis.content_type == ContentType.TUTORIAL:
            follow_ups.extend([
                "What are common mistakes to avoid?",
                "How can I practice these concepts?",
                "What's the next level after mastering this?"
            ])
        
        # Based on key topics
        for topic in analysis.key_topics[:2]:
            follow_ups.append(f"Can you dive deeper into {topic}?")
        
        console.print(f"✅ Generated {len(follow_ups)} follow-up suggestions", style="green")
        return follow_ups[:5]  # Limit to top 5


# Example usage and testing
if __name__ == "__main__":
    agent = AgentController()
    
    # Test content analysis
    sample_text = """
    Machine Learning (ML) is a subset of artificial intelligence that enables computers to learn 
    and improve from experience without being explicitly programmed. This paper presents a 
    comprehensive survey of supervised learning algorithms, focusing on their mathematical 
    foundations and practical applications in real-world scenarios.
    """
    
    try:
        analysis = agent.analyze_content(sample_text, "ml_survey.pdf")
        print(f"Analysis: {analysis}")
        
        personas = agent.select_personas(analysis, "intermediate")
        print(f"Personas: {personas}")
        
        adapted_tone = agent.adapt_tone("casual", analysis)
        print(f"Adapted tone: {adapted_tone}")
        
        strategy = agent.generate_dialogue_strategy(analysis, "Explain machine learning", personas)
        print(f"Strategy: {strategy}")
        
    except Exception as e:
        print(f"Error testing agent: {e}")
        print("Make sure OpenAI API key is set and dependencies are installed.")
