# DialogueAI 🎯

An AI-powered system that transforms static documents (PDFs, TXT files) into dynamic two-person dialogues between a curious learner and a knowledgeable expert.

## 🌟 Features

- **Document Ingestion**: Upload PDFs or TXT files
- **RAG Integration**: Retrieval-Augmented Generation for accurate content
- **Dialogue Generation**: Creates natural conversations between learner and expert
- **Agent AI**: Controls tone, style, and flow
- **Multiple Outputs**: Text, audio (TTS), and optional video

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **AI Framework**: LangChain
- **Vector DB**: FAISS
- **LLM**: OpenAI GPT-4
- **Document Processing**: PyMuPDF, pdfplumber
- **Optional TTS**: ElevenLabs, pyttsx3

## 🚀 Quick Start

1. **Clone and setup**:
   ```bash
   git clone <your-repo>
   cd dialoge-ai
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run a simple example**:
   ```python
   from src.dialogue_ai import DialogueAI
   
   ai = DialogueAI()
   result = ai.process_document("data/sample.pdf", tone="casual")
   print(result)
   ```

## 📁 Project Structure

```
dialoge-ai/
├── src/
│   ├── __init__.py
│   ├── document_processor.py    # PDF/TXT parsing
│   ├── rag_system.py           # Vector DB & retrieval
│   ├── dialogue_generator.py   # GenAI dialogue creation
│   ├── agent_controller.py     # Agent AI logic
│   └── dialogue_ai.py          # Main orchestrator
├── data/                       # Input documents
├── outputs/                    # Generated dialogues
├── tests/                      # Unit tests
└── requirements.txt
```

## 🎯 Usage Examples

### Basic Usage
```python
ai = DialogueAI()
dialogue = ai.create_dialogue("path/to/document.pdf")
```

### Advanced Configuration
```python
ai = DialogueAI(
    tone="academic",
    level="beginner", 
    voices={"expert": "professional", "learner": "curious"}
)
dialogue = ai.create_dialogue("research_paper.pdf")
```

## 🔧 Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black src/ tests/
```

## 📄 License

MIT License - see LICENSE file for details.
