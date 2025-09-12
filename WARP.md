# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

DialogueAI is an AI-powered system that transforms static documents (PDFs, TXT files) into dynamic two-person dialogues between a curious learner and knowledgeable expert. It uses Retrieval-Augmented Generation (RAG) with multiple AI provider support including OpenAI, Google Gemini, and Puter.js (free, no API key required).

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (optional - system works without them)
```

### Running the Application
```bash
# Start web interface (preferred method)
python web_app.py
# OR use the launcher
python launch_app.py
# OR use the batch files (Windows)
START_HERE.bat
start_web.bat
```

### Testing
```bash
# Run demo test (no API keys required)
python test_demo.py

# Run full integration test
python test_full_demo.py

# Run Puter.js integration test
python test_puter_integration.py

# Test specific components
python -m src.document_processor
python -m src.rag_system
python -m src.agent_controller
```

### Command Line Interface
```bash
# Index a document only
python -m src.cli --index --file data/sample.pdf

# Generate dialogue from document
python -m src.cli --file data/sample.pdf --ask "Explain this like a podcast"

# Customize tone and level
python -m src.cli --file data/sample.pdf --ask "Explain RAG" --tone academic --level beginner

# Adjust retrieval chunks
python -m src.cli --file data/sample.pdf --ask "Key concepts" --top_k 3
```

### Code Formatting
```bash
# Format code
black src/ tests/

# Lint code (if configured)
flake8 src/
```

## Architecture Overview

### Core Components

**DialogueAI Orchestrator** (`src/dialogue_ai.py`)
- Main entry point that coordinates all components
- Handles document indexing and dialogue generation pipeline
- Manages configuration from environment variables

**Document Processing Pipeline**
- `DocumentProcessor` (`src/document_processor.py`): Extracts text from PDFs, TXT, DOCX files using PyMuPDF and pdfplumber
- `RAGSystem` (`src/rag_system.py`): Chunks documents, creates embeddings, manages FAISS vector store
- Supports both OpenAI embeddings and SentenceTransformers fallback

**Dialogue Generation Engines**
- `DialogueGenerator` (`src/dialogue_generator.py`): OpenAI-based dialogue generation
- `GeminiDialogueGenerator` (`src/gemini_dialogue_generator.py`): Google AI integration
- `PuterDialogueGenerator` (`src/puter_dialogue_generator.py`): Free AI via Puter.js platform

**Agent Intelligence** (`src/agent_controller.py`)
- Analyzes document content type and complexity
- Selects appropriate personas (learner/expert styles)
- Adapts tone based on content analysis
- Generates dialogue strategies and follow-up recommendations

### AI Provider Architecture

The system supports multiple AI providers with fallback logic:

1. **Auto Mode**: Automatically selects best available provider
2. **Puter.js**: Free AI access, no API keys required (always available)
3. **Google Gemini**: Uses GOOGLE_API_KEY from environment
4. **OpenAI**: Uses OPENAI_API_KEY (currently disabled in web interface to avoid quota issues)

### Web Interface (`web_app.py`)

Flask-based web application with:
- File upload and processing
- AI provider selection
- Dialogue customization (tone, difficulty)
- Result viewing and downloading
- Status API endpoint at `/api/status`

### Data Flow

1. **Document Upload** → DocumentProcessor extracts text
2. **Text Processing** → RAGSystem chunks and embeds content
3. **Context Retrieval** → FAISS vector search finds relevant chunks
4. **Agent Analysis** → AgentController analyzes content and selects strategy
5. **Dialogue Generation** → Selected AI provider creates conversation
6. **Output** → Formatted dialogue with source citations

### Configuration System

Environment variables (`.env`):
- `OPENAI_API_KEY`: OpenAI access (optional)
- `GOOGLE_API_KEY`: Google AI access (optional)
- `ELEVENLABS_API_KEY`: Text-to-speech (optional)
- Model and RAG parameters with sensible defaults

### Error Handling & Fallbacks

- **Embedding Fallback**: OpenAI → SentenceTransformers
- **AI Provider Fallback**: OpenAI → Google → Puter.js → Mock dialogue
- **Document Processing**: PyMuPDF → pdfplumber fallback
- **Encoding**: UTF-8 → latin-1 → cp1252 → iso-8859-1

## Development Patterns

### Module Structure
- Each core component is self-contained with clear interfaces
- Rich console logging throughout for debugging
- Dataclasses for configuration objects
- Type hints and docstrings for all public methods

### Testing Strategy
- `test_demo.py`: Core RAG functionality without API dependencies
- `test_full_demo.py`: End-to-end testing with API integration
- Component-specific tests in individual modules
- Graceful degradation when APIs unavailable

### File Organization
```
src/
├── __init__.py
├── dialogue_ai.py          # Main orchestrator
├── document_processor.py   # PDF/TXT/DOCX processing
├── rag_system.py          # Vector DB & retrieval
├── dialogue_generator.py   # OpenAI dialogue generation
├── gemini_dialogue_*.py    # Google AI integration
├── puter_dialogue_*.py     # Puter.js integration
├── agent_controller.py     # Intelligent agent decisions
└── cli.py                 # Command line interface

web_app.py                 # Flask web application
templates/                 # HTML templates
data/                      # Input documents
outputs/                   # Generated dialogues
uploads/                   # Temporary uploads
```

### Key Design Principles

- **Provider Agnostic**: Abstract AI providers behind common interfaces
- **Graceful Degradation**: System works even without API keys
- **Rich Feedback**: Comprehensive logging and error messages
- **Modular Architecture**: Components can be used independently
- **Configuration Driven**: Environment variables control behavior

### Puter.js Integration

Special integration with Puter.js platform provides free AI access:
- No API keys required
- Client-side execution capability
- Multiple model access
- See `PUTER_INTEGRATION.md` for detailed documentation

## Common Development Tasks

### Adding New AI Providers
1. Create new generator class inheriting from base pattern
2. Add provider detection in `web_app.py`
3. Update provider selection logic
4. Add configuration options

### Extending Document Support
1. Add file extension to `ALLOWED_EXTENSIONS`
2. Implement processing method in `DocumentProcessor`
3. Add format detection logic
4. Test with sample documents

### Modifying Dialogue Format
1. Update prompt templates in dialogue generators
2. Adjust output parsing if needed
3. Test across all AI providers
4. Update documentation examples

### Performance Optimization
- Adjust `chunk_size` and `chunk_overlap` in RAG settings
- Tune embedding model selection
- Optimize vector store operations
- Cache processed documents when appropriate

## Environment Notes

- **Windows Batch Files**: `START_HERE.bat` and `start_web.bat` for easy launching
- **Virtual Environment**: Batch files automatically create and activate venv
- **Dependencies**: Comprehensive requirements.txt with optional features
- **Cross-Platform**: Code works on Windows, macOS, and Linux
