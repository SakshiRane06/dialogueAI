# DialogueAI 🎯

An AI-powered system that transforms static documents (PDFs, TXT files) into dynamic two-person dialogues between a curious learner and a knowledgeable expert.

## 🌟 Features

- **Document Ingestion**: Upload PDFs, TXT, or DOCX files
- **RAG Integration**: Retrieval-Augmented Generation for accurate content
- **Dialogue Generation**: Creates natural conversations between learner and expert
- **Multiple AI Providers**: OpenAI, Google Gemini, and Puter.js (free!)
- **Agent Intelligence**: Smart content analysis and dialogue strategies
- **Web Interface**: Beautiful Flask-based web application
- **CLI Interface**: Command-line tools for batch processing

## 🛐️ Tech Stack

- **Language**: Python 3.8+
- **AI Framework**: LangChain + Custom Puter.js Integration
- **Vector DB**: FAISS with SentenceTransformers embeddings
- **AI Providers**: OpenAI GPT-4, Google Gemini, Puter.js (free)
- **Document Processing**: PyMuPDF, pdfplumber, python-docx
- **Web Framework**: Flask with beautiful UI
- **Embeddings**: OpenAI embeddings + SentenceTransformers fallback

## 🚀 Quick Start

### **Method 1: Windows Batch File (Easiest)**
```bash
# Double-click or run:
START_HERE.bat
```

### **Method 2: Web Application**
```bash
# Install dependencies
pip install -r requirements.txt

# Start web interface (recommended)
python debug_web_app.py
# OR standard version
python web_app.py

# Open browser to:
# http://localhost:5001 (debug version)
# http://localhost:5000 (standard version)
```

### **Method 3: Command Line**
```bash
# Generate dialogue directly
python -m src.cli --file data/sample.txt --ask "Explain RAG like a podcast"

# Process with custom settings
python -m src.cli --file data/sample.txt --tone casual --level beginner
```

### **Method 4: Application Launcher**
```bash
python launch_app.py
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

## 🤖 AI Providers

### **Puter.js (Free & Recommended)**
- ✅ **No API keys required**
- ✅ **Intelligent context analysis**
- ✅ **Works out of the box**
- ✅ **Perfect for testing and development**

### **Google Gemini**
- Requires `GOOGLE_API_KEY` in .env file
- High-quality dialogue generation
- Good for production use

### **OpenAI GPT-4**
- Requires `OPENAI_API_KEY` in .env file
- Premium quality responses
- Higher cost per request

## 🔧 Development

### **Testing**
```bash
# Test core functionality (no API keys needed)
python test_demo.py

# Test Puter.js integration
python test_puter_integration.py

# Test full system
python test_full_demo.py
```

### **Code Quality**
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/
```

## 📄 License

MIT License - see LICENSE file for details.
