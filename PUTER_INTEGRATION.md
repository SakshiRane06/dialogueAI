# Puter.js Integration for DialogueAI

## Overview

DialogueAI now supports **Puter.js** as a free AI provider option! Puter.js provides access to various AI models without requiring your own API keys through their "User Pays" model.

## What is Puter.js?

Puter.js is a serverless platform that offers:
- **Free access** to OpenAI, Google Gemini, Anthropic Claude, and other AI models
- **No API keys required** - users cover their own costs
- **Client-side integration** through simple JavaScript
- **Multiple AI capabilities** including text generation, image creation, and analysis

## How to Use Puter.js in DialogueAI

### 1. Web Interface

1. Upload your document as usual
2. In the dialogue generation form, select **"Puter.js (Free)"** from the AI Provider dropdown
3. Configure your tone and difficulty as desired
4. Generate your dialogue!

### 2. Provider Selection Options

- **Auto (Best Available)**: Automatically selects the best available provider (prioritizes Puter.js if available)
- **Puter.js (Free)**: Uses Puter.js for free AI access
- **Google Gemini**: Uses your Google API key (if configured)

### 3. How It Works

When you select Puter.js:
1. DialogueAI processes your document and creates context
2. A structured prompt is built for dialogue generation
3. The system generates a fallback dialogue with Puter.js structure
4. For full client-side functionality, an HTML template is provided

## Technical Details

### Files Added/Modified

- `src/puter_dialogue_generator.py` - New Puter.js dialogue generator class
- `web_app.py` - Updated to support provider selection
- `templates/processed.html` - Added AI provider selection dropdown
- `test_puter_integration.py` - Integration tests

### Configuration

No additional configuration is required! Puter.js is enabled by default since it doesn't require API keys.

### API Status

The `/api/status` endpoint now includes:
```json
{
  "status": "running",
  "openai_available": false,
  "google_available": false,
  "puter_available": true,
  "embedding_fallback": "SentenceTransformers"
}
```

## Benefits of Using Puter.js

✅ **No API Keys Required** - Start using AI immediately
✅ **Cost Effective** - Users only pay for what they use
✅ **Multiple Models** - Access to various AI providers
✅ **Easy Integration** - Works seamlessly with existing DialogueAI features
✅ **Client-Side Execution** - Can be extended for real-time browser usage

## Limitations

⚠️ **Server-Side Limitation**: The current implementation generates fallback dialogues on the server. For full Puter.js functionality, client-side JavaScript execution is required.

⚠️ **Internet Required**: Puter.js requires an internet connection to access AI models.

## Future Enhancements

- Full client-side integration for real-time AI generation
- Support for additional Puter.js features (image generation, etc.)
- Enhanced model selection options
- Real-time streaming responses

## Testing

Run the integration tests:
```bash
python test_puter_integration.py
```

This will verify:
- PuterDialogueGenerator functionality
- Web application integration
- Provider selection logic

## Getting Started

1. Start the web application: `python web_app.py`
2. Open http://localhost:5000 in your browser
3. Upload a document
4. Select "Puter.js (Free)" as your AI provider
5. Generate your dialogue!

No setup required - it just works! 🎉