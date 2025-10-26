#!/usr/bin/env python3
"""
Debug version of web_app.py with enhanced logging
"""

import os
import sys
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import tempfile
import uuid
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, 'src')

from src.document_processor import DocumentProcessor
from src.rag_system import RAGSystem
from src.dialogue_generator import DialogueGenerator
from src.puter_dialogue_generator import PuterDialogueGenerator
from src.multi_agent_dialogue import MultiAgentDialogue, MultiAgentConfig

# Try to import Google AI, but make it optional
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except Exception:
    GOOGLE_AI_AVAILABLE = False
    genai = None

# Puter.js is always available (no API key required)
PUTER_AVAILABLE = True

try:
    from dotenv import load_dotenv
    load_dotenv()
    print(f"ELEVENLABS_API_KEY present at startup: {bool(os.getenv('ELEVENLABS_API_KEY'))}")
except Exception as e:
    print(f"dotenv load failed: {e}")
except Exception:
    pass

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-in-production')

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Create folders if they don't exist
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(RESULTS_FOLDER).mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class DebugWebDialogueGenerator:
    """Debug version with enhanced logging"""
    
    def __init__(self):
        print("🔧 Initializing DebugWebDialogueGenerator")
        self.processor = DocumentProcessor()
        self.rag = None
        
    def process_document(self, file_path: str) -> tuple:
        """Process uploaded document and return content and stats."""
        try:
            print(f"📄 Processing document: {file_path}")
            
            # Process document
            content = self.processor.process_document(Path(file_path))
            cleaned_content = self.processor.clean_text(content)
            print(f"✅ Document content extracted: {len(cleaned_content)} characters")
            
            # Initialize RAG system (use SentenceTransformers to avoid quota issues)
            print("🔧 Initializing RAG system...")
            self.rag = RAGSystem(
                use_openai=False,  # Force SentenceTransformers to avoid OpenAI quota issues
                chunk_size=800,
                chunk_overlap=100
            )
            
            # Create document chunks and vector store
            docs = self.rag.process_document(cleaned_content, source=os.path.basename(file_path))
            vector_store = self.rag.create_vector_store(docs)
            
            stats = {
                'characters': len(cleaned_content),
                'chunks': len(docs),
                'embedding_model': self.rag.embedding_model,
                'using_openai': self.rag.use_openai
            }
            
            print(f"✅ Document processed successfully - Stats: {stats}")
            return cleaned_content, stats, True
            
        except Exception as e:
            print(f"❌ Error processing document: {str(e)}")
            return str(e), {}, False
    
    def generate_dialogue(self, user_goal: str, tone: str = "conversational", difficulty: str = "intermediate", provider: str = "auto", quality: str = "standard") -> tuple:
        """Generate dialogue using the processed document."""
        try:
            print(f"🗣️ Starting dialogue generation:")
            print(f"   - Goal: {user_goal}")
            print(f"   - Tone: {tone}")
            print(f"   - Difficulty: {difficulty}")
            print(f"   - Provider: {provider}")
            print(f"   - Quality: {quality}")
            
            if not self.rag or not self.rag.vector_store:
                error_msg = "Error: No document processed"
                print(f"❌ {error_msg}")
                return error_msg, False
            
            print("🔍 Getting context from RAG system...")
            # Get context from RAG system
            context = self.rag.get_context_for_query(user_goal, max_tokens=2500)
            print(f"✅ Context retrieved: {len(context)} characters")
            
            # Multi-agent NotebookLM-style
            if quality == "notebook":
                print("🧩 Using multi-agent NotebookLM-style pipeline")
                dialogue = self._generate_with_multi_agent(user_goal, context, tone, difficulty, provider)
                return dialogue, True

            # Choose AI provider based on preference and availability
            print(f"🤖 Selecting provider: {provider}")
            if provider == "puter" or (provider == "auto" and PUTER_AVAILABLE):
                print("🟢 Using Puter.js")
                dialogue = self._generate_with_puter(user_goal, context, tone, difficulty)
            elif provider == "google" or (provider == "auto" and os.getenv('GOOGLE_API_KEY') and GOOGLE_AI_AVAILABLE):
                print("🟢 Using Google AI")
                google_api_key = os.getenv('GOOGLE_API_KEY')
                dialogue = self._generate_with_google(user_goal, context, tone, difficulty, google_api_key)
            else:
                print("🟡 Using mock dialogue")
                dialogue = self._generate_mock_dialogue(user_goal, context, tone, difficulty)
            
            print(f"✅ Dialogue generated: {len(dialogue)} characters")
            return dialogue, True
            
        except Exception as e:
            error_msg = f"Error generating dialogue: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg, False
    
    def _generate_with_puter(self, user_goal: str, context: str, tone: str, difficulty: str) -> str:
        """Generate dialogue using Puter.js (free, no API key required)."""
        try:
            print("🔧 Creating Puter.js generator...")
            from src.puter_dialogue_generator import PuterDialogueGenerator, PuterDialogueConfig
            
            config = PuterDialogueConfig(
                tone=tone,
                level=difficulty,
                max_turns=12,
                temperature=0.7,
                model_name="gpt-4o-mini"  # Free model via Puter.js
            )
            
            generator = PuterDialogueGenerator(config)
            dialogue = generator.generate(user_goal, context)
            
            print("✅ Puter.js dialogue generated successfully")
            return dialogue
            
        except Exception as e:
            print(f"⚠️ Puter.js generation failed: {e}, falling back to mock")
            return self._generate_mock_dialogue(user_goal, context, tone, difficulty)
    
    def _generate_with_google(self, user_goal: str, context: str, tone: str, difficulty: str, api_key: str) -> str:
        """Generate dialogue using Google AI."""
        if not GOOGLE_AI_AVAILABLE:
            return self._generate_mock_dialogue(user_goal, context, tone, difficulty)
            
        try:
            import google.ai.generativelanguage as glm
            
            prompt = f"""
You are DialogueAI, creating an engaging two-person dialogue between a Host (👩) and Expert (👨).

Guidelines:
- Tone: {tone}
- Difficulty: {difficulty}
- Max turns: 12 (each turn is Host then Expert)
- Use short, natural conversation
- Start broad, then deepen based on CONTEXT
- Do not include citations or bracketed sources in the output
- Must alternate speakers starting with Host
- Output format:
👩 Host: <line>
👨 Expert: <line>

USER GOAL: {user_goal}

CONTEXT (use selectively):
{context}

Generate the dialogue now.
"""
            
            # Try the newer API first
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                return response.text
            except:
                # Fallback to older API
                client = glm.GenerativeServiceClient()
                
                request = glm.GenerateTextRequest(
                    model='models/text-bison-001',
                    prompt=glm.TextPrompt(text=prompt),
                    temperature=0.7,
                    max_output_tokens=2000
                )
                
                response = client.generate_text(request)
                
                if response.candidates:
                    return response.candidates[0].output
                else:
                    return self._generate_mock_dialogue(user_goal, context, tone, difficulty)
            
        except Exception as e:
            print(f"⚠️ Google AI generation failed: {e}, falling back to mock")
            return self._generate_mock_dialogue(user_goal, context, tone, difficulty)
    
    def _generate_mock_dialogue(self, user_goal: str, context: str, tone: str, difficulty: str) -> str:
        """Generate a mock dialogue when AI services are unavailable."""
        topic = user_goal.split()[-3:] if len(user_goal.split()) > 3 else ["the document content"]
        topic_str = " ".join(topic)
        
        return f"""👦 Learner: Hi! I'd like to understand {topic_str} from the document you processed.

👨 Expert: Hello! I'd be happy to explain {topic_str}. Based on the document, this is a fascinating topic with several key aspects.

👦 Learner: That sounds interesting! Can you break down the main concepts for me?

👨 Expert: Absolutely! The document covers several important points that I can walk you through systematically.

👦 Learner: What's the most important thing I should understand first?

👨 Expert: Great question! The foundation of understanding {topic_str} starts with grasping the core principles outlined in the source material.

👦 Learner: How does this apply in real-world situations?

👨 Expert: The practical applications are quite extensive. The document highlights several use cases and implementations that demonstrate its value.

👦 Learner: Are there any challenges or limitations I should be aware of?

👨 Expert: Yes, like any concept, there are important considerations and potential challenges that the document addresses.

👦 Learner: This has been really helpful! Is there anything else I should explore?

👨 Expert: I'm glad you found it useful! The document contains additional insights that would deepen your understanding even further.

[Note: This is a demonstration dialogue. Connect your API keys in the .env file for AI-generated content!]"""


# Global storage for dialogue generators (in production, use proper session management)
dialogue_generators = {}


@app.route('/')
def index():
    """Home page with file upload form."""
    print("🏠 Home page accessed")
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    print("📤 Upload request received")
    
    if 'file' not in request.files:
        print("❌ No file in request")
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        print("❌ Empty filename")
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        
        print(f"📁 Processing file: {filename} -> {file_path}")
        
        try:
            # Save uploaded file
            file.save(file_path)
            print(f"✅ File saved to: {file_path}")
            
            # Create new dialogue generator for this session
            session_dialogue_gen = DebugWebDialogueGenerator()
            
            # Process document
            content, stats, success = session_dialogue_gen.process_document(file_path)
            
            if success:
                # Store the dialogue generator for this session
                dialogue_generators[file_id] = session_dialogue_gen
                print(f"✅ Stored dialogue generator for session: {file_id}")
                
                # Store session data (in production, use proper session management)
                session_data = {
                    'file_id': file_id,
                    'filename': filename,
                    'stats': stats,
                    'processed': True
                }
                
                print(f"✅ Rendering processed_simple.html with stats: {stats}")
                return render_template('processed_simple.html', 
                                     filename=filename,
                                     stats=stats,
                                     file_id=file_id)
            else:
                print(f"❌ Processing failed: {content}")
                flash(f'Error processing file: {content}')
                return redirect(url_for('index'))
                
        except Exception as e:
            print(f"❌ Upload error: {str(e)}")
            flash(f'Error uploading file: {str(e)}')
            return redirect(url_for('index'))
        
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Cleaned up: {file_path}")
    
    else:
        print(f"❌ Invalid file type: {file.filename}")
        flash('Invalid file type. Please upload PDF, TXT, or DOCX files.')
        return redirect(url_for('index'))


@app.route('/generate', methods=['POST'])
def generate():
    """Generate dialogue based on user input."""
    print("🗣️ Generate request received")
    
    try:
        user_goal = request.form.get('user_goal', '').strip()
        tone = request.form.get('tone', 'conversational')
        difficulty = request.form.get('difficulty', 'intermediate')
        provider = request.form.get('provider', 'auto')
        quality = request.form.get('quality', 'standard')
        file_id = request.form.get('file_id', '')
        filename = request.form.get('filename', 'document')
        
        print(f"📝 Form data received:")
        print(f"   - user_goal: {user_goal}")
        print(f"   - tone: {tone}")
        print(f"   - difficulty: {difficulty}")
        print(f"   - provider: {provider}")
        print(f"   - quality: {quality}")
        print(f"   - file_id: {file_id}")
        print(f"   - filename: {filename}")
        
        if not user_goal:
            print("❌ No user goal provided")
            flash('Please provide a goal for the dialogue')
            return redirect(url_for('index'))
        
        # Get the dialogue generator for this session
        if file_id not in dialogue_generators:
            print(f"❌ Session {file_id} not found in dialogue_generators")
            print(f"Available sessions: {list(dialogue_generators.keys())}")
            flash('Session expired. Please upload your document again.')
            return redirect(url_for('index'))
        
        session_dialogue_gen = dialogue_generators[file_id]
        print(f"✅ Retrieved dialogue generator for session: {file_id}")
        
        # Generate dialogue with selected provider
        print("🚀 Starting dialogue generation...")
        dialogue, success = session_dialogue_gen.generate_dialogue(user_goal, tone, difficulty, provider, quality)
        
        if success:
            print("✅ Dialogue generation successful!")
            
            # Do not save to disk; render inline only on the website
            print("💾 Skipping file save; rendering inline only")

            # Clean up the dialogue generator to free memory
            if file_id in dialogue_generators:
                del dialogue_generators[file_id]
                print(f"🗑️ Cleaned up session: {file_id}")
            
            print("✅ Rendering result.html")
            return render_template('result.html',
                                 dialogue=dialogue,
                                 user_goal=user_goal,
                                 tone=tone,
                                 difficulty=difficulty,
                                 filename=filename)
        else:
            print(f"❌ Dialogue generation failed: {dialogue}")
            flash(f'Error generating dialogue: {dialogue}')
            return redirect(url_for('index'))
    
    except Exception as e:
        print(f"❌ Generate error: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error: {str(e)}')
        return redirect(url_for('index'))


@app.route('/download/<filename>')
def download_file(filename):
    """Download generated dialogue file."""
    try:
        file_path = os.path.join(RESULTS_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            flash('File not found')
            return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error downloading file: {str(e)}')
        return redirect(url_for('index'))


@app.route('/api/status', methods=['GET'])
def api_status():
    import os
    return jsonify({
        'elevenlabs_key_present': bool(os.getenv('ELEVENLABS_API_KEY')),
        'host_voice': os.getenv('ELEVEN_HOST_VOICE_ID'),
        'expert_voice': os.getenv('ELEVEN_EXPERT_VOICE_ID')
    })

@app.route('/tts', methods=['POST'])
def tts_audio():
    """Generate audio from provided text with alternating voices (Edge TTS)."""
    try:
        text = request.form.get('text', '').strip()
        if not text:
            return ('Missing text', 400)
        try:
            import edge_tts
            import asyncio, io
        except Exception:
            return jsonify({'error': 'edge-tts not installed. Run: pip install edge-tts'}), 500

        async def synth_dual(input_text: str):
            male_voice = 'en-US-GuyNeural'
            female_voice = 'en-US-JennyNeural'
            lines = [ln.strip() for ln in input_text.splitlines() if ln.strip()]
            buf = io.BytesIO()
            for ln in lines:
                is_expert = ln.startswith('👨') or ln.lower().startswith('expert:')
                voice = male_voice if is_expert else female_voice
                spoken = ln.split(':', 1)[1].strip() if ':' in ln else ln
                comm = edge_tts.Communicate(spoken, voice=voice)
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        buf.write(chunk["data"])
            buf.seek(0)
            return buf
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_buf = loop.run_until_complete(synth_dual(text))
        loop.close()
        return send_file(audio_buf, mimetype='audio/mpeg', as_attachment=False, download_name='dialogue_dual.mp3')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tts_eleven', methods=['POST'])
def tts_eleven():
    """Generate audio using ElevenLabs, alternating voices by speaker labels.
    Falls back to Edge TTS if ElevenLabs returns a non-200 status (e.g., 401).
    """
    try:
        text = request.form.get('text', '').strip()
        if not text:
            return ('Missing text', 400)
        import os, io, requests
        api_key = os.getenv('ELEVENLABS_API_KEY')
        if not api_key:
            return jsonify({'error': 'ELEVENLABS_API_KEY not set in environment'}), 500
        host_voice_id = os.getenv('ELEVEN_HOST_VOICE_ID', '21m00Tcm4TlvDq8ikWAM')  # Rachel
        expert_voice_id = os.getenv('ELEVEN_EXPERT_VOICE_ID', 'pNInz6obpgDQGcFmaJgB')  # Adam
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        buf = io.BytesIO()
        use_edge_fallback = False
        last_err_status = None
        last_err_detail = None
        for ln in lines:
            is_expert = ln.startswith('👨') or ln.lower().startswith('expert:')
            voice_id = expert_voice_id if is_expert else host_voice_id
            spoken = ln.split(':', 1)[1].strip() if ':' in ln else ln
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
            payload = {
                "text": spoken,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.4,
                    "similarity_boost": 0.7,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            headers = {
                "xi-api-key": api_key,
                "accept": "audio/mpeg",
                "Content-Type": "application/json"
            }
            try:
                with requests.post(url, json=payload, headers=headers, stream=True) as r:
                    if r.status_code != 200:
                        # Capture error and fall back to Edge TTS
                        last_err_status = r.status_code
                        last_err_detail = r.text[:500]
                        use_edge_fallback = True
                        break
                    for chunk in r.iter_content(chunk_size=4096):
                        if chunk:
                            buf.write(chunk)
            except Exception as e:
                # Network error; fall back
                last_err_status = 500
                last_err_detail = f"ElevenLabs request failed: {e}"
                use_edge_fallback = True
                break
        if not use_edge_fallback:
            buf.seek(0)
            return send_file(buf, mimetype='audio/mpeg', as_attachment=False, download_name='dialogue_eleven.mp3')
        # Fallback to Edge TTS
        try:
            import edge_tts
            import asyncio
            async def synth_dual(input_text: str):
                male_voice = 'en-US-GuyNeural'
                female_voice = 'en-US-JennyNeural'
                lines2 = [ln.strip() for ln in input_text.splitlines() if ln.strip()]
                buf2 = io.BytesIO()
                for ln2 in lines2:
                    is_expert2 = ln2.startswith('👨') or ln2.lower().startswith('expert:')
                    voice2 = male_voice if is_expert2 else female_voice
                    spoken2 = ln2.split(':', 1)[1].strip() if ':' in ln2 else ln2
                    comm = edge_tts.Communicate(spoken2, voice=voice2)
                    async for chunk in comm.stream():
                        if chunk["type"] == "audio":
                            buf2.write(chunk["data"])
                buf2.seek(0)
                return buf2
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            audio_buf = loop.run_until_complete(synth_dual(text))
            loop.close()
            resp = send_file(audio_buf, mimetype='audio/mpeg', as_attachment=False, download_name='dialogue_dual_fallback.mp3')
            try:
                resp.headers['X-TTS-Fallback'] = f"edge-tts; cause={last_err_status}"
            except Exception:
                pass
            return resp
        except Exception:
            if last_err_status and last_err_detail:
                return jsonify({'error': f'ElevenLabs HTTP {last_err_status}', 'detail': last_err_detail}), last_err_status
            return jsonify({'error': 'TTS fallback failed and no audio generated'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tts_dual', methods=['POST'])
def tts_dual():
    """Generate audio from provided text with alternating voices (Edge TTS)."""
    try:
        text = request.form.get('text', '').strip()
        if not text:
            return ('Missing text', 400)
        try:
            import edge_tts
            import asyncio, io
        except Exception:
            return jsonify({'error': 'edge-tts not installed. Run: pip install edge-tts'}), 500

        async def synth_dual(input_text: str):
            male_voice = 'en-US-GuyNeural'
            female_voice = 'en-US-JennyNeural'
            lines = [ln.strip() for ln in input_text.splitlines() if ln.strip()]
            buf = io.BytesIO()
            for ln in lines:
                is_expert = ln.startswith('👨') or ln.lower().startswith('expert:')
                voice = male_voice if is_expert else female_voice
                spoken = ln.split(':', 1)[1].strip() if ':' in ln else ln
                comm = edge_tts.Communicate(spoken, voice=voice)
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        buf.write(chunk["data"])
            buf.seek(0)
            return buf
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_buf = loop.run_until_complete(synth_dual(text))
        loop.close()
        return send_file(audio_buf, mimetype='audio/mpeg', as_attachment=False, download_name='dialogue_dual.mp3')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔧 DialogueAI Debug Web Application Starting...")
    print("📍 Local URL: http://localhost:5001")
    print("📍 Network URL: http://127.0.0.1:5001")
    print("🛑 Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

    try:
        app.run(debug=True, host='127.0.0.1', port=5001, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        input("Press Enter to exit...")
    def _generate_with_multi_agent(self, user_goal: str, context: str, tone: str, difficulty: str, provider: str) -> str:
        """Generate dialogue using the multi-agent pipeline (NotebookLM-style)."""
        cfg = MultiAgentConfig(tone=tone, level=difficulty, max_turns=12, style_mode="notebook")
        composer = MultiAgentDialogue(cfg)
        return composer.generate(user_goal, context, provider)
