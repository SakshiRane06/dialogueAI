"""
Simplified DialogueAI Web Application
A basic Flask interface that works without complex dependencies
"""

import os
import uuid
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'dialogueai-demo-key'

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


def process_text_file(file_path):
    """Simple text file processor"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, True
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            return content, True
        except Exception as e:
            return f"Error reading file: {e}", False
    except Exception as e:
        return f"Error: {e}", False


def generate_mock_dialogue(user_goal, filename, tone="conversational", difficulty="intermediate"):
    """Generate a demonstration dialogue"""
    
    dialogue = f"""👦 Learner: Hi! I'd like to understand the content from {filename} that relates to "{user_goal}".

👨 Expert: Hello! I'd be happy to help you explore that topic. Based on the document you've uploaded, this is definitely an interesting area to discuss.

👦 Learner: That sounds great! Can you give me an overview of the main concepts?

👨 Expert: Absolutely! The document covers several key areas that are fundamental to understanding the subject matter. Let me break it down systematically for you.

👦 Learner: What would you say is the most important thing for me to understand first?

👨 Expert: Great question! I'd recommend starting with the core principles. These form the foundation that everything else builds upon.

👦 Learner: How does this apply in real-world scenarios?

👨 Expert: Excellent point! The practical applications are quite extensive. The concepts discussed in the document have direct relevance to many current situations and challenges.

👦 Learner: Are there any common misconceptions or challenges I should be aware of?

👨 Expert: Yes, that's very insightful of you to ask. There are indeed several areas where people commonly get confused or face difficulties when first learning about this topic.

👦 Learner: This has been really helpful! What would you recommend as my next steps?

👨 Expert: I'm glad you found this useful! Based on what we've discussed, I'd suggest diving deeper into the specific areas that resonated most with your goals.

👦 Learner: Thank you! This conversation format really helps me understand the material better.

👨 Expert: You're very welcome! The dialogue approach is indeed an effective way to explore complex topics. It allows for natural progression from basic concepts to more advanced understanding.

[Note: This is a demonstration dialogue. For AI-powered content generation, please configure your API keys in the .env file.]"""
    
    return dialogue


@app.route('/')
def index():
    """Home page with file upload form."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    print(f"📨 Upload request received")  # Debug log
    
    if 'file' not in request.files:
        print("❌ No file in request")
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    print(f"📁 File received: {file.filename}")
    
    if file.filename == '':
        print("❌ Empty filename")
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        print(f"✅ File type allowed: {file.filename}")
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        print(f"💾 Saving to: {file_path}")
        
        try:
            # Save uploaded file
            file.save(file_path)
            print(f"✅ File saved successfully")
            
            # Simple processing for demonstration
            print(f"📝 Processing file: {filename}")
            if filename.lower().endswith('.txt'):
                print("Processing as text file")
                content, success = process_text_file(file_path)
            else:
                print("Processing as other file type (demo mode)")
                content = "File uploaded successfully! (Full processing requires additional dependencies)"
                success = True
            
            print(f"📄 Processing result: success={success}")
            
            if success:
                stats = {
                    'characters': len(content) if isinstance(content, str) else 0,
                    'chunks': max(1, len(content) // 800) if isinstance(content, str) else 1,
                    'embedding_model': 'Demo Mode',
                    'using_openai': False
                }
                
                print(f"📈 Stats: {stats}")
                print(f"🚀 Rendering processed.html")
                
                return render_template('processed.html', 
                                     filename=filename,
                                     stats=stats,
                                     file_id=file_id)
            else:
                flash(f'Error processing file: {content}')
                return redirect(url_for('index'))
                
        except Exception as e:
            flash(f'Error uploading file: {str(e)}')
            return redirect(url_for('index'))
        
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    else:
        flash('Invalid file type. Please upload PDF, TXT, or DOCX files.')
        return redirect(url_for('index'))


@app.route('/generate', methods=['POST'])
def generate():
    """Generate dialogue based on user input."""
    try:
        user_goal = request.form.get('user_goal', '').strip()
        tone = request.form.get('tone', 'conversational')
        difficulty = request.form.get('difficulty', 'intermediate')
        file_id = request.form.get('file_id', '')
        filename = request.form.get('filename', 'document')
        
        if not user_goal:
            flash('Please provide a goal for the dialogue')
            return redirect(url_for('index'))
        
        # Generate dialogue
        dialogue = generate_mock_dialogue(user_goal, filename, tone, difficulty)
        
        # Save dialogue to file
        output_filename = f"dialogue_{file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        output_path = os.path.join(RESULTS_FOLDER, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"DialogueAI Generated Conversation\n")
            f.write(f"Source Document: {filename}\n")
            f.write(f"User Goal: {user_goal}\n")
            f.write(f"Tone: {tone}\n")
            f.write(f"Difficulty: {difficulty}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(dialogue)
        
        return render_template('result.html',
                             dialogue=dialogue,
                             user_goal=user_goal,
                             tone=tone,
                             difficulty=difficulty,
                             filename=filename,
                             output_file=output_filename)
    
    except Exception as e:
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


@app.route('/api/status')
def api_status():
    """API endpoint to check system status."""
    return jsonify({
        'status': 'running - demo mode',
        'openai_available': False,
        'google_available': False,
        'google_ai_installed': False,
        'embedding_fallback': 'Demo Mode (no ML dependencies)'
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 DialogueAI Web Application (Demo Mode)")
    print("📍 Local URL: http://localhost:5000")  
    print("📍 Network URL: http://127.0.0.1:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("📝 Note: This is demo mode - upload TXT files for best results")
    print("="*60 + "\n")
    
    try:
        app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        input("Press Enter to exit...")
