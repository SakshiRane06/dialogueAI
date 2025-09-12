# 🚀 Quick Setup Guide

## For Fresh Device Installation

### **1. Clone Repository**
```bash
git clone https://github.com/YOUR_USERNAME/dialogueAI.git
cd dialogueAI
```

### **2. Install Python Dependencies**
```bash
pip install -r requirements.txt
```

### **2.5. Set up API Keys (Optional)**
```bash
# Copy the production environment file with real API keys
cp .env.production .env

# OR use the system without API keys (Puter.js works for free!)
# The default .env file is already configured for Puter.js
```

### **3. Start the Application**

**Option A: Web Interface (Recommended)**
```bash
python debug_web_app.py
```
Then open: http://localhost:5001

**Option B: Windows Batch File**
```bash
START_HERE.bat
```

**Option C: Command Line**
```bash
python -m src.cli --file data/sample.txt --ask "Explain RAG like a podcast"
```

### **4. Test Your Setup**
```bash
# Test core functionality (no API keys needed)
python test_demo.py

# Test Puter.js integration  
python test_puter_integration.py
```

## 🎯 What's Included

✅ **API Keys**: Pre-configured in .env file  
✅ **Sample Data**: Ready to test with data/sample.txt  
✅ **All Dependencies**: Listed in requirements.txt  
✅ **Multiple Interfaces**: Web, CLI, batch files  
✅ **Free AI Provider**: Puter.js works without API keys  
✅ **Documentation**: WARP.md for development guidance  

## 🎉 Ready to Use!

1. **Upload** your document (PDF, TXT, DOCX)
2. **Describe** what you want to learn
3. **Generate** intelligent dialogue
4. **Download** or copy your results

The system uses intelligent context analysis to create meaningful conversations based on your actual document content!
