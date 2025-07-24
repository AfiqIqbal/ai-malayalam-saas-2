
# Malayalam Voice Assistant with Ollama

A voice-enabled AI assistant with native Malayalam support, powered by Ollama and local language models. This project enables natural voice interactions in both English and Malayalam, with automatic translation between languages.

## 🌟 Key Features

* **Bilingual Support**: Seamlessly understand and respond in both Malayalam and English
* **Voice Interface**: Speech-to-text and text-to-speech with real-time translation
* **Local Processing**: Runs entirely on your machine with Ollama, ensuring data privacy
* **Lightweight**: Uses efficient models that run well on consumer hardware
* **Easy to Use**: Simple command-line interface for quick interactions

---

## 🚀 Quick Start

### Prerequisites

* Python 3.10+
* [Ollama](https://ollama.ai) installed and running locally
* A working microphone and speakers
* FFmpeg & PortAudio installed

---

## 🛠️ Installation

You can set up the project using **either Conda** or **pip+venv**:

---

### 🔹 Option 1: Using Conda (Recommended)

1. Create a Conda environment:

   ```bash
   conda create -n malayalam-voice python=3.10
   conda activate malayalam-voice
   ```

2. Install Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Install FFmpeg and PortAudio (if not already installed):

   * On Ubuntu/Debian:

     ```bash
     sudo apt install ffmpeg portaudio19-dev
     ```
   * On macOS:

     ```bash
     brew install ffmpeg portaudio
     ```
   * On Windows: Use [FFmpeg Windows build](https://ffmpeg.org/download.html) and install PortAudio via `conda`:

     ```bash
     conda install -c anaconda portaudio
     ```

---

### 🔹 Option 2: Using pip and Virtualenv

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/malayalam-ai-saas.git
   cd malayalam-ai-saas
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   .\venv\Scripts\activate        # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Model Setup (Ollama)

1. Pull the required model:

   ```bash
   ollama pull mistral
   ```

2. (Optional) Test integration:

   ```bash
   python test_ollama.py
   ```

---

## 🚀 Usage

### CLI Version

```bash
python voice_assistant_nllb.py
```

* Press Enter to start recording
* Speak in English or Malayalam
* Get a voice/text response in your language

---

### API Server (FastAPI)

Start the backend server:

```bash
uvicorn backend.main:app --reload
```

Access API docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📁 Project Structure

```
malayalam-ai-saas/
├── backend/                  # FastAPI app
│   └── main.py
├── voice_assistant_nllb.py   # CLI-based assistant
├── requirements.txt
├── requirements_minimal.txt
├── .env.example
└── recordings/
```

---

## 📦 Exporting Conda Environment (Optional)

To export your working conda environment to a YAML file:

```bash
conda env export --from-history > environment.yml
```

To recreate it on another system:

```bash
conda env create -f environment.yml
```

---

## 📚 Documentation

Visit our [Documentation Portal](https://docs.malayalam-ai.com) for detailed API usage and deployment guides.

---

## 🛣️ Roadmap

### Phase 1: Core Platform

* [x] Basic Malayalam text processing
* [x] Translation pipeline
* [ ] Malayalam ASR integration
* [ ] Basic RAG implementation
* [ ] API endpoints

### Phase 2: Business Features

* [ ] Multi-tenant setup
* [ ] Knowledge base
* [ ] Analytics
* [ ] Billing

### Phase 3: Advanced Capabilities

* [ ] Sentiment analysis
* [ ] Intent recognition
* [ ] Payment integration

---

## 🤝 Contributing

We welcome contributions! See our [Contributing Guidelines](CONTRIBUTING.md).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


