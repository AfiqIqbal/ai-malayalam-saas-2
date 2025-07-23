# Malayalam Voice Assistant with Ollama

A voice-enabled AI assistant with native Malayalam support, powered by Ollama and local language models. This project enables natural voice interactions in both English and Malayalam, with automatic translation between languages.

## 🌟 Key Features

- **Bilingual Support**: Seamlessly understand and respond in both Malayalam and English
- **Voice Interface**: Speech-to-text and text-to-speech with real-time translation
- **Local Processing**: Runs entirely on your machine with Ollama, ensuring data privacy
- **Lightweight**: Uses efficient models that run well on consumer hardware
- **Easy to Use**: Simple command-line interface for quick interactions

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally (download from [ollama.ai](https://ollama.ai))
- A working microphone and speakers

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/malayalam-ai-saas.git
   cd malayalam-ai-saas
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # OR
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Pull the required Ollama model:
   ```bash
   ollama pull mistral:7b-instruct
   ```

### Usage

1. Start the voice assistant:
   ```bash
   python voice_assistant_nllb.py
   ```

2. Press Enter to start recording your voice
3. Speak clearly in either English or Malayalam
4. The assistant will process your request and respond accordingly

## 🏗️ Project Structure

```
malayalam-ai-saas/
├── voice_assistant_nllb.py  # Main voice assistant script
├── requirements.txt         # Python dependencies
├── recordings/             # Directory for saved audio recordings
└── README.md               # This documentation file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- FFmpeg (for audio processing)

### Installation

1. Clone the repository and set up the environment:
   ```bash
   git clone https://github.com/your-username/malayalam-ai-ollama.git
   cd malayalam-ai-ollama
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Ollama model (if not already downloaded):
   ```bash
   ollama pull mistral  # Or any other model you prefer
   ```

4. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

5. Open your browser and go to `http://localhost:8000/docs` to test the API endpoints.

## 🛠️ Usage

### Testing the API

1. Start the server if not already running:
   ```bash
   uvicorn main:app --reload
   ```

2. Use the interactive API documentation at `http://localhost:8000/docs` to test the endpoints.

### Testing Ollama Integration

Run the test script to verify everything is working:

```bash
python test_ollama.py
```

### Cleaning Up

To clean up old files and set up the new project structure:

```bash
python cleanup_and_setup.py
```

2. Install dependencies:
   ```bash
   pip install -r requirements_minimal.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Update the .env file with your configuration
   ```

### 🎯 Core Features

### 1. Malayalam Speech Recognition
- High-accuracy Automatic Speech Recognition (ASR) for Malayalam
- Real-time speech-to-text conversion
- Noise reduction and accent adaptation

### 2. Intelligent Response Generation
- Context-aware responses using RAG technology
- Business-specific knowledge base integration
- Multi-turn conversation handling

### 3. Text-to-Speech
- Natural-sounding Malayalam speech synthesis
- Expressive and clear voice output
- Adjustable speaking rate and tone

### 4. Business Dashboard
- Conversation analytics and insights
- Performance metrics and KPIs
- Easy knowledge base management

## 🛠️ Development Roadmap

### Phase 1: Core Platform (Current)
- [x] Basic Malayalam text processing
- [x] Translation pipeline
- [ ] Malayalam ASR integration
- [ ] Basic RAG implementation
- [ ] API endpoints for core functionality

### Phase 2: Business Features
- [ ] Multi-tenant architecture
- [ ] Business onboarding workflow
- [ ] Knowledge base management
- [ ] Basic analytics dashboard
- [ ] Billing and subscription management
- [ ] Role-based access control

### Phase 3: Advanced Features
- [ ] Real-time call monitoring
- [ ] Sentiment analysis
- [ ] Automated quality assurance
- [ ] Custom voice models
- [ ] Integration with popular CRMs

### Phase 4: Scale & Optimization
- [ ] Auto-scaling infrastructure
- [ ] Advanced analytics and reporting
- [ ] Custom ML model training
- [ ] Multi-region deployment

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- FFmpeg
- PortAudio (for audio processing)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/malayalam-ai-saas.git
   cd malayalam-ai-saas
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running the Application

Start the development server:
```bash
uvicorn backend.main:app --reload
```

## 📚 Documentation

For detailed documentation, please visit our [Documentation Portal](https://docs.malayalam-ai.com).

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌐 Connect With Us

- [Website](https://malayalam-ai.com)
- [Twitter](https://twitter.com/malayalam_ai)
- [LinkedIn](https://linkedin.com/company/malayalam-ai)
- [Blog](https://blog.malayalam-ai.com)

## 🙏 Acknowledgments

- Built with ❤️ in Kerala, India
- Special thanks to the open-source community for their invaluable contributions
### Phase 3: Advanced Capabilities
- [ ] Sentiment analysis
- [ ] Intent recognition
- [ ] Payment integration
- [ ] Advanced analytics

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For business inquiries or support, please contact us at [contact@malayalam-ai.com](mailto:contact@malayalam-ai.com)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
