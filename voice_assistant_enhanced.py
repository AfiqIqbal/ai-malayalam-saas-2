# Add this at the very top of the file
import sys
import os
import json
import time
import hashlib
import requests  # Added missing import
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Dict, Optional, Tuple
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
from pathlib import Path
from loguru import logger
from translate import Translator as TranslateTranslator

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5  # Default recording duration
OLLAMA_BASE_URL = "http://localhost:11434/api"
MODEL_NAME = "mistral:7b-instruct-q4_0"  # Using quantized 4-bit model
LANGUAGE = "ml-IN"  # Malayalam (India)
CACHE_TTL_HOURS = 24  # Cache responses for 24 hours
MAX_HISTORY_LENGTH = 10  # Maximum number of messages to keep in history

class ResponseCache:
    """Cache for storing LLM responses with TTL"""
    def __init__(self, ttl_hours: int = 24):
        self.cache = {}
        self.ttl = timedelta(hours=ttl_hours)
    
    def get(self, key: str) -> Optional[str]:
        """Get a cached response if it exists and isn't expired"""
        if key not in self.cache:
            return None
            
        cached_time, response = self.cache[key]
        if datetime.now() - cached_time > self.ttl:
            del self.cache[key]
            return None
            
        return response
    
    def set(self, key: str, response: str) -> None:
        """Store a response in the cache"""
        self.cache[key] = (datetime.now(), response)
    
    def clear_expired(self) -> None:
        """Remove expired cache entries"""
        now = datetime.now()
        expired_keys = [k for k, (t, _) in self.cache.items() 
                       if now - t > self.ttl]
        for k in expired_keys:
            del self.cache[k]

class ConversationManager:
    """Manages conversation history and context"""
    def __init__(self, max_history: int = 10):
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history"""
        self.history.append({"role": role, "content": content})
        
        # Trim history if it gets too long
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self, max_tokens: int = 1000) -> str:
        """Get conversation context as a formatted string"""
        context = []
        current_length = 0
        
        # Add messages in reverse order until we hit the token limit
        for msg in reversed(self.history):
            msg_str = f"{msg['role']}: {msg['content']}"
            if current_length + len(msg_str) > max_tokens:
                break
            context.insert(0, msg_str)
            current_length += len(msg_str)
        
        return "\n".join(context)
    
    def summarize(self) -> str:
        """Generate a summary of the conversation"""
        # Simple implementation - just join recent messages
        # In a production system, you might want to use an LLM for better summarization
        return " ".join([msg["content"] for msg in self.history[-3:]])
    
    def clear(self) -> None:
        """Clear the conversation history"""
        self.history = []

# Initialize global components
response_cache = ResponseCache(ttl_hours=CACHE_TTL_HOURS)
conversation = ConversationManager(max_history=MAX_HISTORY_LENGTH)

class SimpleTranslator:
    def __init__(self, max_chunk_length=400):
        self.translator = None
        self.max_chunk_length = max_chunk_length
        logger.info("Simple Translator initialized")
        
    def _split_into_chunks(self, text, chunk_size=None):
        """Split text into chunks of specified size"""
        if chunk_size is None:
            chunk_size = self.max_chunk_length
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
    def translate(self, text, src_lang="ml", tgt_lang="en"):
        """Translate text with chunking support"""
        if not text:
            return text
            
        try:
            # Initialize translator
            self.translator = TranslateTranslator(
                to_lang=tgt_lang,
                from_lang=src_lang
            )
            
            # Translate the text
            return self.translator.translate(text)
            
        except Exception as e:
            logger.error(f"Translation error ({src_lang}->{tgt_lang}): {e}")
            return text

# Initialize translator
translator = SimpleTranslator()

def is_english(text):
    """Check if text is in English"""
    try:
        return all(ord(c) < 128 for c in text if c.isalpha())
    except:
        return True

class VoiceAssistant:
    def __init__(self, input_device=None):
        self.recognizer = sr.Recognizer()
        self.translator = translator
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.recording = False
        self.audio_data = None
        self.input_device_id = input_device or 0
        self.setup_audio_device()

    def setup_audio_device(self):
        """Configure audio input device"""
        try:
            devices = sd.query_devices()
            print("\n=== Available Audio Devices ===")
            for i, device in enumerate(devices):
                print(f"{i}: {device['name']} (Input Channels: {device['max_input_channels']})")
            print("==============================\n")
            
            if self.input_device_id is None:
                input_devices = [i for i, d in enumerate(devices) 
                               if d['max_input_channels'] > 0]
                if input_devices:
                    self.input_device_id = input_devices[0]
            
            logger.info(f"Using audio device: {devices[self.input_device_id]['name']}")
            
        except Exception as e:
            logger.error(f"Audio device error: {e}")
            print("\nError: Could not initialize audio. Using default settings.")
            self.input_device_id = sd.default.device[0] if hasattr(sd.default, 'device') else 0

    def record_audio(self, duration=5, sample_rate=SAMPLE_RATE):
        """Record audio from microphone"""
        try:
            logger.info(f"Recording for {duration} seconds...")
            print("Speak now...")
            
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32',
                device=self.input_device_id
            )
            sd.wait()
            
            logger.info("Recording finished")
            return audio_data.flatten(), sample_rate
            
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None, None

    def transcribe_audio(self, audio_data, sample_rate):
        """Convert speech to text"""
        try:
            audio_data = (audio_data * 32767).astype(np.int16)
            audio_segment = sr.AudioData(
                audio_data.tobytes(),
                sample_rate=sample_rate,
                sample_width=audio_data.dtype.itemsize
            )
            
            try:
                text = self.recognizer.recognize_google(audio_segment, language='ml-IN')
                logger.info("Malayalam transcription successful")
                return text
            except:
                text = self.recognizer.recognize_google(audio_segment, language='en-US')
                logger.info("English transcription successful")
                return text
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def get_ollama_response(self, prompt: str) -> str:
        """Get response from Ollama with caching"""
        # Check cache first
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        cached_response = response_cache.get(cache_key)
        if cached_response:
            logger.info("Cache hit!")
            return cached_response
            
        try:
            # Call Ollama API
            response = requests.post(
                f"{OLLAMA_BASE_URL}/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get('response', '').strip()
            
            # Cache the response
            response_cache.set(cache_key, response_text)
            return response_text
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return "I'm sorry, I couldn't process your request at the moment."

    def process_query(self, text: str) -> str:
        """Process user query with conversation context"""
        # Add user message to history
        conversation.add_message("User", text)
        
        # Get conversation context
        context = conversation.get_context()
        
        # Create prompt with context
        prompt = f"""Previous conversation:
{context}

Current query: {text}

Assistant:"""
        
        # Get response from LLM
        response = self.get_ollama_response(prompt)
        
        # Add assistant response to history
        conversation.add_message("Assistant", response)
        
        return response

    def speak(self, text: str, language: str = 'ml') -> None:
        """Convert text to speech"""
        if not text:
            return
            
        print(f"\n[Assistant]: {text}")
        
        try:
            # Create a temporary file that will be automatically deleted when closed
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
                temp_file = fp.name
            
            try:
                # Generate speech and save to temp file
                tts = gTTS(text=text, lang=language, slow=False)
                tts.save(temp_file)
                
                # Initialize pygame mixer and play the audio
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                clock = pygame.time.Clock()
                while pygame.mixer.music.get_busy():
                    clock.tick(10)
                
                # Stop and unload the music to release the file
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                
            except Exception as e:
                logger.error(f"Speech synthesis/playback error: {e}")
                print("(Audio playback failed - showing text only)")
                
            finally:
                # Ensure the temp file is deleted
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Could not delete temp file {temp_file}: {e}")
                    
        except Exception as e:
            logger.error(f"TTS error: {e}")
            print("(Audio playback failed - showing text only)")

def main():
    """Main function to run the enhanced voice assistant"""
    print("\n=== Enhanced Malayalam Voice Assistant ===")
    print(f"Using model: {MODEL_NAME}")
    print("Press Enter to start recording (or 'q' to quit)\n")
    
    try:
        assistant = VoiceAssistant()
        
        while True:
            try:
                user_input = input("Press Enter to start recording (or 'q' to quit)...")
                if user_input.lower() == 'q':
                    print("\nExiting...")
                    break
                    
                # Record and transcribe
                print("\nRecording... (speak now)")
                audio_data, sample_rate = assistant.record_audio()
                if audio_data is None:
                    print("No audio recorded. Please try again.")
                    continue
                    
                print("\nProcessing your request...")
                text = assistant.transcribe_audio(audio_data, sample_rate)
                if not text:
                    print("Could not transcribe audio. Please try again.")
                    continue
                
                # Process the query
                if not is_english(text):
                    english_text = translator.translate(text, src_lang="ml", tgt_lang="en")
                    print(f"\nYou said (translated to English): {english_text}")
                    
                    # Get response in English
                    response = assistant.process_query(english_text)
                    
                    # Translate response to Malayalam
                    malayalam_response = translator.translate(response, src_lang="en", tgt_lang="ml")
                    print(f"\nAssistant (Malayalam): {malayalam_response}")
                    assistant.speak(malayalam_response, 'ml')
                    
                else:
                    print(f"\nYou said: {text}")
                    response = assistant.process_query(text)
                    print(f"\nAssistant: {response}")
                    assistant.speak(response, 'en')
                    
                print("\n" + "-"*30)
                
                # Periodically clean up expired cache entries
                if len(conversation.history) % 5 == 0:
                    response_cache.clear_expired()
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\nAn error occurred: {str(e)}\nPlease try again.")
                
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        print(f"\nA critical error occurred: {str(e)}")
    finally:
        print("\nThank you for using the Enhanced Malayalam Voice Assistant!")
        # Save conversation history before exiting
        try:
            os.makedirs('conversations', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(f'conversations/conversation_{timestamp}.json', 'w') as f:
                json.dump(conversation.history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")

if __name__ == "__main__":
    main()
