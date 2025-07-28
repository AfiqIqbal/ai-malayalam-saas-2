import os
import sys
import json
import time
import wave
import struct
import logging
import tempfile
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union

import pyaudio
import speech_recognition as sr
from gtts import gTTS
import pygame
import soundfile as sf
import sounddevice as sd
from ctransformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('voice_assistant.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
SILENCE_THRESHOLD = 1000  # Adjust based on your microphone
SILENCE_DURATION = 1.0  # seconds of silence to mark end of speech
MAX_RECORDING_SECONDS = 10
# Using a smaller, more accessible model from Hugging Face Hub
MODEL_REPO = "TheBloke/Mistral-7B-v0.1-GGUF"
MODEL_FILE = "mistral-7b-v0.1.Q4_K_M.gguf"  # Quantized model for better performance

class VoiceAssistant:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.operation_timeout = 5.0
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Initialize the language model
        logger.info("Loading language model...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_REPO,
                model_file=MODEL_FILE,
                model_type="mistral",  # Changed to mistral for the new model
                gpu_layers=50,  # Adjust based on your GPU
                context_length=2048,
                batch_size=1,
                threads=4
            )
            logger.info("Language model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to a simpler model...")
            # Fallback to a smaller model if the main one fails
            self.model = AutoModelForCausalLM.from_pretrained(
                "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                model_type="mistral",
                gpu_layers=20,
                context_length=1024
            )
        
        # Conversation history
        self.conversation_history = [
            {"role": "system", "content": "നിങ്ങൾ ഒരു സഹായകനാണ്. എല്ലാ ചോദ്യങ്ങൾക്കും മലയാളത്തിൽ മാത്രം മറുപടി നൽകുക. ചുരുക്കവും വ്യക്തവുമായ ഉത്തരങ്ങൾ നൽകുക."}
        ]

    def setup_audio_device(self, device_index=None):
        """Set up the audio device for recording."""
        try:
            # List available audio devices
            info = self.audio.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            
            print("\n=== Available Audio Devices ===")
            for i in range(num_devices):
                device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
                device_name = device_info.get('name')
                input_channels = device_info.get('maxInputChannels', 0)
                print(f"{i}: {device_name} (Input Channels: {input_channels})")
            print("="*30)
            
            # If no device index is provided, try to find a suitable one
            if device_index is None:
                for i in range(num_devices):
                    device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
                    if device_info.get('maxInputChannels', 0) > 0 and 'microphone' in device_info.get('name', '').lower():
                        device_index = i
                        break
                
                if device_index is None:
                    device_index = self.audio.get_default_input_device_info().get('index')
            
            device_info = self.audio.get_device_info_by_index(device_index)
            logger.info(f"Using audio device: {device_info.get('name')}")
            logger.info(f"Audio settings - Sample rate: {SAMPLE_RATE}Hz, Channels: {CHANNELS}")
            
            return device_index
            
        except Exception as e:
            logger.error(f"Error setting up audio device: {e}")
            return None

    def record_audio(self, duration=5):
        """Record audio from the microphone."""
        logger.info(f"Recording for {duration} seconds...")
        
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        frames = []
        print("Speak now...")
        
        try:
            for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE * duration)):
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)
        except Exception as e:
            logger.error(f"Error during recording: {e}")
            return None
        finally:
            stream.stop_stream()
            stream.close()
            
        logger.info("Recording finished successfully")
        return b''.join(frames)

    def transcribe_audio(self, audio_data):
        """Transcribe audio data to text using Whisper."""
        try:
            # Save audio data to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                wf = wave.open(f.name, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_data)
                wf.close()
                
                # Use speech_recognition to transcribe
                with sr.AudioFile(f.name) as source:
                    audio = self.recognizer.record(source)
                    
                try:
                    # First try to transcribe in Malayalam
                    text = self.recognizer.recognize_google(audio, language='ml-IN')
                    logger.info(f"Transcribed (Malayalam): {text}")
                    return text, 'ml'
                except sr.UnknownValueError:
                    logger.warning("Could not understand audio")
                    return None, None
                except sr.RequestError as e:
                    logger.error(f"Could not request results from Google Speech Recognition service; {e}")
                    return None, None
                
        except Exception as e:
            logger.error(f"Error in transcribe_audio: {e}")
            return None, None
        finally:
            try:
                if os.path.exists(f.name):
                    os.unlink(f.name)
            except Exception as e:
                logger.error(f"Error cleaning up temp file: {e}")

    def text_to_speech(self, text, lang='ml'):
        """Convert text to speech and play it."""
        try:
            # Generate speech using gTTS
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_file = f.name
                tts.save(temp_file)
            
            # Play the audio using pygame
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Clean up
            pygame.mixer.music.unload()
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Could not delete temp file {temp_file}: {e}")
                
        except Exception as e:
            logger.error(f"Error in text_to_speech: {e}")

    def get_llm_response(self, prompt):
        """Get a response from the language model."""
        try:
            # Format the prompt for the Mistral model
            full_prompt = f"<s>[INST] {prompt} [/INST]"
            
            # Generate response
            response = self.model(
                full_prompt,
                max_length=200,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                stop_sequences=["</s>"]
            )
            
            # Clean up the response
            response_text = response.strip()
            response_text = response_text.replace("</s>", "").strip()
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error in get_llm_response: {e}")
            return "ക്ഷമിക്കണം, എനിക്ക് ഒരു മികച്ച മറുപടി ലഭിച്ചില്ല. ദയവായി വീണ്ടും ശ്രമിക്കുക."

    def process_query(self, query, lang='ml'):
        """Process the user query and generate a response."""
        if not query or not query.strip():
            return "എനിക്ക് നിങ്ങളെ മനസ്സിലാകുന്നില്ല. ദയവായി വീണ്ടും ശ്രമിക്കുക."
        
        print(f"\nProcessing query: {query}")
        
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Get the last few messages for context
        context_messages = self.conversation_history[-4:]  # Last 4 messages for context
        
        # Format the prompt with conversation history
        prompt_parts = []
        for msg in context_messages:
            if msg["role"] == "system":
                prompt_parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            else:
                prompt_parts.append(f"Assistant: {msg['content']}")
        
        prompt = "\n".join(prompt_parts) + "\nAssistant:"
        
        # Get response from the model
        response = self.get_llm_response(prompt)
        
        # Add assistant's response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response

    def run(self):
        """Run the voice assistant."""
        print("\n=== Malayalam Voice Assistant ===")
        print("Press Ctrl+C to exit\n")
        
        # Set up audio device
        device_index = self.setup_audio_device()
        
        try:
            while True:
                # Record audio
                audio_data = self.record_audio(duration=5)
                
                if audio_data:
                    # Transcribe audio
                    text, lang = self.transcribe_audio(audio_data)
                    
                    if text:
                        print(f"\nYou ({lang}): {text}")
                        
                        # Process the query
                        response = self.process_query(text, lang)
                        
                        # Print and speak the response
                        print(f"\n[Assistant]: {response}")
                        self.text_to_speech(response, lang='ml' if lang == 'ml' else 'en')
                    else:
                        print("Could not understand audio. Please try again.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            print(f"An error occurred: {e}")
        finally:
            self.audio.terminate()
            pygame.quit()

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
