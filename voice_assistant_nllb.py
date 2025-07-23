# Add this at the very top of the file
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the fix before any other imports
try:
    import aifc_fix
except ImportError:
    logger = None  # Will be defined later
    if logger:
        logger.warning("aifc_fix.py not found. Audio recording might not work.")

import time
from datetime import datetime
import sounddevice as sd
import numpy as np
import speech_recognition as sr
from gtts import gTTS
import tempfile
import subprocess
import platform
from scipy.io import wavfile
import requests
import json
import wave
import struct
from pathlib import Path
from loguru import logger
import sys
from translate import Translator as TranslateTranslator

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5  # Default recording duration
OLLAMA_BASE_URL = "http://localhost:11434/api"
MODEL_NAME = "mistral:7b-instruct"  # Using available model
LANGUAGE = "ml-IN"  # Malayalam (India)

class SimpleTranslator:
    def __init__(self, max_chunk_length=400):
        self.translator = None
        self.max_chunk_length = max_chunk_length  # Maximum characters per chunk
        logger.info("Simple Translator initialized")
        
    def _split_into_chunks(self, text, chunk_size=None):
        """Split text into chunks of specified size, trying to break at sentence boundaries"""
        if chunk_size is None:
            chunk_size = self.max_chunk_length
            
        # If text is already short enough, return as single chunk
        if len(text) <= chunk_size:
            return [text]
            
        # Try to split at sentence boundaries first
        sentences = []
        current_sentence = []
        current_length = 0
        
        # Split by common sentence terminators
        for part in text.replace('?', '?\n').replace('!', '!\n').replace('।', '।\n').split('\n'):
            part = part.strip()
            if not part:
                continue
                
            if current_length + len(part) + 1 <= chunk_size:
                current_sentence.append(part)
                current_length += len(part) + 1
            else:
                if current_sentence:
                    sentences.append(' '.join(current_sentence))
                current_sentence = [part]
                current_length = len(part)
                
        if current_sentence:
            sentences.append(' '.join(current_sentence))
            
        # If we couldn't split by sentences, split by words
        if not sentences or any(len(s) > chunk_size for s in sentences):
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)
                    current_length += len(word) + 1
                    
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
            return chunks
            
        return sentences
        
    def translate(self, text, src_lang="ml", tgt_lang="en"):
        """Translate text using the SimpleTranslator with chunking support"""
        if not text or not text.strip():
            return text
            
        try:
            # If text is short, process directly
            if len(text) <= self.max_chunk_length:
                return self._translate_chunk(text, src_lang, tgt_lang)
                
            # Split text into chunks and translate each chunk
            chunks = self._split_into_chunks(text)
            translated_chunks = []
            
            for chunk in chunks:
                if chunk.strip():  # Only process non-empty chunks
                    translated = self._translate_chunk(chunk, src_lang, tgt_lang)
                    translated_chunks.append(translated)
                    
            # Join the translated chunks with spaces
            return ' '.join(translated_chunks)
            
        except Exception as e:
            logger.error(f"Translation error ({src_lang}->{tgt_lang}): {e}")
            return text  # Return original text if translation fails
            
    def _translate_chunk(self, text, src_lang, tgt_lang):
        """Translate a single chunk of text"""
        try:
            # Map language codes to MyMemory format
            lang_map = {
                'ml': 'ml',  # Malayalam
                'en': 'en'   # English
            }
            
            # Initialize translator with source and target languages
            self.translator = TranslateTranslator(
                to_lang=lang_map.get(tgt_lang, 'en'),
                from_lang=lang_map.get(src_lang, 'auto')
            )
            
            # Translate the text
            translated_text = self.translator.translate(text)
            return translated_text
            
        except Exception as e:
            logger.error(f"Chunk translation error ({src_lang}->{tgt_lang}): {e}")
            return text

# Initialize translator
try:
    translator = SimpleTranslator()
    logger.info("Simple Translator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize translator: {e}")
    translator = SimpleTranslator()  # Fallback to SimpleTranslator even if logging fails
    logger.info("Using fallback Simple Translator")

def translate_to_english(text):
    """Translate text from Malayalam to English using NLLB"""
    return translator.translate(text, src_lang="ml", tgt_lang="en")

def is_english(text):
    """Check if the given text is in English
    
    Args:
        text (str): The text to check
        
    Returns:
        bool: True if the text is in English, False otherwise
    """
    try:
        # Simple check: if the text contains any Malayalam Unicode characters, it's not English
        # Malayalam Unicode range: U+0D00 to U+0D7F
        for char in text:
            if 'ഀ' <= char <= 'ൿ':
                return False
        return True
    except Exception as e:
        logger.warning(f"Error in is_english: {e}")
        # Default to True if we can't determine the language
        return True

def translate_to_malayalam(text):
    """Translate text from English to Malayalam using NLLB"""
    return translator.translate(text, src_lang="en", tgt_lang="ml")

class VoiceAssistant:
    def __init__(self, input_device=None):
        """Initialize the voice assistant"""
        self.recognizer = sr.Recognizer()
        self.translator = SimpleTranslator()
        self.sample_rate = 16000
        self.channels = 1
        self.recording = False
        self.audio_data = None
        
        # Set up audio device
        try:
            # Get list of all audio devices
            devices = sd.query_devices()
            
            # Print available devices for debugging
            print("\n=== Available Audio Devices ===")
            for i, device in enumerate(devices):
                print(f"{i}: {device['name']} (Input Channels: {device['max_input_channels']})")
            print("==============================\n")
            
            # If no input device specified, try to find one with input channels
            if input_device is None:
                input_devices = [i for i, d in enumerate(devices) 
                               if d['max_input_channels'] > 0]
                
                if not input_devices:
                    raise RuntimeError("No input devices found!")
                    
                input_device = input_devices[0]
            
            self.input_device_id = input_device
            logger.info(f"Using audio device: {devices[input_device]['name']}")
            
        except Exception as e:
            logger.error(f"Error initializing audio: {e}")
            print("\nError: Could not initialize audio. Using default settings.")
            print("Audio recording may not work properly.\n")
            self.input_device_id = sd.default.device[0] if hasattr(sd.default, 'device') else 0

    def record_audio(self, duration=5, sample_rate=SAMPLE_RATE):
        """Record audio from the default microphone
        
        Args:
            duration (int): Duration of recording in seconds
            sample_rate (int): Sample rate for the recording
            
        Returns:
            tuple: (audio_data, sample_rate) if successful, (None, None) if failed
        """
        try:
            logger.info(f"Recording for {duration} seconds...")
            print("Speak now...")
            
            # Record audio
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32',
                device=self.input_device_id
            )
            sd.wait()  # Wait until recording is finished
            
            logger.info("Recording finished")
            
            # Save the recording
            saved_file = self.save_audio(audio_data, sample_rate)
            if saved_file:
                logger.info(f"Saved recording to {saved_file}")
            
            return audio_data.flatten(), sample_rate
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            print("Error recording audio. Make sure your microphone is properly connected and not being used by another application.\n")
            return None, None

    def save_audio(self, audio_data, sample_rate):
        """Save recorded audio to a file with timestamp
        
        Args:
            audio_data (numpy.ndarray): The audio data to save
            sample_rate (int): The sample rate of the audio data
            
        Returns:
            str: Path to the saved audio file, or None if saving failed
        """
        if audio_data is None:
            logger.warning("No audio data provided to save")
            return None
            
        try:
            # Create recordings directory if it doesn't exist
            os.makedirs('recordings', exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'recordings/recording_{timestamp}.wav'
            
            # Ensure audio_data is in the correct format
            if isinstance(audio_data, np.ndarray):
                # Convert to int16 if needed
                if audio_data.dtype != np.int16:
                    if np.issubdtype(audio_data.dtype, np.floating):
                        audio_data = (audio_data * 32767).astype(np.int16)
                    else:
                        audio_data = audio_data.astype(np.int16)
            
            # Save the audio file using scipy's wavfile.write
            wavfile.write(filename, sample_rate, audio_data)
            logger.info(f"Audio saved as {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return None

    def transcribe_audio(self, audio_data, sample_rate):
        """Convert speech in audio data to text
        
        Args:
            audio_data (numpy.ndarray): The audio data to transcribe
            sample_rate (int): The sample rate of the audio data
            
        Returns:
            str: The transcribed text, or None if transcription failed
        """
        try:
            # Convert numpy array to audio data
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Create an AudioData object for the recognizer
            audio_segment = sr.AudioData(
                audio_data.tobytes(),
                sample_rate=sample_rate,
                sample_width=audio_data.dtype.itemsize
            )
            
            # Try Malayalam first, then fall back to English
            try:
                text = self.recognizer.recognize_google(audio_segment, language='ml-IN')
                logger.info("Malayalam transcription successful")
                return text
            except sr.UnknownValueError:
                logger.info("Trying English transcription...")
                text = self.recognizer.recognize_google(audio_segment, language='en-US')
                logger.info("English transcription successful")
                return text
            
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return None
            
        except sr.RequestError as e:
            logger.error(f"Could not request results from Google Speech Recognition service; {e}")
            return None
            
        except Exception as e:
            logger.error(f"Error in transcribe_audio: {e}")
            return None

    def get_ollama_response(self, prompt):
        """Get response from Ollama's Mistral model"""
        try:
            logger.info(f"Sending prompt to Ollama: {prompt[:100]}...")
            
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
            return result.get('response', '').strip()
            
        except Exception as e:
            logger.error(f"Error getting response from Ollama: {e}")
            return "I'm sorry, I couldn't process your request at the moment."

    def speak(self, text, language='ml'):
        """Convert text to speech and play it using pygame
        
        Args:
            text (str): The text to speak
            language (str): Language code (default: 'ml' for Malayalam)
        """
        try:
            # Show the text that would be spoken
            print(f"\n[Assistant]: {text}")
            
            # Create a temporary file to store the speech
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_file = fp.name
            
            try:
                # Generate speech using gTTS
                tts = gTTS(text=text, lang=language, slow=False)
                tts.save(temp_file)
                
                # Initialize pygame mixer
                import pygame
                pygame.mixer.init()
                
                # Load and play the audio file
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                
                # Wait for the audio to finish playing
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Could not delete temporary file {temp_file}: {e}")
                
        except Exception as e:
            logger.error(f"Error in speak(): {e}")
            print("\n(Audio playback failed - showing text only)")

def main():
    """Main function to run the voice assistant"""
    print("\n=== Malayalam Voice Assistant ===")
    print("Press Enter to start recording (or 'q' to quit)\n")
    
    try:
        # Initialize the voice assistant
        assistant = VoiceAssistant()
        
        while True:
            try:
                user_input = input("Press Enter to start recording (or 'q' to quit)...")
                if user_input.lower() == 'q':
                    print("\nExiting...")
                    break
                    
                # Record audio
                print("\nRecording... (speak now)")
                audio_data, sample_rate = assistant.record_audio(duration=5)
                if audio_data is None:
                    print("No audio recorded. Please try again.")
                    continue
                    
                # Transcribe audio to text
                print("\nProcessing your request...")
                text = assistant.transcribe_audio(audio_data, sample_rate)
                if not text:
                    print("Could not transcribe audio. Please try again.")
                    continue
                    
                # Process the text (translate if needed)
                if not is_english(text):
                    # Translate to English for processing
                    english_text = translate_to_english(text)
                    print(f"\nYou said (translated to English): {english_text}")
                    
                    # Generate a meaningful response in English using the LLM
                    prompt = f"User: {english_text}\nAssistant:"
                    response = assistant.get_ollama_response(prompt)
                    
                    # Translate response to Malayalam
                    malayalam_response = translate_to_malayalam(response)
                    print(f"\nAssistant (Malayalam): {malayalam_response}")
                    
                    # Speak the response
                    assistant.speak(malayalam_response)
                    
                else:
                    print(f"\nYou said: {text}")
                    # Generate a meaningful response using the LLM
                    prompt = f"User: {text}\nAssistant:"
                    response = assistant.get_ollama_response(prompt)
                    print(f"\nAssistant: {response}")
                    assistant.speak(response)
                    
                print("\n------------------------------")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print(f"\nAn error occurred: {str(e)}\nPlease try again.")
                continue
                
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        print(f"\nA critical error occurred: {str(e)}")
    finally:
        print("\nThank you for using the Malayalam Voice Assistant!")

if __name__ == "__main__":
    main()
    exit(0)
