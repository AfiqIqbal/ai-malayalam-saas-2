import os
import json
import queue
import threading
import time
import sounddevice as sd
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import speech_recognition as sr
import logging
from loguru import logger
import sys
import tempfile
from gtts import gTTS
import pygame

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Load environment variables
load_dotenv()

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 5  # Process audio in 5-second chunks
# Audio configuration
SILENCE_THRESHOLD = 0.01  # Lower threshold for better sensitivity
ENERGY_THRESHOLD = 300  # Speech recognition energy threshold
PAUSE_THRESHOLD = 0.8  # Seconds of silence before considering speech complete

# Initialize Google Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY in your .env file")

genai.configure(api_key=GOOGLE_API_KEY)

# Use the main Gemini 2.5 Flash model
model = genai.GenerativeModel(
    'gemini-2.5-flash',
    generation_config={
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 4096,
    },
    system_instruction="""
    You are a helpful AI assistant that can communicate in both English and Malayalam.
    Respond in the same language as the user's input.
    Keep responses concise and natural.
    If the user greets you, respond with an appropriate greeting in the same language.
    """
)

class LiveAudioProcessor:
    """Handles live audio recording and processing."""
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.sample_rate = SAMPLE_RATE
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = PAUSE_THRESHOLD
        self.recognizer.energy_threshold = ENERGY_THRESHOLD
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.operation_timeout = 5  # Timeout for API requests
        
    def audio_callback(self, indata, frames, time, status):
        """Called for each audio block from the microphone."""
        if status:
            logger.warning(f"Audio status: {status}")
        if self.is_listening:
            self.audio_queue.put(indata.copy())

    def process_audio_chunk(self, audio_data):
        """Process a chunk of audio data and return transcribed text."""
        try:
            # Skip if audio is too quiet
            if np.abs(audio_data).mean() < SILENCE_THRESHOLD:
                return None, None
                
            # Convert numpy array to AudioData
            audio_data = (audio_data * 32767).astype(np.int16)
            audio_segment = sr.AudioData(
                audio_data.tobytes(),
                sample_rate=self.sample_rate,
                sample_width=audio_data.dtype.itemsize
            )
            
            # Try Malayalam first, then English
            try:
                # First try with Malayalam
                text = self.recognizer.recognize_google(
                    audio_segment,
                    language='ml-IN',  # Force Malayalam
                    show_all=False
                )
                logger.info(f"Malayalam transcription: {text}")
                return text, 'ml'
                
            except sr.UnknownValueError:
                try:
                    # If Malayalam fails, try English
                    text = self.recognizer.recognize_google(
                        audio_segment,
                        language='en-US',
                        show_all=False
                    )
                    logger.info(f"English transcription: {text}")
                    return text, 'en'
                    
                except sr.UnknownValueError:
                    logger.debug("Speech not recognized")
                    return None, None
                    
            except sr.RequestError as e:
                logger.error(f"Could not request results; {e}")
                return None, None
                    
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return None, None

class GeminiAssistant:
    def __init__(self):
        self.audio_processor = LiveAudioProcessor()
        self.conversation_history = []
        
    def text_to_speech(self, text, language='en'):
        """Convert text to speech and play it."""
        try:
            # Determine language for TTS
            tts_lang = 'ml' if language == 'ml' else 'en'
            
            # Create a temporary file for the speech
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_file = fp.name
            
            try:
                # Generate speech
                tts = gTTS(text=text, lang=tts_lang, slow=False)
                tts.save(temp_file)
                
                # Play the audio
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            print("\n(Text-to-speech error, displaying text only)\n")

    def get_gemini_response(self, text, language='en'):
        """Get response from Gemini with conversation context."""
        try:
            # Add language context to the prompt if it's Malayalam
            if language == 'ml':
                system_prompt = "You are a helpful assistant that responds in Malayalam. Keep your responses concise and natural in Malayalam."
                text = f"{system_prompt}\n\nUser: {text}"
            
            # Add user message to history
            self.conversation_history.append({"role": "user", "parts": [text]})
            
            # Keep only the last 10 messages for context
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # Generate response with appropriate settings for the language
            generation_config = {
                "temperature": 0.7 if language == 'en' else 0.3,  # Lower temperature for Malayalam
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
            
            response = model.generate_content(
                self.conversation_history,
                generation_config=generation_config,
                stream=True
            )
            
            # Stream the response
            full_response = ""
            print("\nAssistant: ", end='', flush=True)
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    print(chunk.text, end='', flush=True)
            
            # Add assistant response to history if we got one
            if full_response:
                self.conversation_history.append({"role": "model", "parts": [full_response]})
                # Convert response to speech
                self.text_to_speech(full_response, language)
            else:
                full_response = "I'm sorry, I didn't get that. Could you please repeat?"
                if language == 'ml':
                    full_response = "ക്ഷമിക്കണം, എനിക്ക് മനസ്സിലായില്ല. ദയവായി വീണ്ടും പറയാമോ?"
                print(full_response)
                self.text_to_speech(full_response, language)
            
            return full_response
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "I'm sorry, I encountered an error processing your request."
    
    def start_listening(self):
        """Start the voice assistant with live audio processing."""
        print("\n" + "="*50)
        print("Gemini 2.5 Flash-Lite Voice Assistant")
        print("="*50)
        print("\nListening for speech... (Press Ctrl+C to exit)")
        print("Speak clearly into your microphone after the beep...\n")
        print("-"*50 + "\n")
        
        # Initial beep to indicate readiness
        try:
            import winsound
            winsound.Beep(1000, 200)
        except:
            print("\a")  # Fallback beep
        
        self.audio_processor.is_listening = True
        
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                callback=self.audio_processor.audio_callback,
                blocksize=int(SAMPLE_RATE * CHUNK_DURATION)
            ):
                print("Listening... (speak now)")
                
                while True:
                    try:
                        # Get audio chunk from queue
                        audio_data = self.audio_processor.audio_queue.get(timeout=5)
                        if audio_data is not None:
                            # Process the audio chunk
                            text, language = self.audio_processor.process_audio_chunk(audio_data)
                            
                            if text:
                                print(f"\n\nYou: {text}")
                                print("\nAssistant: ", end='', flush=True)
                                
                                # Get response from Gemini
                                response = self.get_gemini_response(text, language)
                                print("\n" + "-"*50 + "\n")
                                
                    except queue.Empty:
                        continue
                    except KeyboardInterrupt:
                        print("\nExiting...")
                        break
                    except Exception as e:
                        logger.error(f"Error in main loop: {e}")
                        continue
                        
        except Exception as e:
            logger.critical(f"Fatal error: {e}")
        finally:
            self.audio_processor.is_listening = False
            print("\nThank you for using the Gemini Voice Assistant!")

if __name__ == "__main__":
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("# Google Gemini API Key\nGOOGLE_API_KEY=your_api_key_here\n")
        print("Created .env file. Please add your Google API key and restart the assistant.")
    else:
        assistant = GeminiAssistant()
        assistant.start_listening()
