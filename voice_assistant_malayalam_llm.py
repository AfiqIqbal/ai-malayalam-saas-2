# Malayalam Voice Assistant using hf.co/abhinand/malayalam-llama-7b-instruct-v0.1-GGUF
import sys
import os
import json
import time
import hashlib
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
from pathlib import Path
from loguru import logger

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5  # Default recording duration
OLLAMA_BASE_URL = "http://localhost:11434/api"
# Using the exact model name from 'ollama list' command
MODEL_NAME = "hf.co/abhinand/malayalam-llama-7b-instruct-v0.1-GGUF:Q4_K_M"
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
        return " ".join([msg["content"] for msg in self.history[-3:]])
    
    def clear(self) -> None:
        """Clear the conversation history"""
        self.history = []

# Initialize global components
response_cache = ResponseCache(ttl_hours=CACHE_TTL_HOURS)
conversation = ConversationManager(max_history=MAX_HISTORY_LENGTH)

def is_malayalam(text):
    """Check if text contains Malayalam characters"""
    try:
        # Malayalam Unicode range: U+0D00 to U+0D7F
        return any(0x0D00 <= ord(char) <= 0x0D7F for char in text)
    except:
        return False

class VoiceAssistant:
    def __init__(self, input_device=None):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000  # Higher threshold to reduce background noise
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Shorter pause threshold
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.recording = False
        self.audio_data = None
        self.input_device_id = input_device or 0
        self.setup_audio_device()

    def setup_audio_device(self):
        """Configure audio input device"""
        try:
            # List all audio devices
            print("\n=== Available Audio Devices ===")
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                print(f"{i}: {device['name']} (Input Channels: {device['max_input_channels']})")
            print("="*30)
            
            # Try to find the best input device
            input_devices = [i for i, d in enumerate(devices) 
                           if d['max_input_channels'] > 0 and 
                           ('mic' in d['name'].lower() or 
                            'array' in d['name'].lower() or
                            'input' in d['name'].lower())]
            
            if not input_devices:
                logger.warning("No suitable input devices found. Using default.")
                self.input_device_id = 0
            else:
                # Try to find the most suitable device
                preferred_device = None
                
                # First try: Look for 'array' in name
                array_devices = [i for i in input_devices 
                               if 'array' in devices[i]['name'].lower()]
                if array_devices:
                    preferred_device = array_devices[0]
                
                # Second try: Look for 'realtek' in name (common on Windows)
                if preferred_device is None:
                    realtek_devices = [i for i in input_devices 
                                     if 'realtek' in devices[i]['name'].lower()]
                    if realtek_devices:
                        preferred_device = realtek_devices[0]
                
                # Third try: Just use the first input device
                if preferred_device is None:
                    preferred_device = input_devices[0]
                
                self.input_device_id = preferred_device
                
            # Set the default input device
            sd.default.device = self.input_device_id
            
            # Configure the recognizer
            self.recognizer.energy_threshold = 4000  # Higher threshold to reduce background noise
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8  # Shorter pause threshold
            
            logger.info(f"Using audio device: {devices[self.input_device_id]['name']}")
            logger.info(f"Audio settings - Sample rate: {SAMPLE_RATE}Hz, Channels: {CHANNELS}")
            
        except Exception as e:
            logger.error(f"Error setting up audio device: {e}")
            self.input_device_id = 0  # Fallback to default initialize audio. Using default settings.")
            self.input_device_id = sd.default.device[0] if hasattr(sd.default, 'device') else 0

    def record_audio(self, duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
        """Record audio from microphone"""
        self.recording = True
        self.audio_data = None
        
        try:
            logger.info(f"Recording for {duration} seconds...")
            print("Speak now...")
            
            # Configure audio settings
            sd.default.samplerate = sample_rate
            sd.default.channels = self.channels
            sd.default.dtype = 'float32'
            
            # Record audio with error handling
            try:
                audio_data = sd.rec(
                    int(duration * sample_rate),
                    samplerate=sample_rate,
                    channels=self.channels,
                    dtype='float32',
                    device=self.input_device_id
                )
                
                # Wait for recording to finish
                sd.wait()
                
                # Check if we got any audio data
                if audio_data.size == 0 or np.max(np.abs(audio_data)) < 0.01:  # Very quiet threshold
                    logger.warning("No audio detected or audio level too low")
                    return None, None
                    
                # Normalize audio to prevent clipping
                max_amplitude = np.max(np.abs(audio_data))
                if max_amplitude > 0:
                    audio_data = audio_data / (max_amplitude * 1.1)  # Leave some headroom
                
                self.audio_data = audio_data
                logger.info("Recording finished successfully")
                return audio_data, sample_rate
                
            except Exception as e:
                logger.error(f"Error during audio recording: {e}")
                return None, None
                
        except Exception as e:
            logger.error(f"Recording setup error: {e}")
            return None, None
            
        finally:
            self.recording = False

    def transcribe_audio(self, audio_data, sample_rate):
        """Transcribe audio to text using Google's speech recognition"""
        if audio_data is None:
            return None, None
            
        try:
            # Convert numpy array to audio data
            audio_int16 = (audio_data * 32767).astype(np.int16)
            audio_data = sr.AudioData(
                audio_int16.tobytes(),
                sample_rate=sample_rate,
                sample_width=audio_int16.dtype.itemsize,
            )
            
            # Only try Malayalam since we're using a Malayalam LLM
            text = self.recognizer.recognize_google(audio_data, language=LANGUAGE)
            logger.info(f"Transcribed (Malayalam): {text}")
            return text, 'ml'
                
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None, None
        except sr.RequestError as e:
            logger.error(f"Could not request results; {e}")
            return None, None
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None, None

    def get_ollama_response(self, prompt: str) -> str:
        """Get response from Ollama with caching"""
        try:
            print("\nSending request to Ollama...")  # Debug log
            
            # Check cache first
            cache_key = hashlib.md5(prompt.encode()).hexdigest()
            cached_response = response_cache.get(cache_key)
            if cached_response:
                logger.info("Using cached response")
                return cached_response
            
            # Prepare the request data with system prompt for better responses
            system_prompt = """
            നിങ്ങൾ ഒരു സഹായകനാണ്. എല്ലാ ചോദ്യങ്ങൾക്കും മലയാളത്തിൽ മാത്രം മറുപടി നൽകുക. 
            ചുരുക്കവും വ്യക്തവുമായ ഉത്തരങ്ങൾ നൽകുക.
            """
            
            # Format messages for the API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Prepare messages for the chat completion API
            # Extract system message if it exists
            system_msg = next((m['content'] for m in messages if m['role'] == 'system'), None)
            
            # Prepare messages in the format expected by the chat completion API
            chat_messages = []
            if system_msg:
                chat_messages.append({"role": "system", "content": system_msg})
            
            # Add user messages
            for msg in messages:
                if msg['role'] == 'user':
                    chat_messages.append({"role": "user", "content": msg['content']})
            
            logger.info(f"Sending chat messages: {json.dumps(chat_messages, ensure_ascii=False)[:300]}...")
            
            data = {
                "model": "hf.co/abhinand/malayalam-llama-7b-instruct-v0.1-GGUF:Q4_K_M",
                "messages": chat_messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 200  # Limit response length
                }
            }
            
            logger.info(f"Sending request to Ollama with model: {MODEL_NAME}")
            
            # Send request to Ollama chat completion endpoint
            response = requests.post(
                f"{OLLAMA_BASE_URL}/chat",  # Using chat completion endpoint
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=120  # Increased timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Full API response: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}...")
                
                # Extract response from chat completion format
                response_text = ""
                
                # Try to get the assistant's message content
                if "message" in result and isinstance(result["message"], dict):
                    response_text = result["message"].get("content", "").strip()
                
                # Fallback to other possible response formats
                if not response_text and "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if isinstance(choice, dict):
                        if "message" in choice and "content" in choice["message"]:
                            response_text = choice["message"]["content"].strip()
                        elif "text" in choice:
                            response_text = choice["text"].strip()
                
                # If we still don't have a response, try the raw response field
                if not response_text and "response" in result:
                    response_text = str(result["response"]).strip()
                
                logger.info(f"Extracted response text: {response_text[:200]}...")
                
                if response_text:
                    # Clean up the response
                    response_text = response_text.replace("</s>", "").replace("<s>", "").strip()
                    # Cache the response
                    response_cache.set(cache_key, response_text)
                    return response_text
                else:
                    logger.error(f"Empty or invalid response from Ollama. Full response: {json.dumps(result, indent=2, ensure_ascii=False)}")
                    return "ക്ഷമിക്കണം, എനിക്ക് ഒരു മികച്ച മറുപടി ലഭിച്ചില്ല. ദയവായി വീണ്ടും ശ്രമിക്കുക."  # Sorry, I couldn't get a good response. Please try again.
            else:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"ക്ഷമിക്കണം, ഒരു പിശക് സംഭവിച്ചു: {response.status_code}"  # Sorry, an error occurred
            
        except requests.exceptions.Timeout:
            logger.error("Request to Ollama timed out")
            return "ക്ഷമിക്കണം, പ്രതികരിക്കാൻ സമയമെടുത്തിരിക്കുന്നു. ദയവായി കുറച്ച് നിമിഷങ്ങൾക്ക് ശേഷം വീണ്ടും ശ്രമിക്കുക."  # Sorry, taking too long to respond. Please try again in a few minutes.
            
        except Exception as e:
            logger.error(f"Error in get_ollama_response: {str(e)}", exc_info=True)
            return "ക്ഷമിക്കണം, ഒരു പിശക് സംഭവിച്ചു. ദയവായി പിന്നീട് ശ്രമിക്കുക."  # Sorry, an error occurred. Please try again later.

    def process_query(self, text: str) -> str:
        """Process user query with conversation context"""
        try:
            print(f"\nProcessing query: {text}")  # Debug log
            
            # Add user message to history
            conversation.add_message("user", text)
            
            # Get conversation history in chat format
            messages = [
                {"role": "system", "content": "നിങ്ങൾ ഒരു സഹായകനാണ്. എല്ലാ ചോദ്യങ്ങൾക്കും മലയാളത്തിൽ മാത്രം മറുപടി നൽകുക."}
            ]
            
            # Add conversation history
            for msg in conversation.history[-5:]:  # Last 5 messages for context
                role = "user" if msg["role"].lower() == "user" else "assistant"
                messages.append({"role": role, "content": msg["content"]})
            
            # Log the request
            logger.info(f"Sending to LLM: {messages}")
            
            # Get response from LLM
            response = self.get_ollama_response(text)
            
            if not response:
                response = "ക്ഷമിക്കണം, എനിക്ക് ഒരു മികച്ച മറുപടി ലഭിച്ചില്ല."  # Sorry, I couldn't get a good response.
            
            # Add assistant response to history
            conversation.add_message("assistant", response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in process_query: {e}", exc_info=True)
            return "ക്ഷമിക്കണം, ഒരു പിശക് സംഭവിച്ചു. ദയവായി പിന്നീട് ശ്രമിക്കുക."  # Sorry, an error occurred. Please try again later.

    def text_to_speech(self, text: str, language: str = 'ml') -> None:
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
                    
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")

    def run(self):
        """Main loop for the voice assistant"""
        print("\n=== Malayalam Voice Assistant ===")
        print("Press Ctrl+C to exit\n")
        
        try:
            while True:
                try:
                    # Record audio
                    audio_data, sample_rate = self.record_audio(duration=RECORD_SECONDS)
                    if audio_data is None:
                        print("No audio detected. Please try again.")
                        continue
                    
                    # Transcribe audio to text
                    text, detected_lang = self.transcribe_audio(audio_data, sample_rate)
                    if not text:
                        print("Could not understand audio. Please try again.")
                        continue
                    
                    print(f"You ({detected_lang}): {text}")
                    
                    # Process query
                    response = self.process_query(text)
                    
                    # Print and speak the response
                    print(f"Assistant: {response}")
                    self.text_to_speech(response, language=detected_lang)
                    
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    print("An error occurred. Please try again.")
                    continue
                    
        except Exception as e:
            logger.critical(f"Fatal error: {e}")
            print("A fatal error occurred. The program will now exit.")
            sys.exit(1)

def main():
    """Main function to run the voice assistant"""
    try:
        # Initialize voice assistant
        assistant = VoiceAssistant()
        
        # Run the assistant
        assistant.run()
        
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        print("A fatal error occurred. The program will now exit.")
        sys.exit(1)

if __name__ == "__main__":
    main()
