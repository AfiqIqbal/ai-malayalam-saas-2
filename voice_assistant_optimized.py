import os
import json
import queue
import threading
import time
import tempfile
import sounddevice as sd
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import speech_recognition as sr
import logging
from loguru import logger
import sys
from gtts import gTTS
import pygame
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dataclasses import dataclass, field
from typing import Optional
from rag_system import RAGSystem

# Thread pool for parallel tasks
executor = ThreadPoolExecutor(max_workers=5)

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
SILENCE_THRESHOLD = 0.005  # Lowered from 0.01 to be more sensitive to quiet audio
ENERGY_THRESHOLD = 100  # Lowered from 300 to be more sensitive to voice
PAUSE_THRESHOLD = 0.5  # Lowered from 0.8 to detect shorter pauses between phrases

# Initialize Google Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY in your .env file")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize pygame mixer for audio playback
pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=4096)
pygame.mixer.music.set_volume(1.0)

# RAG Configuration
RULES_FILE = "rules_knowledge_base.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight model for local use
SIMILARITY_THRESHOLD = 0.6  # Threshold for rule matching

@dataclass
class Rule:
    """Represents a rule in the knowledge base"""
    id: str
    title: str
    description: str
    content: str
    embedding: Optional[np.ndarray] = None

class LiveAudioProcessor:
    """Handles live audio recording and processing with optimizations."""
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.sample_rate = SAMPLE_RATE
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = PAUSE_THRESHOLD
        self.recognizer.energy_threshold = ENERGY_THRESHOLD
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.operation_timeout = 5
        self.recognizer.phrase_threshold = 0.3
        
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
            
    def close(self):
        """Clean up resources used by the audio processor."""
        try:
            # Stop any active listening
            self.is_listening = False
            
            # Clear the audio queue to release any waiting threads
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
                    
            # Clean up the recognizer if it exists
            if hasattr(self, 'recognizer') and self.recognizer is not None:
                # The SpeechRecognition library doesn't have a close method,
                # but we can clean up any resources it might be using
                if hasattr(self.recognizer, 'energy_threshold'):
                    self.recognizer.energy_threshold = 0  # Reset energy threshold
                
                # Clear any cached audio data
                if hasattr(self.recognizer, 'dynamic_energy_threshold'):
                    self.recognizer.dynamic_energy_threshold = False
            
            logger.debug("Audio processor resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up audio processor: {e}")
        finally:
            # Ensure we don't try to use this instance again
            self.is_listening = False

class GeminiAssistant:
    def __init__(self, rules_file="rules_knowledge_base.json"):
        # Initialize instance variables
        self.audio_processor = None
        self.conversation_history = []
        self.cleanup_tasks = set()
        self._running = False
        self.rag_system = RAGSystem(rules_file)
        self.max_conversation_history = 5  # Keep last 5 exchanges
        
        try:
            # Initialize pygame mixer first
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=4096)
            pygame.mixer.music.set_volume(1.0)
            
            # Initialize audio processor
            self.audio_processor = LiveAudioProcessor()
            
            # Initialize the model
            self.model = self._init_model()
            
            logger.info("Audio system and model initialized successfully")
            self._running = True
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            self.cleanup()
            raise
    
    async def _cleanup_temp_file(self, file_path, retry_count=5, delay=0.5):
        """Helper method to clean up temporary files with retries."""
        if not file_path or not os.path.exists(file_path):
            return
            
        for attempt in range(retry_count):
            try:
                # Force garbage collection to release file handles
                import gc
                gc.collect()
                
                # Try to close any open file handles
                try:
                    import win32file
                    handle = win32file.CreateFile(
                        file_path, 
                        win32file.GENERIC_WRITE,
                        0,  # No sharing
                        None,
                        win32file.OPEN_EXISTING,
                        win32file.FILE_ATTRIBUTE_NORMAL,
                        None
                    )
                    win32file.CloseHandle(handle)
                except Exception as win_err:
                    logger.debug(f"Could not close handle for {file_path}: {win_err}")
                
                # Try to remove the file
                os.unlink(file_path)
                logger.debug(f"Successfully cleaned up temp file: {file_path}")
                
                # If we got here, remove from cleanup tasks if it exists
                if file_path in self.cleanup_tasks:
                    self.cleanup_tasks.remove(file_path)
                return
                
            except (PermissionError, OSError) as e:
                if attempt == retry_count - 1:  # Last attempt
                    logger.warning(f"Final attempt failed to clean up {file_path}: {e}")
                    # Schedule for cleanup on program exit
                    self.cleanup_tasks.add(file_path)
                    # Try one last time with a longer delay
                    await asyncio.sleep(2)
                    try:
                        if os.path.exists(file_path):
                            os.unlink(file_path)
                            logger.debug(f"Successfully cleaned up temp file on final attempt: {file_path}")
                    except Exception as final_err:
                        logger.error(f"Failed final cleanup attempt for {file_path}: {final_err}")
                else:
                    # Exponential backoff with jitter
                    wait_time = delay * (2 ** attempt) + (random.random() * 0.1)
                    await asyncio.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"Unexpected error cleaning up {file_path}: {e}")
                if attempt == retry_count - 1:  # Last attempt
                    self.cleanup_tasks.add(file_path)
                await asyncio.sleep(delay * (attempt + 1))
        self.voice = "en-US-JennyNeural"  # Fallback voice that's widely available
        
    def _init_model(self):
        """Initialize the Gemini model with optimized settings."""
        return genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            },
            system_instruction="""
            You are a helpful, concise AI assistant that responds in the same language as the user.
            Keep responses brief and to the point.
            If the user greets you, respond with a short, friendly greeting in the same language.
            """
        )
    
    async def text_to_speech(self, text, language='en'):
        """Convert text to speech using gTTS with pygame for playback."""
        if not self._running:
            return
            
        temp_file = None
        try:
            # Determine language for gTTS
            tts_lang = 'ml' if language == 'ml' else 'en'
            
            # Create a temporary file for the speech
            temp_file = tempfile.mktemp(suffix='.mp3')
            
            # Generate speech with gTTS
            tts = gTTS(
                text=text,
                lang=tts_lang,
                slow=False,
                lang_check=False  # Disable language check for better compatibility
            )
            tts.save(temp_file)
            
            # Play the audio file
            try:
                # Stop any currently playing audio
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                
                # Load and play the new audio
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                
                # Wait for playback to finish with timeout
                start_time = time.time()
                while (self._running and 
                       pygame.mixer.music.get_busy() and 
                       (time.time() - start_time) < 30):  # 30s max
                    await asyncio.sleep(0.1)
                
                # Ensure pygame releases the file handle
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                
            except Exception as e:
                logger.error(f"Error playing audio: {e}")
                print("\n(Audio playback error, displaying text only)\n")
                
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            print("\n(Text-to-speech error, displaying text only)\n")
            
        finally:
            # Schedule cleanup of the temporary file
            if temp_file and os.path.exists(temp_file):
                # Add a small delay to ensure file handles are released
                await asyncio.sleep(0.5)
                # Create a task to clean up the file in the background
                asyncio.create_task(self._cleanup_temp_file(temp_file))

    async def get_gemini_response(self, text, language='en'):
        """Get response from Gemini with conversation context and RAG rules"""
        try:
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": text})
            
            # Get relevant rules and create enhanced prompt
            enhanced_prompt = self.rag_system.get_prompt_with_context(text)
            
            # Add conversation history to the prompt
            conversation_history = "\n".join(
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
                for msg in self.conversation_history[-self.max_conversation_history:]
            )
            
            full_prompt = f"""{enhanced_prompt}

Previous conversation:
{conversation_history}

Assistant:"""
            
            # Generate response using Gemini with RAG-enhanced prompt
            loop = asyncio.get_event_loop()
            
            # Define generation config
            gen_config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
            
            try:
                # Get the response as a stream
                response = await loop.run_in_executor(
                    executor,
                    lambda: self.model.generate_content(
                        full_prompt,
                        generation_config=gen_config,
                        stream=True
                    )
                )
                
                # Process the streaming response
                full_response = ""
                for chunk in response:
                    try:
                        # Try different ways to extract text from the chunk
                        if hasattr(chunk, 'text'):
                            chunk_text = chunk.text
                        elif hasattr(chunk, 'parts') and chunk.parts:
                            chunk_text = chunk.parts[0].text
                        elif hasattr(chunk, 'candidates') and chunk.candidates:
                            candidate = chunk.candidates[0]
                            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                chunk_text = candidate.content.parts[0].text
                            else:
                                chunk_text = str(chunk)
                        else:
                            chunk_text = str(chunk)
                        
                        if chunk_text.strip():
                            print(chunk_text, end="", flush=True)
                            full_response += chunk_text
                            
                            # Small delay to make the streaming effect visible
                            await asyncio.sleep(0.02)
                    except Exception as chunk_error:
                        logger.debug(f"Error processing chunk: {chunk_error}")
                        continue
                
                # If we didn't get any content, try to get the full response
                if not full_response.strip():
                    try:
                        if hasattr(response, 'resolve'):
                            resolved = response.resolve()
                            if hasattr(resolved, 'text'):
                                full_response = resolved.text
                            elif hasattr(resolved, 'candidates') and resolved.candidates:
                                candidate = resolved.candidates[0]
                                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                    full_response = candidate.content.parts[0].text
                                else:
                                    full_response = str(resolved)
                            else:
                                full_response = str(resolved)
                    except Exception as resolve_error:
                        logger.warning(f"Could not resolve response: {resolve_error}")
                
                # Clean up any markdown formatting if present
                if full_response:
                    full_response = full_response.replace('**', '').replace('*', '').strip()
                
            except Exception as e:
                logger.error(f"Error processing Gemini response: {e}")
                raise
            
            # Stream the response if we haven't already
            if not full_response:
                print("\nAssistant: ", end='', flush=True)
                full_response = "I'm sorry, I didn't get that. Could you please repeat?"
                if language == 'ml':
                    full_response = "ക്ഷമിക്കണം, എനിക്ക് മനസ്സിലായില്ല. ദയവായി വീണ്ടും പറയാമോ?"
                print(full_response)
            
            # Add assistant response to history if we got one
            if full_response:
                self.conversation_history.append({"role": "model", "content": full_response})
                # Convert response to speech
                await self.text_to_speech(full_response, language)
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error in Gemini API: {e}")
            error_msg = "I'm having trouble connecting to the AI service. Please try again."
            if language == 'ml':
                error_msg = "എനിക്ക് AI സേവനവുമായി ബന്ധിപ്പിക്കാൻ കഴിയുന്നില്ല. ദയവായി വീണ്ടും ശ്രമിക്കുക."
            print(f"\n{error_msg}")
            return error_msg
    
    async def start_listening(self):
        """Start the voice assistant with live audio processing."""
        print("\n" + "="*50)
        print("Gemini 2.5 Flash Voice Assistant (Optimized)")
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
                            # Process the audio chunk in a separate thread
                            text, language = self.audio_processor.process_audio_chunk(audio_data)
                            
                            if text:
                                print(f"\n\nYou: {text}")
                                print("\nAssistant: ", end='', flush=True)
                                
                                # Get response from Gemini
                                await self.get_gemini_response(text, language)
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

    def cleanup(self):
        """Clean up resources and temporary files."""
        self._running = False
        
        # Stop any playing audio and unload music
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.fadeout(500)  # Fade out audio
                pygame.time.delay(600)  # Wait for fadeout to complete
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                pygame.mixer.music.unload()
        except Exception as e:
            logger.error(f"Error stopping audio: {e}")
        
        # Clean up any remaining temporary files
        temp_files = list(self.cleanup_tasks)
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    # Try to force close any file handles
                    try:
                        import gc
                        gc.collect()
                    except:
                        pass
                    
                    # Try multiple times to delete the file
                    for _ in range(3):
                        try:
                            os.unlink(temp_file)
                            if temp_file in self.cleanup_tasks:
                                self.cleanup_tasks.remove(temp_file)
                            break
                        except (PermissionError, OSError):
                            pygame.time.delay(100)  # Small delay before retry
            except Exception as e:
                logger.warning(f"Could not clean up {temp_file}: {e}")
        
        # Audio processor cleanup (only if it has a close method)
        if hasattr(self, 'audio_processor') and self.audio_processor is not None:
            if hasattr(self.audio_processor, 'close') and callable(self.audio_processor.close):
                try:
                    self.audio_processor.close()
                except Exception as e:
                    logger.error(f"Error closing audio processor: {e}")

async def main():
    assistant = None
    try:
        assistant = GeminiAssistant()
        await assistant.start_listening()
        
    except KeyboardInterrupt:
        print("\n\nExiting...\n")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\nAn error occurred: {e}\n")
        
    finally:
        print("\nThank you for using the Gemini Voice Assistant!")
        
        # Clean up resources
        if assistant:
            try:
                assistant.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        
        # Ensure pygame is properly terminated
        try:
            pygame.mixer.quit()
            pygame.quit()
        except Exception as e:
            logger.error(f"Error quitting pygame: {e}")

if __name__ == "__main__":
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("# Google Gemini API Key\nGOOGLE_API_KEY=your_api_key_here\n")
        print("Created .env file. Please add your Google API key and restart the assistant.")
    else:
        asyncio.run(main())
