import time
import cProfile
import pstats
import io
from voice_assistant_nllb import VoiceAssistant, translate_to_english, translate_to_malayalam
import numpy as np

# Create a test audio clip (1 second of silence)
test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence at 16kHz

# Initialize the assistant
assistant = VoiceAssistant()

def profile_audio_processing():
    """Profile the audio processing pipeline"""
    print("\n=== Profiling Audio Processing ===")
    
    # Test recording (simulated with silence)
    start = time.time()
    audio_data, sr = assistant.record_audio(duration=2)
    print(f"Audio recording (2s): {time.time() - start:.2f}s")
    
    # Test audio saving
    start = time.time()
    saved_file = assistant.save_audio(test_audio, 16000)
    print(f"Audio saving: {time.time() - start:.2f}s")

def profile_translation():
    """Profile the translation pipeline"""
    print("\n=== Profiling Translation ===")
    
    # Test English to Malayalam
    test_text_en = "Hello, how are you today?"
    start = time.time()
    translated = translate_to_malayalam(test_text_en)
    print(f"EN->ML Translation: {time.time() - start:.2f}s")
    
    # Test Malayalam to English
    test_text_ml = "നിങ്ങൾക്ക് എങ്ങനെയുണ്ട്?"
    start = time.time()
    translated = translate_to_english(test_text_ml)
    print(f"ML->EN Translation: {time.time() - start:.2f}s")

def profile_llm():
    """Profile the LLM response generation"""
    print("\n=== Profiling LLM Response ===")
    
    # Test LLM response
    test_prompt = "What is the weather today?"
    start = time.time()
    response = assistant.get_ollama_response(test_prompt)
    print(f"LLM Response: {time.time() - start:.2f}s")

def main():
    print("=== Starting Performance Profiling ===")
    
    # Run each profile function with cProfile
    for func in [profile_audio_processing, profile_translation, profile_llm]:
        pr = cProfile.Profile()
        pr.enable()
        func()
        pr.disable()
        
        # Print profiling results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Show top 10 functions by cumulative time
        print("\nDetailed profiling:")
        print(s.getvalue())

if __name__ == "__main__":
    main()
