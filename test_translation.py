import sys
import os
import io
import sys
from pathlib import Path

# Set UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent / "backend"))

from ai.rag_pipeline import MalayalamRAGPipeline

def test_translation():
    print("Testing Malayalam Translation...")
    
    # Initialize the RAG pipeline
    print("Initializing RAG pipeline...")
    rag = MalayalamRAGPipeline()
    
    # Test with some sample English text
    test_phrases = [
        "Hello, how are you?",
        "What is your name?",
        "Tell me about Kerala",
        "How can I help you today?",
        "Thank you for your assistance"
    ]
    
    print("\nTesting English to Malayalam translation:")
    print("-" * 50)
    for phrase in test_phrases:
        print(f"\nEnglish: {phrase}")
        malayalam = rag.translate_to_malayalam(phrase)
        try:
            print(f"Malayalam: {malayalam}")
        except UnicodeEncodeError:
            # Fallback for environments that don't support Unicode output
            print("Malayalam: [Malayalam text output] (Unicode not fully supported in this terminal)")
    
    # Test generate_response method
    print("\nTesting generate_response method:")
    print("-" * 50)
    test_queries = [
        "What is the capital of Kerala?",
        "Tell me about Kerala's culture",
        "What are the main languages spoken in Kerala?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = rag.generate_response(query)
        try:
            print(f"Response: {response}")
        except UnicodeEncodeError:
            # Fallback for environments that don't support Unicode output
            print("Response: [Malayalam response] (Unicode not fully supported in this terminal)")
    
    print("\nTranslation testing completed!")

if __name__ == "__main__":
    test_translation()
