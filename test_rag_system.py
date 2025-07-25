import asyncio
import json
import os
from dotenv import load_dotenv
from voice_assistant_optimized import GeminiAssistant
from rag_system import RAGSystem, RULES_FILE

# Load environment variables
load_dotenv()

# Check if API key is set
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY in your .env file")

async def test_rag_system():
    # Initialize the RAG system with test rules
    test_rules = {
        "rules": [
            {
                "id": "test_rule_1",
                "title": "Greeting Response",
                "description": "How to respond to greetings",
                "content": "When the user greets you, respond with a friendly greeting and ask how you can help them today."
            },
            {
                "id": "test_rule_2",
                "malayalam_rule": True,
                "title": "Malayalam Response",
                "description": "Respond in Malayalam when asked in Malayalam",
                "content": "When the user speaks in Malayalam, respond in Malayalam. Keep responses natural and conversational."
            },
            {
                "id": "test_rule_3",
                "title": "Unknown Information",
                "description": "How to handle unknown information",
                "content": "If you don't know the answer to a question, say 'I'm sorry, I don't have that information. Could you try asking something else?'"
            }
        ]
    }
    
    # Save test rules
    with open(RULES_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_rules, f, indent=2)
    
    # Initialize the assistant with test rules
    print("Initializing Gemini Assistant with test rules...")
    assistant = GeminiAssistant()
    
    # Initialize the model
    try:
        assistant.model = assistant._init_model()
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        print("Please make sure you have set up your Google API key in the .env file")
        return
    
    # Test cases
    test_cases = [
        ("Hello!", "English greeting"),
        ("നമസ്കാരം", "Malayalam greeting"),
        ("What's the capital of France?", "General knowledge question"),
        ("What's the meaning of life?", "Philosophical question")
    ]
    
    # Run test cases
    for query, description in test_cases:
        print(f"\n{'='*50}")
        print(f"Test: {description}")
        print(f"Query: {query}")
        print("-" * 30)
        
        # Get response from assistant
        try:
            # Add language context to the prompt
            language = 'ml' if any(c > 'ऀ' for c in query) else 'en'
            response = await assistant.get_gemini_response(query, language=language)
        except Exception as e:
            print(f"Error getting response: {e}")
            response = "I'm having trouble connecting to the AI service. Please try again."
        
        print(f"\nResponse: {response}")
        
        # Simple verification
        if "hello" in query.lower() and "hello" in response.lower():
            print("✅ Greeting response verified")
        elif any(c > '\u0D00' for c in query) and any(c > '\u0D00' for c in response):
            # Check if response contains Malayalam characters
            print("✅ Malayalam response verified")
        elif "france" in query.lower() and "paris" in response.lower():
            print("✅ General knowledge verified")
        else:
            print("⚠️  Response verification inconclusive - please verify manually")
    
    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(test_rag_system())
