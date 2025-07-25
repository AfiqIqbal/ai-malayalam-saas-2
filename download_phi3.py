import requests
import json
from loguru import logger

def download_phi3_model():
    """Download the Phi-3 Mini model using Ollama"""
    OLLAMA_BASE_URL = "http://localhost:11434/api"
    MODEL_NAME = "phi3:3.8b"
    
    try:
        print(f"\n=== Downloading {MODEL_NAME} Model ===")
        print("This may take a while depending on your internet connection\n")
        
        # Start the pull request
        response = requests.post(
            f"{OLLAMA_BASE_URL}/pull",
            json={"name": MODEL_NAME},
            stream=True
        )
        response.raise_for_status()
        
        # Stream the download progress
        for line in response.iter_lines():
            if line:
                try:
                    status = json.loads(line)
                    if 'status' in status:
                        print(f"Status: {status['status']}")
                    if 'completed' in status and 'total' in status:
                        progress = (status['completed'] / status['total']) * 100
                        print(f"Progress: {progress:.1f}% ({status['completed']}/{status['total']} bytes)")
                except json.JSONDecodeError:
                    continue
        
        print(f"\n{'-'*50}")
        print(f"Successfully downloaded {MODEL_NAME} model!")
        print("You can now run the assistant with:")
        print("python voice_assistant_phi3.py")
        print("-"*50 + "\n")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\nError: Failed to download the model: {str(e)}")
        print("Please make sure Ollama is running and you have an active internet connection.")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    download_phi3_model()
