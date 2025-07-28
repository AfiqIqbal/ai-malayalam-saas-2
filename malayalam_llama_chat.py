import os
import sys
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MalayalamLLaMA:
    def __init__(self, model_path: str = None):
        """Initialize the Malayalam LLaMA model.
        
        Args:
            model_path: Path to the local GGUF model file. If None, it will be downloaded.
        """
        self.model = None
        self.model_path = model_path or os.path.join(
            os.path.expanduser("~"), 
            ".cache/huggingface/hub/models--abhinand--malayalam-llama-7b-instruct-v0.1-GGUF/snapshots",
            "malayalam-llama-7b-instruct-v0.1.Q4_K_M.gguf"
        )
        
    def load_model(self):
        """Load the LLaMA model."""
        try:
            from llama_cpp import Llama
            
            logger.info(f"Loading Malayalam LLaMA model from {self.model_path}...")
            
            # Initialize the model
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=2048,           # Context window size
                n_threads=4,          # Number of CPU threads to use
                n_gpu_layers=20,      # Number of layers to offload to GPU (-1 = all)
            )
            
            logger.info("Model loaded successfully!")
            return True
            
        except ImportError:
            logger.error("llama-cpp-python is not installed. Please install it with:")
            logger.error("pip install llama-cpp-python")
            return False
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("\nIf you don't have the model file, you can download it with:")
            logger.info("from huggingface_hub import hf_hub_download")
            logger.info("model_path = hf_hub_download(\
                repo_id='abhinand/malayalam-llama-7b-instruct-v0.1-GGUF',\
                filename='malayalam-llama-7b-instruct-v0.1.Q4_K_M.gguf'\
            )")
            return False
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate a response from the model.
        
        Args:
            prompt: The input prompt in Malayalam
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The generated response as a string
        """
        if not self.model:
            if not self.load_model():
                return "Error: Model could not be loaded."
        
        try:
            # Format the prompt for the instruction-tuned model
            formatted_prompt = f"""<s>[INST] <<SYS>>
നിങ്ങൾ ഒരു സഹായകനാണ്. എല്ലാ ചോദ്യങ്ങൾക്കും മലയാളത്തിൽ മാത്രം മറുപടി നൽകുക.
<</SYS>>

{prompt} [/INST]"""
            
            # Generate response
            response = self.model(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                echo=False,
                stop=["</s>", "[INST]", "\n"]
            )
            
            # Extract the generated text
            if 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0]['text'].strip()
            else:
                return "Error: No response generated."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

def main():
    """Main function to run the chat interface."""
    print("\n=== Malayalam LLaMA Chat ===")
    print("Type 'exit' to quit\n")
    
    # Initialize the model
    llama = MalayalamLLaMA()
    
    # Test if model loads
    if not llama.load_model():
        print("Failed to load the model. Please check the error messages above.")
        return
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou (Malayalam): ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nനന്ദി! വീണ്ടും സന്ദർശിക്കുക!")
                break
                
            if not user_input:
                continue
                
            # Get response from model
            print("\nAssistant (Malayalam): ", end='', flush=True)
            response = llama.generate_response(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            continue

if __name__ == "__main__":
    main()
