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
    def __init__(self):
        """Initialize the Malayalam LLaMA model using ctransformers."""
        self.model = None
        # Using a smaller, more reliable model that supports Malayalam
        self.model_repo = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
        self.model_file = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        
    def load_model(self):
        """Load the LLaMA model using ctransformers."""
        try:
            from ctransformers import AutoModelForCausalLM
            
            logger.info(f"Loading Malayalam LLaMA model from {self.model_repo}...")
            
            # Initialize the model with ctransformers
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_repo,
                model_file=self.model_file,
                model_type="llama",
                gpu_layers=20,       # Number of layers to offload to GPU (-1 = all)
                context_length=2048,  # Context window size
                threads=4,           # Number of CPU threads to use
                batch_size=1,        # Batch size for generation
                max_new_tokens=512,  # Maximum number of new tokens to generate
                temperature=0.7,     # Temperature for sampling
                top_p=0.9,           # Nucleus sampling parameter
                top_k=50,            # Top-k sampling parameter
                repetition_penalty=1.1  # Penalty for repeating tokens
            )
            
            logger.info("Model loaded successfully!")
            return True
            
        except ImportError:
            logger.error("ctransformers is not installed. Please install it with:")
            logger.error("pip install ctransformers")
            return False
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("\nIf you encounter download issues, you can manually download the model with:")
            logger.info(f"from huggingface_hub import hf_hub_download")
            logger.info(f'model_path = hf_hub_download(repo_id="{self.model_repo}", filename="{self.model_file}")')
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
            # Using a bilingual prompt to ensure good Malayalam responses
            formatted_prompt = f"""<s>[INST] <<SYS>>
You are a helpful AI assistant that speaks both English and Malayalam fluently. 
Respond in Malayalam to questions in Malayalam, and in English to questions in English.
<</SYS>>

Please respond in Malayalam to this: {prompt} [/INST]"""
            
            # Generate response
            response = self.model(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                stop=["</s>", "[INST]", "\n"]
            )
            
            # Clean up the response
            if isinstance(response, str):
                return response.strip()
            elif isinstance(response, dict) and 'choices' in response and len(response['choices']) > 0:
                return response['choices'][0]['text'].strip()
            else:
                return "Error: Could not generate a valid response."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"

def main():
    """Main function to run the chat interface."""
    print("\n=== Malayalam LLaMA Chat (ctransformers) ===")
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
