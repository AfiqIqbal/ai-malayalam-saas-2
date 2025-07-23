import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the service
from app.services.qa_service import qa_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('qa_test.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

async def test_qa_endpoint():
    """Test the QA endpoint with sample questions."""
    test_cases = [
        {
            "name": "English - Services Query",
            "question": "What services do you offer?",
            "language": "en"
        },
        {
            "name": "English - Password Reset",
            "question": "How can I reset my password?",
            "language": "en"
        },
        {
            "name": "Malayalam - Services Query",
            "question": "നിങ്ങൾ എന്തൊക്കെ സേവനങ്ങളാണ് നൽകുന്നത്?",
            "language": "ml"
        },
        {
            "name": "Malayalam - Password Reset",
            "question": "എന്റെ പാസ്‌വേഡ് റീസെറ്റ് ചെയ്യാൻ എങ്ങനെ?",
            "language": "ml"
        }
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_case in test_cases:
        total_tests += 1
        print(f"\n{'='*80}")
        print(f"TEST {total_tests}: {test_case['name']}")
        print(f"Question: {test_case['question']} ({test_case['language']})")
        print(f"{'='*80}")
        
        try:
            # Create a simple request dictionary
            request = {
                "question": test_case["question"],
                "language": test_case["language"],
                "top_k": 2
            }
            
            # Get the response
            response = await qa_service.get_answer(**request)
            
            # Convert response to dict if it's a Pydantic model
            if hasattr(response, 'dict'):
                response = response.dict()
            
            # Print the response
            print("\nRESPONSE:")
            print(f"Answer: {response.get('answer', 'No answer')}")
            print(f"Confidence: {response.get('confidence', 0):.4f}")
            
            # Print sources if available
            if 'sources' in response and response['sources']:
                print("\nSOURCES:")
                for i, src in enumerate(response['sources'], 1):
                    if hasattr(src, 'dict'):
                        src = src.dict()
                    print(f"{i}. [{src.get('score', 0):.4f}] {src.get('text', '')[:150]}...")
            
            passed_tests += 1
            print("✅ TEST PASSED")
            
        except Exception as e:
            print(f"❌ TEST FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests / total_tests * 100:.1f}%" if total_tests > 0 else "No tests run")

async def run_tests():
    """Run all test cases."""
    await test_qa_endpoint()

if __name__ == "__main__":
    asyncio.run(run_tests())
