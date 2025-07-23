import sys
from unittest.mock import MagicMock

# Create a mock aifc module
class MockAifc:
    class Aifc_read:
        pass
    class Aifc_write:
        pass

# Add the mock module to sys.modules
sys.modules['aifc'] = MockAifc()
