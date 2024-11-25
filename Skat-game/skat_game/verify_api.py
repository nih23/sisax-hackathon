import os
from dotenv import load_dotenv
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_openai_api_key():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return False
        
    try:
        # Initialize client
        client = OpenAI(api_key=api_key)
        
        # Test API key with a simple request
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=5
        )
        
        logger.info("API key verified successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying API key: {e}")
        return False

if __name__ == "__main__":
    verify_openai_api_key()