from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
WANDB_API_KEY = os.getenv('WANDB_API_KEY')

# Validate required environment variables
def validate_env():
    required_vars = ['HUGGINGFACE_API_TOKEN', 'WANDB_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please check your .env file and ensure all required variables are set."
        )
