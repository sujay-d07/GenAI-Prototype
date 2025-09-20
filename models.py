import logging
from config import Config

# Updated imports for latest LangChain versions
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages AI models for embeddings and language generation"""
    
    def __init__(self):
        self.config = Config()
        self.config.validate_config()
        
        # Configure Google Generative AI
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        
        self.embeddings = None
        self.llm = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding and language models"""
        try:
            # Initialize embeddings with updated import
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.config.EMBEDDING_MODEL,
                google_api_key=self.config.GEMINI_API_KEY
            )
            
            # Initialize LLM with ChatGoogleGenerativeAI for better conversation handling
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.LLM_MODEL,
                google_api_key=self.config.GEMINI_API_KEY,
                temperature=0.1,
                max_output_tokens=2048,
                convert_system_message_to_human=True  # For better system prompt handling
            )
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def get_embeddings(self):
        """Get the embeddings model"""
        return self.embeddings
    
    def get_llm(self):
        """Get the language model"""
        return self.llm
    
    def test_models(self):
        """Test if models are working correctly"""
        try:
            # Test embeddings
            test_text = "This is a test document for legal analysis."
            embedding_result = self.embeddings.embed_query(test_text)
            
            if embedding_result and len(embedding_result) > 0:
                logger.info(f"Embeddings test passed - dimension: {len(embedding_result)}")
            else:
                raise Exception("Embeddings test failed")
            
            # Test LLM
            llm_result = self.llm.invoke("Say 'Hello, Legal AI is working!'")
            logger.info(f"LLM test passed - response: {llm_result.content[:100]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"Model testing failed: {e}")
            return False

# Global model manager instance
model_manager = None

def get_model_manager():
    """Get or create the global model manager instance"""
    global model_manager
    if model_manager is None:
        model_manager = ModelManager()
    return model_manager