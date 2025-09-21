import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Enhanced Legal RAG Pipeline with Document Categorization"""
    
    # Model Configuration
    EMBEDDING_MODEL = "models/text-embedding-004"
    LLM_MODEL = "gemini-2.0-flash-exp"
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Cloudflare R2 Configuration
    R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
    R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
    R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
    
    # Storage settings
    USE_R2_STORAGE = True  # Set to False to use local storage
    R2_DOCUMENTS_PREFIX = "documents/"  # Prefix for document storage in R2

    @classmethod
    def validate_config(cls):
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
        
        # Validate R2 configuration if enabled
        if cls.USE_R2_STORAGE:
            required_r2_vars = ['R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'R2_BUCKET_NAME', 'R2_ENDPOINT_URL']
            missing_vars = [var for var in required_r2_vars if not getattr(cls, var)]
            if missing_vars:
                raise ValueError(f"R2 storage is enabled but missing environment variables: {', '.join(missing_vars)}")
    
    # Text Splitting Parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval Parameters
    TOP_K = 5
    
    # Paths
    UPLOAD_FOLDER = "uploads"
    VECTOR_STORE_FOLDER = "vector_stores"
    CATEGORY_STORE_FOLDER = "category_stores"  # New folder for category-specific stores
    LOGS_FOLDER = "logs"
    
    # File Settings
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    # Document Categories - Main category keys with user-friendly names
    LEGAL_CATEGORIES = {
        'contract': 'Contract/Agreement',
        'policy': 'Policy Document',
        'terms_of_service': 'Terms of Service',
        'privacy_policy': 'Privacy Policy',
        'license': 'License Agreement',
        'employment': 'Employment Document',
        'financial': 'Financial Agreement',
        'legal_notice': 'Legal Notice',
        'regulatory': 'Regulatory Document',
        'other': 'Other Legal Document'
    }
    
    # Category descriptions for better LLM categorization
    CATEGORY_DESCRIPTIONS = {
        'contract': 'Contracts, agreements, service agreements, vendor contracts, partnership agreements, business contracts, procurement agreements',
        'policy': 'Company policies, internal policies, procedural documents, guidelines, operational policies, corporate governance documents',
        'terms_of_service': 'Terms of service, terms and conditions, user agreements, platform rules, service terms, website terms',
        'privacy_policy': 'Privacy policies, data protection policies, cookie policies, data handling agreements, GDPR compliance documents',
        'license': 'Software licenses, intellectual property licenses, usage licenses, distribution agreements, licensing terms',
        'employment': 'Employment contracts, job descriptions, HR documents, non-disclosure agreements, non-compete clauses, employee handbooks',
        'financial': 'Financial agreements, loan documents, investment agreements, payment terms, financial policies, banking agreements',
        'legal_notice': 'Legal notices, disclaimers, warnings, legal communications, liability notices, compliance notices',
        'regulatory': 'Regulatory documents, compliance documents, legal requirements, government regulations, industry standards',
        'other': 'Any other legal document that doesn\'t fit the above categories, miscellaneous legal documents'
    }
    
    # Extended keyword patterns for fallback categorization
    CATEGORY_KEYWORDS = {
        'contract': [
            'contract', 'agreement', 'service agreement', 'vendor', 'partnership', 
            'parties agree', 'hereby agree', 'this agreement', 'contractor', 'client',
            'service provider', 'terms of contract', 'contract terms', 'signing party',
            'agreement between', 'contract period', 'contract duration'
        ],
        'policy': [
            'policy', 'procedure', 'guideline', 'company policy', 'internal policy',
            'corporate policy', 'operational procedure', 'policy document', 'governance',
            'code of conduct', 'standards', 'best practices', 'protocol'
        ],
        'terms_of_service': [
            'terms of service', 'terms and conditions', 'user agreement', 'platform',
            'terms of use', 'service terms', 'user terms', 'website terms',
            'acceptable use', 'user responsibilities', 'platform rules'
        ],
        'privacy_policy': [
            'privacy policy', 'data protection', 'personal information', 'cookies',
            'data processing', 'privacy notice', 'data collection', 'personal data',
            'gdpr', 'data privacy', 'information security', 'data retention'
        ],
        'license': [
            'license', 'intellectual property', 'software license', 'usage rights',
            'licensing agreement', 'license terms', 'licensed software', 'copyright license',
            'patent license', 'trademark license', 'end user license', 'EULA'
        ],
        'employment': [
            'employment', 'employee', 'job description', 'non-disclosure', 'non-compete',
            'employment contract', 'employee agreement', 'work agreement', 'job offer',
            'employment terms', 'salary', 'compensation', 'benefits', 'NDA'
        ],
        'financial': [
            'financial', 'payment', 'loan', 'investment', 'money', 'financial agreement',
            'banking', 'credit', 'financing', 'payment terms', 'financial obligation',
            'monetary', 'fiscal', 'budget', 'funding', 'invoice'
        ],
        'legal_notice': [
            'notice', 'disclaimer', 'warning', 'legal notice', 'liability notice',
            'legal disclaimer', 'limitation of liability', 'legal warning',
            'compliance notice', 'regulatory notice'
        ],
        'regulatory': [
            'regulation', 'compliance', 'regulatory', 'government', 'legal requirement',
            'regulatory compliance', 'industry regulation', 'statutory requirement',
            'regulatory framework', 'compliance standard', 'legal standard'
        ]
    }
    
    # Categorization settings
    CATEGORIZATION_SETTINGS = {
        'use_cache': True,
        'cache_file': 'categorization_cache.json',
        'min_confidence_threshold': 0.3,
        'max_content_length': 3000,  # Max chars to send to LLM for categorization
        'fallback_enabled': True,
        'export_reports': True
    }
    
    # Vector store settings for categories
    CATEGORY_STORE_SETTINGS = {
        'similarity_threshold': 0.7,
        'max_retrievals_per_category': 5,
        'enable_cross_category_search': True,
        'store_metadata': True,
        'auto_save': True
    }
    
    # Comparison settings
    COMPARISON_SETTINGS = {
        'max_docs_per_category': 10,
        'include_confidence_scores': True,
        'detailed_analysis': True,
        'highlight_differences': True
    }
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Validate category definitions
        if not cls.LEGAL_CATEGORIES:
            raise ValueError("No legal categories defined")
        
        # Ensure all categories have descriptions
        for category in cls.LEGAL_CATEGORIES.keys():
            if category not in cls.CATEGORY_DESCRIPTIONS:
                raise ValueError(f"Missing description for category: {category}")
        
        # Create necessary directories
        directories = [
            cls.UPLOAD_FOLDER,
            cls.VECTOR_STORE_FOLDER,
            cls.CATEGORY_STORE_FOLDER,
            cls.LOGS_FOLDER
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Could not create directory {directory}: {e}")
        
        # Validate file settings
        if cls.MAX_FILE_SIZE <= 0:
            raise ValueError("MAX_FILE_SIZE must be positive")
        
        if not cls.ALLOWED_EXTENSIONS:
            raise ValueError("No allowed file extensions defined")
        
        # Validate model settings
        if cls.CHUNK_SIZE <= 0 or cls.CHUNK_OVERLAP < 0:
            raise ValueError("Invalid text splitting parameters")
        
        if cls.TOP_K <= 0:
            raise ValueError("TOP_K must be positive")
        
        return True
    
    @classmethod
    def get_category_info(cls, category: str = None):
        """Get information about categories"""
        if category:
            if category not in cls.LEGAL_CATEGORIES:
                return None
            
            return {
                'key': category,
                'name': cls.LEGAL_CATEGORIES[category],
                'description': cls.CATEGORY_DESCRIPTIONS.get(category, ''),
                'keywords': cls.CATEGORY_KEYWORDS.get(category, [])
            }
        else:
            return {
                'total_categories': len(cls.LEGAL_CATEGORIES),
                'categories': {
                    key: {
                        'name': name,
                        'description': cls.CATEGORY_DESCRIPTIONS.get(key, ''),
                        'keyword_count': len(cls.CATEGORY_KEYWORDS.get(key, []))
                    }
                    for key, name in cls.LEGAL_CATEGORIES.items()
                }
            }
    
    @classmethod
    def is_valid_category(cls, category: str):
        """Check if a category is valid"""
        return category in cls.LEGAL_CATEGORIES
    
    @classmethod
    def get_all_categories(cls):
        """Get list of all category keys"""
        return list(cls.LEGAL_CATEGORIES.keys())
    
    @classmethod
    def get_category_name(cls, category_key: str):
        """Get friendly name for a category key"""
        return cls.LEGAL_CATEGORIES.get(category_key, category_key)
    
    @classmethod
    def print_config_summary(cls):
        """Print a summary of the configuration"""
        print("🔧 Enhanced Legal RAG Pipeline Configuration")
        print("=" * 50)
        print(f"📁 Upload Folder: {cls.UPLOAD_FOLDER}")
        print(f"📁 Vector Store Folder: {cls.VECTOR_STORE_FOLDER}")
        print(f"📁 Category Store Folder: {cls.CATEGORY_STORE_FOLDER}")
        print(f"📁 Logs Folder: {cls.LOGS_FOLDER}")
        print(f"\n🤖 LLM Model: {cls.LLM_MODEL}")
        print(f"🔍 Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"📊 Chunk Size: {cls.CHUNK_SIZE}")
        print(f"🔄 Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"🎯 Top K Results: {cls.TOP_K}")
        print(f"\n📂 Supported Categories: {len(cls.LEGAL_CATEGORIES)}")
        
        for key, name in cls.LEGAL_CATEGORIES.items():
            print(f"   • {key}: {name}")
        
        print(f"\n📎 Supported File Types: {', '.join(cls.ALLOWED_EXTENSIONS)}")
        print(f"📏 Max File Size: {cls.MAX_FILE_SIZE // (1024*1024)} MB")
        
        api_key_status = "✅ Set" if cls.GEMINI_API_KEY else "❌ Missing"
        print(f"🔑 API Key Status: {api_key_status}")

# Example usage and testing
if __name__ == "__main__":
    try:
        # Validate configuration
        Config.validate_config()
        
        # Print configuration summary
        Config.print_config_summary()
        
        print("\n✅ Configuration validation successful!")
        
        # Test category functions
        print(f"\n📋 Category Info Test:")
        contract_info = Config.get_category_info('contract')
        print(f"Contract Category: {contract_info}")
        
        print(f"\nAll Categories: {Config.get_all_categories()}")
        print(f"Total Categories: {len(Config.get_all_categories())}")
        
        # Test validation functions
        print(f"\nValidation Tests:")
        print(f"'contract' is valid: {Config.is_valid_category('contract')}")
        print(f"'invalid_category' is valid: {Config.is_valid_category('invalid_category')}")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")