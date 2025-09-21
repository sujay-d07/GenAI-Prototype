# GenAI-Prototype

A Command-Line Chatbot which helps in Demystifying Legal Documents with AI-powered analysis, categorization, and cloud storage support.

## 🌟 Features

- **🤖 AI-Powered Document Analysis**: Automated legal document categorization and analysis
- **🌩️ Cloud Storage Integration**: Cloudflare R2 storage support for scalable document management  
- **📂 Multi-Format Support**: PDF, DOCX, DOC, and TXT document processing
- **🏷️ Smart Categorization**: Automatic document categorization (contracts, policies, agreements, etc.)
- **🔍 Advanced RAG System**: Category-specific vector stores for precise document retrieval
- **⚖️ Document Comparison**: Compare documents within and across categories
- **💬 Interactive Analysis**: Ask questions about your legal documents
- **📊 Comprehensive Reports**: Generate detailed categorization and analysis reports

## 🚀 Quick Start

### 1. Installation

After cloning the repository, navigate to the project folder and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file with your API credentials:

```env
# Required: Gemini API Key
GEMINI_API_KEY=your_gemini_api_key

# Optional: Cloudflare R2 Storage (for cloud document storage)
R2_ACCESS_KEY_ID=your_r2_access_key_id
R2_SECRET_ACCESS_KEY=your_r2_secret_access_key
R2_BUCKET_NAME=your_bucket_name
R2_ENDPOINT_URL=https://your_account_id.r2.cloudflarestorage.com
```

### 3. Running the Application

#### Command-Line Interface
```bash
python interactive_legal_rag.py
```

#### Python API
```python
from main_pipeline import LegalRAGPipeline

# Initialize the enhanced pipeline
pipeline = LegalRAGPipeline()

# Process documents with categorization
result = pipeline.process_new_documents_with_categories(
    file_paths=['contract.pdf', 'policy.docx'], 
    store_prefix='legal_docs_2024'
)

# Query documents
response = pipeline.query_documents("What are the key obligations?")
print(response['answer'])
```

## 🌩️ Cloudflare R2 Storage

This system supports Cloudflare R2 cloud storage for scalable document management:

### Upload and Process from Cloud
```python
# Upload local documents to R2
upload_results = pipeline.upload_documents_to_r2(['contract.pdf', 'agreement.docx'])

# Process documents directly from R2
r2_keys = [upload['r2_key'] for upload in upload_results['successful_uploads']]
result = pipeline.process_documents_from_r2(r2_keys, 'cloud_docs_2024')
```

### Hybrid Processing
```python
# Process mix of local files and R2 documents
mixed_sources = [
    'local_documents/new_contract.pdf',    # Local file
    'documents/existing_policy.pdf',       # R2 key
    'local_documents/agreement.docx'       # Local file
]

result = pipeline.process_documents_hybrid(
    mixed_sources, 
    upload_local_to_r2=True  # Auto-upload local files to R2
)
```

📖 **[Complete R2 Storage Guide](R2_STORAGE_GUIDE.md)**

## 🏷️ Document Categories

The system automatically categorizes documents into:

- **Contracts** - Service agreements, vendor contracts, partnerships
- **Policies** - Company policies, procedures, guidelines  
- **Terms of Service** - User agreements, platform terms
- **Privacy Policies** - Data protection, privacy notices
- **Employment** - Job contracts, NDAs, employment terms
- **Financial** - Loan documents, payment agreements
- **Legal Notices** - Disclaimers, liability notices
- **Regulatory** - Compliance documents, regulations
- **Licenses** - Software licenses, IP agreements

## 🔧 Advanced Usage

### Category-Specific Analysis
```python
# Query within a specific category
response = pipeline.query_category("What are termination clauses?", "contract")

# Get category summary
summary = pipeline.get_document_summary("employment")

# Find obligations in contracts
obligations = pipeline.find_key_obligations("contract")
```

### Document Comparison
```python
# Compare documents between categories
comparison = pipeline.compare_documents(
    "Compare privacy policies", 
    "privacy_policy", 
    "terms_of_service"
)

# Compare specific aspects
termination_comparison = pipeline.compare_termination_clauses("contract", "employment")
```

### Storage Management
```python
# List all available documents
docs = pipeline.list_available_documents()
print(f"Local: {docs['total_local']}, R2: {docs['total_r2']}")

# Get storage information
storage_info = pipeline.get_storage_information()
```

## 🧪 Testing

Test the R2 integration and system functionality:

```bash
python test_r2_integration.py
```

## 📁 Project Structure

```
GenAI-Prototype/
├── main_pipeline.py           # Main pipeline with R2 integration
├── document_processor.py      # Document loading and R2 operations
├── r2_storage_manager.py      # Cloudflare R2 storage manager
├── document_categorizer.py    # AI-powered categorization
├── config.py                  # Configuration with R2 settings
├── interactive_legal_rag.py   # CLI interface
├── test_r2_integration.py     # R2 integration tests
├── R2_STORAGE_GUIDE.md        # Complete R2 documentation
├── requirements.txt           # Dependencies (includes boto3)
├── .env                       # Environment variables
└── uploads/                   # Local document storage
```

## 🛠️ Configuration

### Storage Options
```python
# In config.py
USE_R2_STORAGE = True          # Enable cloud storage
R2_DOCUMENTS_PREFIX = "documents/"  # R2 storage prefix
```

### Processing Settings
```python
CHUNK_SIZE = 1000              # Text chunk size
CHUNK_OVERLAP = 200            # Overlap between chunks
TOP_K = 5                      # Number of results to retrieve
```

## 🔒 Security

- Store API keys in `.env` file (never commit to version control)
- Use R2 bucket policies for access control
- Enable appropriate CORS settings for web access
- Use presigned URLs for temporary access

## 📊 System Status

Check system status and capabilities:

```python
status = pipeline.get_enhanced_pipeline_status()
print(f"Pipeline Ready: {status['pipeline_ready']}")
print(f"R2 Storage: {status['features']['r2_storage']}")
print(f"Available Categories: {len(status['available_categories'])}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

## 🆘 Support

For issues and questions:

1. Check the [R2 Storage Guide](R2_STORAGE_GUIDE.md) for cloud storage setup
2. Run `python test_r2_integration.py` to diagnose issues
3. Enable debug logging for detailed error information
4. Create an issue in the repository

---

**🎉 Ready to analyze legal documents with AI-powered categorization and cloud storage!**
