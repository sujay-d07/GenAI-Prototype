# Cloudflare R2 Storage Integration

This document explains how to use the enhanced Legal Document AI system with Cloudflare R2 storage instead of local file storage.

## 🌩️ Overview

The system has been upgraded to support Cloudflare R2 storage as the primary document storage backend. This provides several advantages:

- **Scalable Storage**: No local disk space limitations
- **Remote Access**: Documents accessible from anywhere
- **Backup & Redundancy**: Built-in data protection
- **Cost Effective**: Pay only for what you use
- **High Performance**: Fast global CDN access

## 📋 Prerequisites

1. **Cloudflare R2 Account**: You need a Cloudflare account with R2 storage enabled
2. **R2 Bucket**: Create a bucket in your Cloudflare R2 dashboard
3. **API Credentials**: Generate R2 API tokens with read/write access

## ⚙️ Configuration

### 1. Environment Variables

Add these variables to your `.env` file:

```env
# Existing Gemini API key
GEMINI_API_KEY=your_gemini_api_key

# Cloudflare R2 Configuration
R2_ACCESS_KEY_ID=your_r2_access_key_id
R2_SECRET_ACCESS_KEY=your_r2_secret_access_key
R2_BUCKET_NAME=your_bucket_name
R2_ENDPOINT_URL=https://your_account_id.r2.cloudflarestorage.com
```

### 2. Dependencies

The system automatically installs the required `boto3` package for R2 connectivity.

### 3. Enable R2 Storage

R2 storage is enabled by default in `config.py`. To disable it and use local storage:

```python
# In config.py
USE_R2_STORAGE = False  # Set to True to use R2
```

## 🚀 Usage

### Basic Document Upload

```python
from main_pipeline import LegalRAGPipeline

# Initialize pipeline
pipeline = LegalRAGPipeline()

# Upload local documents to R2
local_files = ['contract.pdf', 'policy.docx', 'agreement.txt']
upload_results = pipeline.upload_documents_to_r2(local_files)

print(f"Successfully uploaded: {upload_results['success_count']} documents")
for upload in upload_results['successful_uploads']:
    print(f"  • {upload['original_filename']} -> {upload['r2_key']}")
```

### Process Documents from R2

```python
# Get R2 keys from previous uploads
r2_keys = [upload['r2_key'] for upload in upload_results['successful_uploads']]

# Process documents directly from R2 storage
result = pipeline.process_documents_from_r2(r2_keys, "legal_docs_2024")

print(f"Processed {result['documents_processed']} documents")
print(f"Categories found: {result['categories_found']}")
```

### Hybrid Processing (Local + R2)

```python
# Mix of local files and R2 keys
mixed_sources = [
    "local_documents/new_contract.pdf",        # Local file
    "documents/existing_policy.pdf",           # R2 key
    "local_documents/updated_agreement.docx"   # Local file
]

# Process with automatic upload of local files to R2
result = pipeline.process_documents_hybrid(
    mixed_sources, 
    store_prefix="hybrid_legal_docs_2024",
    upload_local_to_r2=True  # Auto-upload local files
)

print(f"Hybrid processing complete: {result['source_type']}")
```

### Direct R2 Operations

```python
from r2_storage_manager import R2StorageManager

# Initialize R2 manager
storage = R2StorageManager()

# List all documents
documents = storage.list_documents()
print(f"Found {len(documents)} documents in R2")

# Upload a specific file
upload_result = storage.upload_file(
    "path/to/document.pdf",
    add_timestamp=True  # Avoids filename conflicts
)

# Download a document
download_result = storage.download_file(
    "documents/document_20240101_120000.pdf",
    "local_downloads/document.pdf"
)

# Check if file exists
exists = storage.file_exists("documents/some_document.pdf")

# Delete a document
deleted = storage.delete_file("documents/old_document.pdf")
```

## 📊 Storage Management

### View Storage Statistics

```python
# Get comprehensive storage information
storage_info = pipeline.get_storage_information()

print(f"R2 Storage Enabled: {storage_info['r2_enabled']}")
print(f"R2 Documents: {storage_info['r2_stats']['total_documents']}")
print(f"R2 Storage Used: {storage_info['r2_stats']['total_size_mb']} MB")
print(f"Local Documents: {storage_info['local_stats']['total_files']}")
```

### List Available Documents

```python
# List documents from both local and R2 storage
available_docs = pipeline.list_available_documents()

print(f"Local Documents: {available_docs['total_local']}")
print(f"R2 Documents: {available_docs['total_r2']}")

# Show sample documents
for doc in available_docs['r2_documents'][:5]:
    print(f"  R2: {doc['filename']} ({doc['size']} bytes)")

for doc in available_docs['local_documents'][:5]:
    print(f"  Local: {doc['filename']} ({doc['size']} bytes)")
```

## 🔄 Migration from Local Storage

### Option 1: Upload Existing Local Documents

```python
# Upload all local documents to R2
import os
from pathlib import Path

local_files = []
for file_path in Path("uploads").iterdir():
    if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.txt']:
        local_files.append(str(file_path))

# Upload to R2
upload_results = pipeline.upload_documents_to_r2(local_files)
print(f"Migrated {upload_results['success_count']} documents to R2")
```

### Option 2: Process with Auto-Upload

```python
# Process existing local files and automatically upload to R2
result = pipeline.process_new_documents_with_categories(
    file_paths=local_files,
    store_prefix="migrated_docs_2024",
    upload_local_to_r2=True  # Automatically upload to R2
)
```

## 🎯 Advanced Features

### Presigned URLs

```python
from r2_storage_manager import R2StorageManager

storage = R2StorageManager()

# Generate a presigned URL for download (valid for 1 hour)
download_url = storage.generate_presigned_url(
    "documents/contract.pdf",
    expiration=3600  # 1 hour
)

# Generate a presigned URL for upload
upload_url = storage.generate_presigned_url(
    "documents/new_document.pdf",
    expiration=1800,  # 30 minutes
    method='put_object'
)
```

### Custom Metadata

```python
# Upload with custom metadata
metadata = {
    'document_type': 'legal_contract',
    'department': 'legal',
    'confidentiality': 'high'
}

upload_result = storage.upload_file(
    "contract.pdf",
    metadata=metadata
)

# Retrieve metadata
file_metadata = storage.get_file_metadata("documents/contract.pdf")
print(f"Custom metadata: {file_metadata['metadata']}")
```

## 🧪 Testing

Run the R2 integration test to verify your setup:

```bash
python test_r2_integration.py
```

This will test:
- Configuration validation
- R2 connectivity
- Upload/download operations
- Document listing
- Pipeline integration

## 🔧 Configuration Options

### Document Storage Prefix

Documents are stored with a configurable prefix in your R2 bucket:

```python
# In config.py
R2_DOCUMENTS_PREFIX = "documents/"  # Default prefix
```

### Storage Settings

```python
# In config.py
USE_R2_STORAGE = True  # Enable R2 storage
R2_DOCUMENTS_PREFIX = "legal_docs/"  # Custom prefix
```

## 🛠️ Troubleshooting

### Connection Issues

1. **Verify Credentials**: Check your R2 access keys in `.env`
2. **Bucket Permissions**: Ensure your API token has read/write access
3. **Endpoint URL**: Verify your account-specific endpoint URL

### Common Errors

```python
# Test your configuration
from config import Config
config = Config()
config.validate_config()  # This will show any configuration issues
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Considerations

- **Batch Operations**: Upload multiple files in batches for better performance
- **Cleanup**: Set `cleanup_local=True` to remove temporary downloads
- **Caching**: Local temporary files are cached during processing
- **Timestamp**: Use `add_timestamp=True` to avoid filename conflicts

## 🔒 Security

- **Credentials**: Never commit `.env` file to version control
- **Metadata**: Avoid storing sensitive information in file metadata
- **Access Control**: Use R2 bucket policies for fine-grained access control
- **Presigned URLs**: Use short expiration times for presigned URLs

## 📚 API Reference

### R2StorageManager Methods

- `upload_file(local_path, r2_key=None, add_timestamp=False, metadata=None)`
- `upload_file_content(content, filename, add_timestamp=False, metadata=None)`
- `download_file(r2_key, local_path=None)`
- `download_file_content(r2_key)`
- `list_documents(prefix=None, max_keys=1000)`
- `delete_file(r2_key)`
- `file_exists(r2_key)`
- `get_file_metadata(r2_key)`
- `generate_presigned_url(r2_key, expiration=3600, method='get_object')`

### DocumentProcessor R2 Methods

- `upload_document_to_r2(local_path, add_timestamp=True, metadata=None)`
- `load_document_from_r2(r2_key, categorize=True, cleanup_local=True)`
- `load_multiple_documents_from_r2(r2_keys, categorize=True, cleanup_local=True)`
- `load_documents_hybrid(sources, categorize=True, upload_local_to_r2=False, cleanup_local=True)`
- `list_available_documents()`
- `get_storage_info()`

### LegalRAGPipeline R2 Methods

- `upload_documents_to_r2(local_files, add_timestamp=True)`
- `process_documents_from_r2(r2_keys, store_prefix=None)`
- `process_documents_hybrid(sources, store_prefix=None, upload_local_to_r2=False)`
- `list_available_documents()`
- `get_storage_information()`

---

## 🎉 Summary

Your Legal Document AI system now supports Cloudflare R2 storage, providing:

✅ **Scalable document storage in the cloud**  
✅ **Seamless integration with existing workflows**  
✅ **Hybrid processing (local + cloud)**  
✅ **Automatic document categorization**  
✅ **Advanced document analysis and comparison**  
✅ **Cost-effective storage solution**  

The system is backward compatible and can work with both local files and R2 storage simultaneously!