# 📤 How to Upload Documents to Cloudflare R2 Storage

This guide explains all the ways users can upload documents to Cloudflare R2 storage in your Legal Document AI system.

## 🎯 Upload Methods Overview

### 1. **Interactive CLI Upload (Easiest)**
### 2. **Automatic Upload During Processing**
### 3. **Manual Python API Upload**
### 4. **Hybrid Processing with Auto-Upload**
### 5. **Direct R2 Storage Manager**

---

## Method 1: 📱 Interactive CLI Upload (Recommended for Users)

The easiest way for end users is through the interactive CLI:

### Step 1: Start the Interactive System
```bash
python interactive_legal_rag.py
```

### Step 2: Choose Upload Option
From the main menu, select:
```
6. 📤 Upload Documents to R2 Storage
```

### Step 3: Select Documents to Upload
The system will show all documents in your `uploads/` folder:
```
Found 3 local documents:
1. contract.pdf
2. policy.docx
3. agreement.txt

Upload options:
1. Upload all documents
2. Select specific documents
3. Cancel

Choose option (1-3): 2
```

### Step 4: Select Specific Files (if option 2)
```
Select documents to upload (enter numbers separated by commas):
Example: 1,3,5 or 'all' for all: 1,3

📤 Uploading 2 documents to R2...

✅ Upload completed!
   Successfully uploaded: 2
   Failed uploads: 0

📋 Successfully uploaded documents:
   • contract.pdf → documents/contract_20250921_162500.pdf
   • agreement.txt → documents/agreement_20250921_162500.txt

🤔 Process the uploaded documents now? (y/n): y
```

### Step 5: Optional Processing
The system can immediately process the uploaded documents with AI categorization.

---

## Method 2: 🔄 Automatic Upload During Processing

Documents are automatically uploaded to R2 when processing:

### Using Interactive CLI:
1. Place documents in `uploads/` folder
2. Run `python interactive_legal_rag.py`
3. The system will process documents locally first
4. Choose option: `6. 📤 Upload Documents to R2 Storage`

### Using Python API:
```python
from main_pipeline import LegalRAGPipeline

pipeline = LegalRAGPipeline()

# Process local documents and auto-upload to R2
result = pipeline.process_new_documents_with_categories(
    file_paths=['uploads/contract.pdf', 'uploads/policy.docx'],
    store_prefix='my_legal_docs_2024',
    upload_local_to_r2=True  # ← This uploads files automatically
)

print(f"Documents processed: {result['documents_processed']}")
print(f"Categories: {result['categories_found']}")
```

---

## Method 3: 🐍 Manual Python API Upload

For developers and advanced users:

### Direct Upload Method:
```python
from main_pipeline import LegalRAGPipeline

pipeline = LegalRAGPipeline()

# Upload specific documents
files_to_upload = [
    'uploads/contract.pdf',
    'uploads/privacy_policy.docx',
    'uploads/terms_of_service.txt'
]

upload_results = pipeline.upload_documents_to_r2(
    files_to_upload,
    add_timestamp=True  # Avoids filename conflicts
)

print(f"Successfully uploaded: {upload_results['success_count']}")
print(f"Failed uploads: {upload_results['failure_count']}")

# Get the R2 keys for processing
successful_uploads = upload_results['successful_uploads']
for upload in successful_uploads:
    print(f"✅ {upload['original_filename']} → {upload['r2_key']}")
```

### Process Uploaded Documents:
```python
# Extract R2 keys from successful uploads
r2_keys = [upload['r2_key'] for upload in upload_results['successful_uploads']]

# Process documents directly from R2
processing_result = pipeline.process_documents_from_r2(
    r2_keys,
    store_prefix='r2_legal_docs_2024'
)

print(f"Categories found: {processing_result['categories_found']}")
```

---

## Method 4: 🔀 Hybrid Processing with Auto-Upload

Mix local files and R2 documents in one workflow:

```python
from main_pipeline import LegalRAGPipeline

pipeline = LegalRAGPipeline()

# Mix of local files and existing R2 keys
mixed_sources = [
    'uploads/new_contract.pdf',           # Local file (will be uploaded)
    'uploads/updated_policy.docx',        # Local file (will be uploaded)
    'documents/existing_agreement.pdf'    # Already in R2 storage
]

# Process with automatic upload of local files
result = pipeline.process_documents_hybrid(
    file_sources=mixed_sources,
    store_prefix='hybrid_legal_docs_2024',
    upload_local_to_r2=True  # ← Auto-upload local files to R2
)

print(f"Source type: {result['source_type']}")  # Shows "hybrid"
print(f"Documents processed: {result['documents_processed']}")
```

---

## Method 5: 🔧 Direct R2 Storage Manager

For advanced users who need fine control:

```python
from r2_storage_manager import R2StorageManager

# Initialize R2 storage manager
storage = R2StorageManager()

# Upload a single file
upload_result = storage.upload_file(
    local_file_path='uploads/important_contract.pdf',
    add_timestamp=True,  # Adds timestamp to filename
    metadata={
        'document_type': 'contract',
        'importance': 'high',
        'department': 'legal'
    }
)

print(f"Uploaded to: {upload_result['r2_key']}")
print(f"File URL: {upload_result['url']}")

# Upload file content directly (from bytes)
with open('uploads/document.pdf', 'rb') as f:
    content = f.read()

upload_result = storage.upload_file_content(
    content=content,
    filename='document.pdf',
    add_timestamp=True
)

# List all documents in R2
documents = storage.list_documents()
for doc in documents:
    print(f"📄 {doc['filename']} ({doc['size']} bytes)")

# Check if file exists
exists = storage.file_exists('documents/contract.pdf')
print(f"File exists: {exists}")
```

---

## 📋 Interactive CLI Menu Options

When users run `python interactive_legal_rag.py`, they get these R2-related options:

```
======================================================================
🏛️  ENHANCED INTERACTIVE LEGAL RAG PIPELINE
======================================================================
📂 Available Categories: contract, policy, privacy_policy
----------------------------------------------------------------------
1. 📋 Get Document Summary
2. ⚖️  Find Key Obligations  
3. 🚪 Find Termination Clauses
4. ❓ Ask Custom Question
5. 🔄 Compare Documents
6. 📤 Upload Documents to R2 Storage         ← NEW!
7. 🌩️  Process Documents from R2            ← NEW!
8. 📊 View Storage Information               ← NEW!
9. 💬 View Conversation History
10. 📁 Show Response Files
11. 🏷️  Show Category Information
12. 📄 Reload Documents
13. ❌ Exit
======================================================================
```

### Option 6: 📤 Upload Documents to R2 Storage
- Lists all documents in `uploads/` folder
- Allows selection of specific files or all files
- Shows upload progress and results
- Optionally processes uploaded documents immediately

### Option 7: 🌩️ Process Documents from R2
- Lists all documents currently in R2 storage
- Allows selection of specific R2 documents to process
- Processes selected documents with AI categorization
- Creates category-specific vector stores

### Option 8: 📊 View Storage Information
- Shows R2 storage statistics (documents, size, file types)
- Shows local storage statistics
- Displays bucket information and configuration
- Shows pipeline status and available categories

---

## 🔍 User Workflow Examples

### Example 1: First-Time User Upload
```bash
# 1. User places documents in uploads/ folder
cp contract.pdf uploads/
cp policy.docx uploads/
cp agreement.txt uploads/

# 2. Start interactive system
python interactive_legal_rag.py

# 3. System processes local documents first (creates local pipeline)

# 4. User chooses option 6 to upload to R2
# 5. System uploads documents with timestamped names
# 6. User chooses to process uploaded documents
# 7. System creates R2-based pipeline with AI categorization
```

### Example 2: Processing Existing R2 Documents
```bash
# 1. User starts interactive system
python interactive_legal_rag.py

# 2. User chooses option 7 to process from R2
# 3. System shows available R2 documents
# 4. User selects documents to process
# 5. System processes with AI categorization
```

### Example 3: Mixed Local and R2 Workflow
```python
# Python API for mixed workflow
pipeline = LegalRAGPipeline()

# Check what's available
available = pipeline.list_available_documents()
print(f"Local: {available['total_local']}, R2: {available['total_r2']}")

# Process everything together
local_files = ['uploads/new_contract.pdf']
r2_keys = ['documents/existing_policy.pdf']

# Option A: Upload locals first, then process all from R2
upload_results = pipeline.upload_documents_to_r2(local_files)
all_r2_keys = r2_keys + [u['r2_key'] for u in upload_results['successful_uploads']]
result = pipeline.process_documents_from_r2(all_r2_keys)

# Option B: Hybrid processing (easier)
mixed_sources = local_files + r2_keys
result = pipeline.process_documents_hybrid(mixed_sources, upload_local_to_r2=True)
```

---

## 🎯 Summary: Upload Methods by User Type

### 👤 **End Users (Non-Technical)**
**Recommended: Interactive CLI (Method 1)**
1. Place documents in `uploads/` folder
2. Run `python interactive_legal_rag.py`
3. Choose option `6. 📤 Upload Documents to R2 Storage`
4. Select documents to upload
5. Optionally process immediately

### 👨‍💻 **Developers/Power Users**
**Recommended: Python API (Method 3)**
```python
# Simple upload and process
pipeline = LegalRAGPipeline()
upload_results = pipeline.upload_documents_to_r2(['file1.pdf', 'file2.docx'])
r2_keys = [u['r2_key'] for u in upload_results['successful_uploads']]
result = pipeline.process_documents_from_r2(r2_keys)
```

### 🔄 **Automated Workflows**
**Recommended: Hybrid Processing (Method 4)**
```python
# Automated pipeline with mixed sources
result = pipeline.process_documents_hybrid(
    file_sources=['local_file.pdf', 'documents/r2_file.pdf'],
    upload_local_to_r2=True
)
```

---

## ✅ Benefits of Each Method

| Method | Best For | Benefits |
|--------|----------|----------|
| Interactive CLI | End users, one-time uploads | Easy, guided, visual feedback |
| Auto-upload during processing | Simple workflows | Seamless, no extra steps |
| Manual Python API | Developers, batch operations | Full control, error handling |
| Hybrid processing | Mixed environments | Flexible, handles any combination |
| Direct R2 manager | Advanced use cases | Fine-grained control, metadata |

---

## 🔧 Configuration Requirements

Users need these in their `.env` file:

```env
# Required for all methods
GEMINI_API_KEY=your_gemini_api_key

# Required for R2 upload functionality  
R2_ACCESS_KEY_ID=your_r2_access_key_id
R2_SECRET_ACCESS_KEY=your_r2_secret_access_key
R2_BUCKET_NAME=your_bucket_name
R2_ENDPOINT_URL=https://your_account_id.r2.cloudflarestorage.com
```

The system will validate these credentials on startup and show clear error messages if anything is missing.

---

## 🎉 Ready to Use!

Your users now have multiple convenient ways to upload documents to Cloudflare R2 storage, from simple interactive menus to powerful Python APIs. The system handles all the complexity behind the scenes while providing a smooth user experience! 🚀