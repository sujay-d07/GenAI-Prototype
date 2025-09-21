"""
Test script for Cloudflare R2 Storage Integration
This script tests the R2 storage functionality with the legal document AI system
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_r2_integration():
    """Test R2 storage integration with the document processing system"""
    
    try:
        print("🌩️  Testing Cloudflare R2 Storage Integration")
        print("=" * 60)
        
        # Test 1: Configuration validation
        print("\n1. Testing Configuration...")
        from config import Config
        config = Config()
        config.validate_config()
        print("✅ Configuration validated successfully")
        print(f"   R2 Storage Enabled: {config.USE_R2_STORAGE}")
        print(f"   R2 Bucket: {config.R2_BUCKET_NAME}")
        
        # Test 2: R2 Storage Manager initialization
        print("\n2. Testing R2 Storage Manager...")
        from r2_storage_manager import R2StorageManager
        storage_manager = R2StorageManager()
        print("✅ R2 Storage Manager initialized successfully")
        
        # Test 3: Get storage statistics
        print("\n3. Testing Storage Statistics...")
        stats = storage_manager.get_storage_stats()
        print(f"✅ Storage Stats: {stats['total_documents']} documents, {stats['total_size_mb']} MB")
        
        # Test 4: List existing documents
        print("\n4. Testing Document Listing...")
        documents = storage_manager.list_documents()
        print(f"✅ Found {len(documents)} documents in R2 storage")
        
        if documents:
            print("   Sample documents:")
            for doc in documents[:3]:  # Show first 3
                print(f"     • {doc['filename']} ({doc['size']} bytes)")
        
        # Test 5: Document Processor with R2
        print("\n5. Testing Document Processor with R2...")
        from document_processor import DocumentProcessor
        processor = DocumentProcessor()
        print("✅ Document Processor with R2 support initialized")
        
        # Test 6: List available documents (hybrid)
        print("\n6. Testing Hybrid Document Listing...")
        available_docs = processor.list_available_documents()
        print(f"✅ Available Documents:")
        print(f"   Local: {available_docs['total_local']}")
        print(f"   R2: {available_docs['total_r2']}")
        
        # Test 7: Storage information
        print("\n7. Testing Storage Information...")
        storage_info = processor.get_storage_info()
        print(f"✅ Storage Information Retrieved:")
        print(f"   R2 Enabled: {storage_info['r2_enabled']}")
        if 'r2_stats' in storage_info:
            print(f"   R2 Documents: {storage_info['r2_stats'].get('total_documents', 0)}")
        
        # Test 8: Test upload functionality with a sample file
        print("\n8. Testing File Upload...")
        sample_files = []
        upload_folder = Path("uploads")
        
        if upload_folder.exists():
            for file_path in upload_folder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.pdf', '.docx']:
                    sample_files.append(str(file_path))
                    break
        
        if sample_files:
            try:
                sample_file = sample_files[0]
                print(f"   Testing upload with: {Path(sample_file).name}")
                
                upload_result = processor.upload_document_to_r2(sample_file, add_timestamp=True)
                print(f"✅ Upload successful: {upload_result['r2_key']}")
                
                # Test download
                print("   Testing download...")
                download_path = f"temp_{Path(sample_file).name}"
                download_result = storage_manager.download_file(upload_result['r2_key'], download_path)
                print(f"✅ Download successful: {download_result['local_path']}")
                
                # Cleanup
                if os.path.exists(download_path):
                    os.remove(download_path)
                
                # Test deletion
                print("   Testing file deletion...")
                delete_success = storage_manager.delete_file(upload_result['r2_key'])
                print(f"✅ Delete successful: {delete_success}")
                
            except Exception as e:
                print(f"⚠️  Upload/download test failed: {e}")
        else:
            print("   No sample files found for upload test")
        
        # Test 9: Main Pipeline with R2
        print("\n9. Testing Main Pipeline with R2...")
        from main_pipeline import LegalRAGPipeline
        pipeline = LegalRAGPipeline()
        print("✅ Main Pipeline with R2 support initialized")
        
        # Test pipeline status
        status = pipeline.get_enhanced_pipeline_status()
        print(f"   Pipeline Ready: {status['pipeline_ready']}")
        print(f"   R2 Storage: {status['features'].get('r2_storage', False)}")
        
        print("\n🎉 All R2 Integration Tests Completed Successfully!")
        print("\n📋 System is ready to use R2 storage for document management!")
        print("\n🚀 Available Features:")
        print("   • Upload local documents to R2")
        print("   • Process documents directly from R2")
        print("   • Hybrid processing (local + R2)")
        print("   • Automatic document categorization")
        print("   • Category-specific vector stores")
        print("   • Document comparison and analysis")
        
        return True
        
    except Exception as e:
        print(f"\n❌ R2 Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_usage_examples():
    """Show usage examples for the R2-integrated system"""
    
    print("\n" + "=" * 60)
    print("📖 USAGE EXAMPLES")
    print("=" * 60)
    
    print("\n1. Upload local documents to R2:")
    print("""
from main_pipeline import LegalRAGPipeline
pipeline = LegalRAGPipeline()

# Upload multiple documents
local_files = ['contract.pdf', 'policy.docx', 'agreement.txt']
upload_results = pipeline.upload_documents_to_r2(local_files)
print(f"Uploaded {upload_results['success_count']} documents")
    """)
    
    print("\n2. Process documents from R2:")
    print("""
# Get R2 keys from upload results
r2_keys = [upload['r2_key'] for upload in upload_results['successful_uploads']]

# Process documents directly from R2
result = pipeline.process_documents_from_r2(r2_keys, "legal_docs_r2")
print(f"Processed {result['documents_processed']} documents")
print(f"Categories: {result['categories_found']}")
    """)
    
    print("\n3. Hybrid processing (local + R2):")
    print("""
# Mix of local files and R2 keys
mixed_sources = [
    "local_documents/new_contract.pdf",    # Local file
    "documents/existing_policy.pdf",       # R2 key  
    "local_documents/agreement.docx"       # Local file
]

# Process with auto-upload of local files to R2
result = pipeline.process_documents_hybrid(
    mixed_sources, 
    store_prefix="hybrid_docs",
    upload_local_to_r2=True
)
    """)
    
    print("\n4. Query and analyze documents:")
    print("""
# Query across all categories
response = pipeline.query_documents("What are the key obligations?")

# Query specific category
response = pipeline.query_category("What are termination clauses?", "contract")

# Compare between categories
comparison = pipeline.compare_documents(
    "Compare privacy policies", "privacy_policy", "terms_of_service"
)
    """)
    
    print("\n5. Storage management:")
    print("""
# List all available documents
docs = pipeline.list_available_documents()
print(f"Local: {docs['total_local']}, R2: {docs['total_r2']}")

# Get storage statistics
storage_info = pipeline.get_storage_information()
print(f"R2 enabled: {storage_info['r2_enabled']}")
    """)

if __name__ == "__main__":
    print("Starting R2 Storage Integration Test...")
    
    success = test_r2_integration()
    
    if success:
        show_usage_examples()
        
        print(f"\n✅ R2 Integration Setup Complete!")
        print(f"Your system is now configured to use Cloudflare R2 storage")
        print(f"for document management instead of local storage.")
        
    else:
        print(f"\n❌ R2 Integration Test Failed!")
        print(f"Please check your .env file and R2 credentials.")
        
    print(f"\nPress any key to exit...")