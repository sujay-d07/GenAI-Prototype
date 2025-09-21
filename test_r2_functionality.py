#!/usr/bin/env python3
"""
Quick test script to verify the new R2 upload functionality works correctly
"""

import os
import sys
from pathlib import Path

def test_new_functionality():
    """Test the newly added R2 functionality"""
    print("🧪 Testing New R2 Upload Functionality")
    print("=" * 50)
    
    # Test 1: Check if all required modules can be imported
    try:
        from r2_storage_manager import R2StorageManager
        from main_pipeline import LegalRAGPipeline
        from interactive_legal_rag import InteractiveLegalRAG
        print("✅ All modules import successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test 2: Check if R2 manager can be initialized
    try:
        storage = R2StorageManager()
        print("✅ R2StorageManager initializes successfully")
    except Exception as e:
        print(f"❌ R2 initialization error: {e}")
        return False
    
    # Test 3: Check if uploads directory exists and has files
    uploads_dir = Path("uploads")
    if uploads_dir.exists():
        files = list(uploads_dir.glob("*"))
        print(f"✅ uploads/ directory found with {len(files)} files")
        for file in files[:3]:
            print(f"   📄 {file.name}")
    else:
        print("❌ uploads/ directory not found")
    
    # Test 4: Check if R2 connection works
    try:
        documents = storage.list_documents()
        print(f"✅ R2 connection works - found {len(documents)} documents")
        for doc in documents[:3]:
            print(f"   📄 {doc['filename']}")
    except Exception as e:
        print(f"❌ R2 connection error: {e}")
        return False
    
    # Test 5: Check if interactive class has new methods
    try:
        interactive = InteractiveLegalRAG()
        
        # Check for new methods
        new_methods = [
            'handle_r2_upload',
            'handle_r2_processing',
            'handle_storage_info',
            'get_multiple_file_choice'
        ]
        
        missing_methods = []
        for method_name in new_methods:
            if not hasattr(interactive, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            print(f"❌ Missing methods: {missing_methods}")
            return False
        else:
            print("✅ All new methods are present in InteractiveLegalRAG")
            
    except Exception as e:
        print(f"❌ Interactive class test error: {e}")
        return False
    
    print("\n🎉 All tests passed! New R2 upload functionality is ready!")
    return True

def show_user_guide():
    """Show a quick user guide"""
    print("\n📖 Quick User Guide for R2 Upload:")
    print("=" * 40)
    print("1. Run: python interactive_legal_rag.py")
    print("2. Choose option 6: 📤 Upload Documents to R2 Storage")
    print("3. Select documents to upload")
    print("4. Optionally process uploaded documents")
    print("\nNew Menu Options:")
    print("  6. 📤 Upload Documents to R2 Storage")
    print("  7. 🌩️  Process Documents from R2")
    print("  8. 📊 View Storage Information")

if __name__ == "__main__":
    success = test_new_functionality()
    if success:
        show_user_guide()
    else:
        print("❌ Tests failed - check configuration and dependencies")
        sys.exit(1)