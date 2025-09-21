import os
# interactive_legal_rag.py - Interactive Legal RAG with Categories and Comparison

import logging
import datetime
from pathlib import Path
from main_pipeline import LegalRAGPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractiveLegalRAG:
    def get_file_choice(self, prompt_message="Choose file"):
        """Prompt user to select a file from uploads folder."""
        files = self.find_legal_documents()
        if not files:
            print("❌ No files found in uploads folder.")
            return None
        print(f"\n{prompt_message}:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {os.path.basename(file)}")
        while True:
            choice = input(f"\nEnter file number (1-{len(files)}): ").strip()
            try:
                idx = int(choice)
                if 1 <= idx <= len(files):
                    return files[idx - 1]
            except ValueError:
                pass
            print("❌ Invalid choice. Please enter a valid number.")
    def __init__(self, uploads_folder="uploads", output_folder="responses"):
        """
        Initialize Enhanced Interactive Legal RAG Pipeline with Categories and Comparison
        
        Args:
            uploads_folder: Folder containing legal documents to process
            output_folder: Folder to save response files
        """
        self.uploads_folder = Path(uploads_folder)
        self.output_folder = Path(output_folder)
        self.pipeline = None
        self.session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_log = []

        # Create necessary directories
        self.uploads_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
        # Also ensure logs, category_stores, and vector_stores exist
        Path('logs').mkdir(exist_ok=True)
        Path('category_stores').mkdir(exist_ok=True)
        Path('vector_stores').mkdir(exist_ok=True)

        # Supported file extensions
        self.supported_extensions = {'.pdf', '.docx', '.txt', '.doc'}
        
    def find_legal_documents(self):
        """Find all legal documents in the uploads folder"""
        documents = []
        
        for file_path in self.uploads_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                documents.append(str(file_path))
        
        return documents
    
    def initialize_pipeline(self, documents):
        """Initialize the enhanced RAG pipeline with found documents"""
        try:
            print("🔧 Initializing Enhanced Legal RAG Pipeline with Categories...")
            self.pipeline = LegalRAGPipeline()
            
            print(f"📂 Processing {len(documents)} document(s)...")
            for doc in documents:
                print(f"   📄 {Path(doc).name}")
            
                # Process documents with categorization
                logger.info("Processing documents with categorization.")
            processing_result = self.pipeline.process_new_documents_with_categories(
                file_paths=documents, 
                store_prefix=f"legal_docs_{self.session_timestamp}"
            )
            
            print("✅ Enhanced pipeline initialized successfully!")
            print(f"   📊 Documents processed: {processing_result['documents_processed']}")
            print(f"   🧩 Chunks created: {processing_result['chunks_created']}")
            print(f"   🏷️  Categories found: {len(processing_result['categories_found'])}")
            print(f"   📂 Categories: {', '.join(processing_result['categories_found'])}")
            print(f"   💾 Store prefix: {processing_result['store_prefix']}")
            
            # Show category distribution
            print(f"\n📈 Category Distribution:")
            for category, count in processing_result['category_distribution'].items():
                category_name = self.pipeline.config.LEGAL_CATEGORIES.get(category, category)
                print(f"   • {category_name}: {count} documents")
            
            return True
            
        except Exception as e:
                logger.error(f"Error initializing pipeline: {str(e)}")
                return False
    
    def save_response_to_file(self, question, response, response_type="query"):
        """Save a question and response to a text file"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"{response_type}_{self.session_timestamp}.txt"
        filepath = self.output_folder / filename
        
        # Prepare content
        separator = "=" * 80
        content = f"\n{separator}\n"
        content += f"TIMESTAMP: {timestamp}\n"
        content += f"TYPE: {response_type.upper()}\n"
        
        # Add category information if available
        if 'category' in response:
            content += f"CATEGORY: {response['category']}\n"
        elif 'category1' in response and 'category2' in response:
            content += f"COMPARISON: {response['category1']} vs {response['category2']}\n"
        
        content += f"{separator}\n"
        content += f"QUESTION: {question}\n"
        content += f"{separator}\n"
        # Robustly extract answer for single or multi-category
        answer = response.get('answer') or response.get('content') or getattr(response, 'content', None)
        if not answer and 'category_results' in response:
            # Aggregate answers from all categories
            answers = []
            for cat, res in response['category_results'].items():
                if isinstance(res, dict) and 'answer' in res:
                    answers.append(f"[{cat.upper()}]:\n{res['answer']}\n")
                elif isinstance(res, dict) and 'error' in res:
                    answers.append(f"[{cat.upper()}]:\nError: {res['error']}\n")
            answer = '\n'.join(answers) if answers else 'No answers available.'
        if not answer:
            answer = str(response)
        content += f"ANSWER:\n{answer}\n"
        
        # Add sources information
        if response.get('sources'):
            if isinstance(response['sources'], dict):
                # Comparison response
                for cat, sources in response['sources'].items():
                    if cat != 'total' and isinstance(sources, list):
                        content += f"\nSOURCES FROM {cat.upper()} ({len(sources)}):\n"
                        for i, source in enumerate(sources, 1):
                            content += f"{i}. {source.get('document', 'Unknown')} (Page {source.get('page_number', 'N/A')})\n"
            else:
                # Regular response
                content += f"\nSOURCES ({len(response['sources'])}):\n"
                for i, source in enumerate(response['sources'], 1):
                    doc_name = source.get('document', 'Unknown') if isinstance(source, dict) else str(source)
                    content += f"{i}. {doc_name}\n"
        
        content += f"{separator}\n\n"
        
        # Append to file
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(content)
        
        # Log to conversation
        # Use the same robust answer extraction as above
        log_answer = None
        if 'answer' in response:
            log_answer = response['answer']
        elif 'category_results' in response:
            answers = []
            for cat, res in response['category_results'].items():
                if isinstance(res, dict) and 'answer' in res:
                    answers.append(f"[{cat.upper()}]:\n{res['answer']}\n")
                elif isinstance(res, dict) and 'error' in res:
                    answers.append(f"[{cat.upper()}]:\nError: {res['error']}\n")
            log_answer = '\n'.join(answers) if answers else 'No answers available.'
        else:
            log_answer = str(response)
        # Ensure answer is always a string for conversation log
        log_answer_str = log_answer
        if hasattr(log_answer_str, 'content'):
            log_answer_str = log_answer_str.content
        elif not isinstance(log_answer_str, str):
            log_answer_str = str(log_answer_str)
        self.conversation_log.append({
            'timestamp': timestamp,
            'question': question,
            'answer': log_answer_str,
            'sources': len(response.get('sources', [])),
            'type': response_type,
            'category': response.get('category', response.get('category1', 'N/A'))
        })
        
        return filepath
    
    def save_conversation_summary(self):
        """Save a summary of the entire conversation session"""
        filename = f"conversation_summary_{self.session_timestamp}.txt"
        filepath = self.output_folder / filename
        
        content = f"ENHANCED LEGAL RAG PIPELINE - CONVERSATION SUMMARY\n"
        content += f"Session: {self.session_timestamp}\n"
        content += f"Total Queries: {len(self.conversation_log)}\n"
        
        if self.pipeline:
            categories = self.pipeline.get_available_categories()
            content += f"Available Categories: {', '.join(categories)}\n"
        
        content += "=" * 60 + "\n\n"
        
        for i, entry in enumerate(self.conversation_log, 1):
            content += f"{i}. [{entry['timestamp']}] ({entry['type'].upper()})\n"
            content += f"   Category: {entry['category']}\n"
            content += f"   Q: {entry['question'][:100]}{'...' if len(entry['question']) > 100 else ''}\n"
            content += f"   A: {entry['answer'][:150]}{'...' if len(entry['answer']) > 150 else ''}\n"
            content += f"   Sources: {entry['sources']}\n\n"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath
    
    def display_menu(self):
        """Display the enhanced interactive menu"""
        print("\n" + "=" * 70)
        print("🏛️  ENHANCED INTERACTIVE LEGAL RAG PIPELINE")
        print("=" * 70)
        
        # Show available categories
        if self.pipeline:
            categories = self.pipeline.get_available_categories()
            if categories:
                print(f"📂 Available Categories: {', '.join(categories)}")
                print("-" * 70)
        
        print("1. 📋 Get Document Summary")
        print("2. ⚖️  Find Key Obligations")
        print("3. 🚪 Find Termination Clauses")
        print("4. ❓ Ask Custom Question")
        print("5. 🔄 Compare Documents")
        print("6. � Upload Documents to R2 Storage")
        print("7. 🌩️  Process Documents from R2")
        print("8. 📊 View Storage Information")
        print("9. �💬 View Conversation History")
        print("10. 📁 Show Response Files")
        print("11. 🏷️  Show Category Information")
        print("12. 📄 Reload Documents")
        print("13. ❌ Exit")
        print("=" * 70)
    
    def get_category_choice(self, prompt_message="Choose category"):
        """Get category choice from user"""
        if not self.pipeline:
            return None
        
        categories = self.pipeline.get_available_categories()
        if not categories:
            print("❌ No categories available.")
            return None
        
        print(f"\n{prompt_message}:")
        print("0. All categories")
        for i, category in enumerate(categories, 1):
            category_name = self.pipeline.config.LEGAL_CATEGORIES.get(category, category)
            print(f"{i}. {category_name} ({category})")
        
        try:
            choice = input(f"\nEnter choice (0-{len(categories)}): ").strip()
            
            if choice == "0":
                return None  # All categories
            
            choice_int = int(choice)
            if 1 <= choice_int <= len(categories):
                return categories[choice_int - 1]
            else:
                print("❌ Invalid choice.")
                return False
                
        except ValueError:
            print("❌ Please enter a valid number.")
            return False
    
    def get_two_category_choice(self):
        """Get two categories for comparison"""
        if not self.pipeline:
            return None, None
        
        categories = self.pipeline.get_available_categories()
        if len(categories) < 2:
            print("❌ At least 2 categories required for comparison.")
            return None, None
        
        print("\nSelect first category:")
        for i, category in enumerate(categories, 1):
            category_name = self.pipeline.config.LEGAL_CATEGORIES.get(category, category)
            print(f"{i}. {category_name} ({category})")
        
        try:
            choice1 = int(input(f"\nEnter first category (1-{len(categories)}): ").strip())
            if not (1 <= choice1 <= len(categories)):
                print("❌ Invalid choice for first category.")
                return None, None
            
            print("\nSelect second category:")
            for i, category in enumerate(categories, 1):
                if i == choice1:
                    continue
                category_name = self.pipeline.config.LEGAL_CATEGORIES.get(category, category)
                print(f"{i}. {category_name} ({category})")
            
            choice2 = int(input(f"\nEnter second category (1-{len(categories)}): ").strip())
            if not (1 <= choice2 <= len(categories)) or choice2 == choice1:
                print("❌ Invalid choice for second category.")
                return None, None
            
            return categories[choice1 - 1], categories[choice2 - 1]
            
        except ValueError:
            print("❌ Please enter valid numbers.")
            return None, None
    
    def handle_document_summary(self):
        """Handle document summary request for a specific file"""
        file_path = self.get_file_choice("📋 Select file for summary")
        if not file_path:
            return
        print(f"\n📋 Generating document summary for {os.path.basename(file_path)}...")
        response = self.pipeline.get_document_summary_by_file(file_path)
        print(f"\n📄 DOCUMENT SUMMARY - {os.path.basename(file_path)}:")
        print("-" * 50)
        answer = response.get('answer') or response.get('content') or getattr(response, 'content', None) or str(response)
        print(answer)
        response_type = f"summary_{os.path.basename(file_path)}"
        question = f"Generate a summary of {os.path.basename(file_path)}"
        filepath = self.save_response_to_file(question, response, response_type)
        print(f"\n💾 Response saved to: {filepath}")
        return response
    
    def handle_key_obligations(self):
        """Handle key obligations request for a specific file"""
        file_path = self.get_file_choice("⚖️  Select file for obligations")
        if not file_path:
            return
        print(f"\n⚖️  Finding key obligations for {os.path.basename(file_path)}...")
        response = self.pipeline.find_obligations_by_file(file_path)
        print(f"\n📜 KEY OBLIGATIONS - {os.path.basename(file_path)}:")
        print("-" * 50)
        answer = response.get('answer') or response.get('content') or getattr(response, 'content', None) or str(response)
        print(answer)
        response_type = f"obligations_{os.path.basename(file_path)}"
        question = f"Find key obligations in {os.path.basename(file_path)}"
        filepath = self.save_response_to_file(question, response, response_type)
        print(f"\n💾 Response saved to: {filepath}")
        return response
    
    def handle_termination_clauses(self):
        """Handle termination clauses request for a specific file"""
        file_path = self.get_file_choice("🚪 Select file for termination clauses")
        if not file_path:
            return
        print(f"\n🚪 Finding termination clauses for {os.path.basename(file_path)}...")
        response = self.pipeline.find_termination_clauses_by_file(file_path)
        print(f"\n📚 TERMINATION CLAUSES - {os.path.basename(file_path)}:")
        print("-" * 50)
        answer = response.get('answer') or response.get('content') or getattr(response, 'content', None) or str(response)
        print(answer)
        response_type = f"termination_{os.path.basename(file_path)}"
        question = f"Find termination clauses in {os.path.basename(file_path)}"
        filepath = self.save_response_to_file(question, response, response_type)
        print(f"\n💾 Response saved to: {filepath}")
        return response
    
    def handle_custom_question(self):
        """Handle custom question input for a specific file"""
        print("\n❓ Custom Question Mode")
        print("-" * 30)
        print("Enter your question about a specific legal document.")
        print("Type 'back' to return to main menu.")
        while True:
            question = input("\n🤔 Your question: ").strip()
            if question.lower() == 'back':
                break
            if not question:
                print("Please enter a valid question.")
                continue
            file_path = self.get_file_choice("❓ Select file for your question")
            if not file_path:
                continue
            print(f"\n🔍 Processing your question for {os.path.basename(file_path)}...")
            try:
                response = self.pipeline.query_documents_by_file(question, file_path)
                print(f"\n💡 ANSWER - {os.path.basename(file_path)}:")
                print("-" * 50)
                answer = response.get('answer') or response.get('content') or getattr(response, 'content', None) or str(response)
                print(answer)
                response_type = f"custom_{os.path.basename(file_path)}"
                filepath = self.save_response_to_file(question, response, response_type)
                print(f"\n💾 Response saved to: {filepath}")
                continue_asking = input("\n❓ Ask another question? (y/n): ").strip().lower()
                if continue_asking != 'y':
                    break
            except Exception as e:
                print(f"❌ Error processing question: {str(e)}")
    
    def handle_document_comparison(self):
        """Handle document comparison between two specific files"""
        print("\n🔄 Document Comparison Mode (By File)")
        print("-" * 35)

        # List available files in uploads/
        uploads_dir = self.pipeline.config.UPLOADS_DIR if hasattr(self.pipeline.config, 'UPLOADS_DIR') else 'uploads'
        try:
            files = [f for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]
        except Exception as e:
            print(f"❌ Could not list uploads directory: {e}")
            return
        if len(files) < 2:
            print("❌ At least 2 files required in uploads/ for comparison.")
            return

        print("\nAvailable files:")
        for idx, fname in enumerate(files, 1):
            print(f"{idx}. {fname}")

        try:
            idx1 = int(input(f"\nEnter number for first file (1-{len(files)}): ").strip())
            idx2 = int(input(f"Enter number for second file (1-{len(files)}): ").strip())
            if idx1 == idx2 or not (1 <= idx1 <= len(files)) or not (1 <= idx2 <= len(files)):
                print("❌ Invalid file selection.")
                return
            file1 = files[idx1 - 1]
            file2 = files[idx2 - 1]
        except Exception:
            print("❌ Please enter valid numbers.")
            return

        print(f"\n📊 Comparing: {file1} vs {file2}")
        print("\nComparison options:")
        print("1. 📋 Compare document summaries")
        print("2. ⚖️  Compare key obligations")
        print("3. 🚪 Compare termination clauses")
        print("4. ❓ Custom comparison question")

        try:
            comp_choice = input("\nEnter comparison type (1-4): ")

            if comp_choice == "1":
                question = "Compare the main content and key provisions of these documents."
                response = self.pipeline.compare_documents_by_file(question, os.path.join(uploads_dir, file1), os.path.join(uploads_dir, file2))
                response_type = "comparison_summary"
            elif comp_choice == "2":
                question = "Compare the key obligations in these documents."
                response = self.pipeline.compare_documents_by_file(question, os.path.join(uploads_dir, file1), os.path.join(uploads_dir, file2))
                response_type = "comparison_obligations"
            elif comp_choice == "3":
                question = "Compare the termination clauses in these documents."
                response = self.pipeline.compare_documents_by_file(question, os.path.join(uploads_dir, file1), os.path.join(uploads_dir, file2))
                response_type = "comparison_termination"
            elif comp_choice == "4":
                question = input("\n🤔 Enter your comparison question: ").strip()
                if not question:
                    print("❌ Please enter a valid question.")
                    return
                response = self.pipeline.compare_documents_by_file(question, os.path.join(uploads_dir, file1), os.path.join(uploads_dir, file2))
                response_type = "comparison_custom"
            else:
                print("❌ Invalid choice.")
                return

            print(f"\n🔍 COMPARISON RESULTS: {file1} vs {file2}")
            print("-" * 60)
            answer = response.get('answer') or response.get('content') or getattr(response, 'content', None) or str(response)
            print(answer)

            # Save to file
            filepath = self.save_response_to_file(question, response, f"{response_type}_{file1}_vs_{file2}")
            print(f"\n💾 Comparison saved to: {filepath}")

        except Exception as e:
            print(f"❌ Error during comparison: {str(e)}")
    
    def handle_r2_upload(self):
        """Handle uploading local documents to R2 storage"""
        if not self.pipeline or not self.pipeline.config.USE_R2_STORAGE:
            print("❌ R2 storage is not enabled or pipeline not initialized.")
            return
        
        print("\n📤 Upload Documents to R2 Storage")
        print("-" * 40)
        
        # List local documents
        local_docs = []
        for file_path in self.uploads_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                local_docs.append(str(file_path))
        
        if not local_docs:
            print("❌ No local documents found in uploads folder.")
            return
        
        print(f"Found {len(local_docs)} local documents:")
        for i, doc_path in enumerate(local_docs, 1):
            print(f"{i}. {Path(doc_path).name}")
        
        print("\nUpload options:")
        print("1. Upload all documents")
        print("2. Select specific documents")
        print("3. Cancel")
        
        try:
            choice = input("\nChoose option (1-3): ").strip()
            
            if choice == "1":
                # Upload all documents
                files_to_upload = local_docs
            elif choice == "2":
                # Select specific documents
                files_to_upload = []
                print("\nSelect documents to upload (enter numbers separated by commas):")
                selection = input("Example: 1,3,5 or 'all' for all: ").strip()
                
                if selection.lower() == 'all':
                    files_to_upload = local_docs
                else:
                    try:
                        indices = [int(x.strip()) - 1 for x in selection.split(',')]
                        files_to_upload = [local_docs[i] for i in indices if 0 <= i < len(local_docs)]
                        
                        if not files_to_upload:
                            print("❌ No valid files selected.")
                            return
                    except (ValueError, IndexError):
                        print("❌ Invalid selection format.")
                        return
            elif choice == "3":
                print("Upload cancelled.")
                return
            else:
                print("❌ Invalid choice.")
                return
            
            print(f"\n📤 Uploading {len(files_to_upload)} documents to R2...")
            
            # Upload documents
            upload_results = self.pipeline.upload_documents_to_r2(
                files_to_upload,
                add_timestamp=True  # Avoid filename conflicts
            )
            
            print(f"\n✅ Upload completed!")
            print(f"   Successfully uploaded: {upload_results['success_count']}")
            print(f"   Failed uploads: {upload_results['failure_count']}")
            
            if upload_results['successful_uploads']:
                print(f"\n📋 Successfully uploaded documents:")
                for upload in upload_results['successful_uploads']:
                    print(f"   • {upload['original_filename']} → {upload['r2_key']}")
            
            if upload_results['failed_uploads']:
                print(f"\n❌ Failed uploads:")
                for failure in upload_results['failed_uploads']:
                    print(f"   • {Path(failure['file_path']).name}: {failure['error']}")
            
            # Ask if user wants to process the uploaded documents
            if upload_results['successful_uploads']:
                process_choice = input(f"\n🤔 Process the uploaded documents now? (y/n): ").strip().lower()
                if process_choice == 'y':
                    r2_keys = [upload['r2_key'] for upload in upload_results['successful_uploads']]
                    print(f"\n🔄 Processing {len(r2_keys)} documents from R2...")
                    
                    try:
                        processing_result = self.pipeline.process_documents_from_r2(
                            r2_keys,
                            f"r2_upload_{self.session_timestamp}"
                        )
                        
                        print(f"✅ Processing completed!")
                        print(f"   Documents processed: {processing_result['documents_processed']}")
                        print(f"   Categories found: {', '.join(processing_result['categories_found'])}")
                        
                        # Update pipeline state
                        self.pipeline.pipeline_ready = True
                        
                    except Exception as e:
                        print(f"❌ Error processing uploaded documents: {str(e)}")
        
        except Exception as e:
            print(f"❌ Error during R2 upload: {str(e)}")
    
    def handle_r2_processing(self):
        """Handle processing documents directly from R2 storage"""
        if not self.pipeline or not self.pipeline.config.USE_R2_STORAGE:
            print("❌ R2 storage is not enabled or pipeline not initialized.")
            return
        
        print("\n🌩️  Process Documents from R2 Storage")
        print("-" * 45)
        
        try:
            # List available R2 documents
            available_docs = self.pipeline.list_available_documents()
            r2_docs = available_docs.get('r2_documents', [])
            
            if not r2_docs:
                print("❌ No documents found in R2 storage.")
                return
            
            print(f"Found {len(r2_docs)} documents in R2 storage:")
            for i, doc in enumerate(r2_docs, 1):
                size_mb = doc['size'] / (1024 * 1024)
                print(f"{i}. {doc['filename']} ({size_mb:.2f} MB)")
            
            print("\nProcessing options:")
            print("1. Process all R2 documents")
            print("2. Select specific documents")
            print("3. Cancel")
            
            choice = input("\nChoose option (1-3): ").strip()
            
            if choice == "1":
                # Process all documents
                r2_keys = [doc['r2_key'] for doc in r2_docs]
            elif choice == "2":
                # Select specific documents
                print("\nSelect documents to process (enter numbers separated by commas):")
                selection = input("Example: 1,3,5 or 'all' for all: ").strip()
                
                if selection.lower() == 'all':
                    r2_keys = [doc['r2_key'] for doc in r2_docs]
                else:
                    try:
                        indices = [int(x.strip()) - 1 for x in selection.split(',')]
                        r2_keys = [r2_docs[i]['r2_key'] for i in indices if 0 <= i < len(r2_docs)]
                        
                        if not r2_keys:
                            print("❌ No valid documents selected.")
                            return
                    except (ValueError, IndexError):
                        print("❌ Invalid selection format.")
                        return
            elif choice == "3":
                print("Processing cancelled.")
                return
            else:
                print("❌ Invalid choice.")
                return
            
            print(f"\n🔄 Processing {len(r2_keys)} documents from R2...")
            
            # Process documents from R2
            processing_result = self.pipeline.process_documents_from_r2(
                r2_keys,
                f"r2_process_{self.session_timestamp}"
            )
            
            print(f"\n✅ Processing completed!")
            print(f"   Documents processed: {processing_result['documents_processed']}")
            print(f"   Chunks created: {processing_result['chunks_created']}")
            print(f"   Categories found: {', '.join(processing_result['categories_found'])}")
            
            # Show category distribution
            print(f"\n📈 Category Distribution:")
            for category, count in processing_result['category_distribution'].items():
                category_name = self.pipeline.config.LEGAL_CATEGORIES.get(category, category)
                print(f"   • {category_name}: {count} documents")
            
            # Update pipeline state
            self.pipeline.pipeline_ready = True
            
        except Exception as e:
            print(f"❌ Error processing R2 documents: {str(e)}")
    
    def handle_storage_info(self):
        """Display comprehensive storage information"""
        print("\n📊 Storage Information")
        print("=" * 40)
        
        if not self.pipeline:
            print("❌ Pipeline not initialized.")
            return
        
        try:
            storage_info = self.pipeline.get_storage_information()
            
            print(f"🌩️  R2 Storage: {'Enabled' if storage_info['r2_enabled'] else 'Disabled'}")
            
            if storage_info['r2_enabled']:
                if 'r2_stats' in storage_info:
                    r2_stats = storage_info['r2_stats']
                    print(f"   📄 R2 Documents: {r2_stats.get('total_documents', 0)}")
                    print(f"   💾 R2 Storage Used: {r2_stats.get('total_size_mb', 0)} MB")
                    print(f"   🪣 R2 Bucket: {storage_info.get('r2_bucket', 'N/A')}")
                    
                    if r2_stats.get('file_types'):
                        print(f"   📁 File Types in R2:")
                        for ext, count in r2_stats['file_types'].items():
                            print(f"     • {ext}: {count} files")
                else:
                    print(f"   ⚠️  R2 Stats: {storage_info.get('r2_error', 'Unknown error')}")
            
            # Local storage info
            local_stats = storage_info.get('local_stats', {})
            print(f"\n💻 Local Storage:")
            print(f"   📄 Local Documents: {local_stats.get('total_files', 0)}")
            if local_stats.get('total_size'):
                local_size_mb = local_stats['total_size'] / (1024 * 1024)
                print(f"   💾 Local Storage Used: {local_size_mb:.2f} MB")
            print(f"   📂 Local Path: {storage_info.get('local_storage_path', 'uploads')}")
            
            # Pipeline status
            if hasattr(self.pipeline, 'pipeline_ready'):
                print(f"\n🚀 Pipeline Status:")
                print(f"   Ready: {self.pipeline.pipeline_ready}")
                if self.pipeline.pipeline_ready:
                    categories = self.pipeline.get_available_categories()
                    print(f"   Categories: {len(categories)} ({', '.join(categories)})")
            
        except Exception as e:
            print(f"❌ Error retrieving storage information: {str(e)}")
    
    def get_multiple_file_choice(self, prompt_message="Choose files"):
        """Prompt user to select multiple files from uploads folder."""
        files = self.find_legal_documents()
        if not files:
            print("❌ No files found in uploads folder.")
            return []
        
        print(f"\n{prompt_message}:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {os.path.basename(file)}")
        
        print("\nEnter file numbers separated by commas (e.g., 1,3,5) or 'all' for all files:")
        
        while True:
            choice = input("Selection: ").strip()
            
            if choice.lower() == 'all':
                return files
            
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                selected_files = []
                
                for idx in indices:
                    if 0 <= idx < len(files):
                        selected_files.append(files[idx])
                    else:
                        print(f"❌ Invalid file number: {idx + 1}")
                        return []
                
                if selected_files:
                    return selected_files
                else:
                    print("❌ No valid files selected.")
                    return []
                    
            except ValueError:
                print("❌ Invalid format. Please enter numbers separated by commas or 'all'.")
                continue
    
    def show_conversation_history(self):
        """Display conversation history"""
        print("\n💬 CONVERSATION HISTORY")
        print("=" * 60)
        
        if not self.conversation_log:
            print("No conversations yet.")
            return
        
        for i, entry in enumerate(self.conversation_log, 1):
            print(f"\n{i}. [{entry['timestamp']}] ({entry['type'].upper()})")
            print(f"   Category: {entry['category']}")
            print(f"   Q: {entry['question']}")
            ans = entry['answer']
            if hasattr(ans, 'content'):
                ans = ans.content
            elif not isinstance(ans, str):
                ans = str(ans)
            print(f"   A: {ans[:200]}{'...' if len(ans) > 200 else ''}")
            print(f"   📚 Sources: {entry['sources']}")
    
    def show_response_files(self):
        """Show all response files created in this session"""
        print("\n📁 RESPONSE FILES")
        print("-" * 30)
        
        session_files = list(self.output_folder.glob(f"*_{self.session_timestamp}.txt"))
        
        if not session_files:
            print("No response files created yet.")
            return
        
        for file_path in session_files:
            file_size = file_path.stat().st_size
            print(f"📄 {file_path.name} ({file_size} bytes)")
        
        print(f"\n📂 All files saved in: {self.output_folder}")
    
    def show_category_information(self):
        """Show detailed category information"""
        print("\n🏷️  CATEGORY INFORMATION")
        print("=" * 50)
        
        if not self.pipeline:
            print("Pipeline not initialized.")
            return
        
        try:
            category_info = self.pipeline.get_category_info()
            categories = self.pipeline.get_available_categories()
            
            print(f"📊 Total Categories: {len(categories)}")
            print(f"📄 Total Documents: {category_info.get('total_documents', 0)}")
            
            if category_info.get('category_details'):
                print(f"\n📂 Category Details:")
                for category, details in category_info['category_details'].items():
                    category_name = self.pipeline.config.LEGAL_CATEGORIES.get(category, category)
                    doc_count = details.get('document_count', 0)
                    print(f"   • {category_name}: {doc_count} documents")
            
            # Show categorizations if available
            categorizations = self.pipeline.get_categorizations()
            if categorizations:
                print(f"\n📈 Categorization Statistics:")
                category_stats = self.pipeline.document_processor.get_categories_summary(categorizations)
                if category_stats.get('category_distribution'):
                    for cat, stats in category_stats['category_distribution'].items():
                        category_name = self.pipeline.config.LEGAL_CATEGORIES.get(cat, cat)
                        print(f"   • {category_name}: {stats['count']} docs ({stats['percentage']}%)")
                
                avg_confidence = category_stats.get('average_confidence', 0)
                print(f"\n🎯 Average Categorization Confidence: {avg_confidence:.2%}")
                
        except Exception as e:
            print(f"❌ Error retrieving category information: {str(e)}")
    
    def run(self):
        """Run the enhanced interactive pipeline"""
        print("🏛️  ENHANCED LEGAL DOCUMENT AI ANALYZER")
        print("=" * 60)
        print("🆕 Features: Auto-categorization, Category-specific analysis, Document comparison")
        
        # Find documents in uploads folder
        documents = self.find_legal_documents()
        
        if not documents:
            print(f"❌ No legal documents found in '{self.uploads_folder}' folder!")
            print(f"📁 Please add PDF, DOCX, or TXT files to the '{self.uploads_folder}' folder.")
            print("Supported formats: .pdf, .docx, .txt, .doc")
            return
        
        # Initialize pipeline
        if not self.initialize_pipeline(documents):
            print("❌ Failed to initialize pipeline. Exiting.")
            return
        
        print("\n✅ Ready for enhanced interactive queries!")
        
        # Main interaction loop
        while True:
            self.display_menu()
            
            try:
                choice = input("\n🎯 Choose an option (1-13): ").strip()
                
                if choice == '1':
                    self.handle_document_summary()
                elif choice == '2':
                    self.handle_key_obligations()
                elif choice == '3':
                    self.handle_termination_clauses()
                elif choice == '4':
                    self.handle_custom_question()
                elif choice == '5':
                    self.handle_document_comparison()
                elif choice == '6':
                    self.handle_r2_upload()
                elif choice == '7':
                    self.handle_r2_processing()
                elif choice == '8':
                    self.handle_storage_info()
                elif choice == '9':
                    self.show_conversation_history()
                elif choice == '10':
                    self.show_response_files()
                elif choice == '11':
                    self.show_category_information()
                elif choice == '12':
                    # Reload documents
                    print("📄 Reloading documents...")
                    documents = self.find_legal_documents()
                    if documents and self.initialize_pipeline(documents):
                        print("✅ Documents reloaded successfully!")
                    else:
                        print("❌ Failed to reload documents.")
                elif choice == '13':
                    print("\n👋 Saving conversation summary...")
                    summary_file = self.save_conversation_summary()
                    print(f"💾 Conversation summary saved to: {summary_file}")
                    
                    # Export categorization report
                    if self.pipeline:
                        try:
                            report_file = self.pipeline.export_categorization_report()
                            print(f"📊 Categorization report saved to: {report_file}")
                        except Exception:
                            pass  # Report might not be available
                    
                    print("🙏 Thank you for using Enhanced Legal RAG Pipeline!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-13.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {str(e)}")
                logger.exception("Error in main loop")

def main():
    """Main function to run the enhanced interactive pipeline"""
    # You can customize these paths
    rag_system = InteractiveLegalRAG(
        uploads_folder="uploads",    # Folder containing your legal documents
        output_folder="responses"    # Folder to save responses
    )
    
    try:
        rag_system.run()
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        logger.exception("Fatal error in main")

if __name__ == "__main__":
    main()