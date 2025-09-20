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
                documents, 
                f"legal_docs_{self.session_timestamp}"
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
        answer = None
        if 'answer' in response:
            answer = response['answer']
        elif 'category_results' in response:
            # Aggregate answers from all categories
            answers = []
            for cat, res in response['category_results'].items():
                if isinstance(res, dict) and 'answer' in res:
                    answers.append(f"[{cat.upper()}]:\n{res['answer']}\n")
                elif isinstance(res, dict) and 'error' in res:
                    answers.append(f"[{cat.upper()}]:\nError: {res['error']}\n")
            answer = '\n'.join(answers) if answers else 'No answers available.'
        else:
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
        self.conversation_log.append({
            'timestamp': timestamp,
            'question': question,
            'answer': log_answer,
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
        
        print("1. 📋 Get Document Summary (All or Specific Category)")
        print("2. ⚖️  Find Key Obligations (All or Specific Category)")
        print("3. 🚪 Find Termination Clauses (All or Specific Category)")
        print("4. ❓ Ask Custom Question (All or Specific Category)")
        print("5. 🔄 Compare Documents Between Categories")
        print("6. 💬 View Conversation History")
        print("7. 📁 Show Response Files")
        print("8. 🏷️  Show Category Information")
        print("9. 📄 Reload Documents")
        print("10. ❌ Exit")
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
        """Handle document summary request with category selection"""
        category = self.get_category_choice("📋 Select category for summary")
        
        if category is False:
            return
        
        print(f"\n📋 Generating document summary{f' for {category}' if category else ' (all categories)'}...")
        response = self.pipeline.get_document_summary(category)

        category_text = f" - {category.upper()}" if category else " - ALL CATEGORIES"
        print(f"\n📄 DOCUMENT SUMMARY{category_text}:")
        print("-" * 50)

        # Robustly extract answer for single or multi-category
        answer = None
        if 'answer' in response:
            answer = response['answer']
        elif 'category_results' in response:
            # Aggregate answers from all categories
            answers = []
            for cat, res in response['category_results'].items():
                if isinstance(res, dict) and 'answer' in res:
                    answers.append(f"[{cat.upper()}]:\n{res['answer']}\n")
                elif isinstance(res, dict) and 'error' in res:
                    answers.append(f"[{cat.upper()}]:\nError: {res['error']}\n")
            answer = '\n'.join(answers) if answers else 'No answers available.'
        else:
            answer = str(response)

        print(answer)

        # Save to file
        response_type = f"summary_{category}" if category else "summary_all"
        question = f"Generate a summary of {category if category else 'all'} documents"
        filepath = self.save_response_to_file(question, response, response_type)
        print(f"\n💾 Response saved to: {filepath}")

        return response
    
    def handle_key_obligations(self):
        """Handle key obligations request with category selection"""
        category = self.get_category_choice("⚖️  Select category for obligations")
        
        if category is False:
            return
        
        print(f"\n⚖️  Finding key obligations{f' for {category}' if category else ' (all categories)'}...")
        response = self.pipeline.find_key_obligations(category)
        
        category_text = f" - {category.upper()}" if category else " - ALL CATEGORIES"
        print(f"\n📜 KEY OBLIGATIONS{category_text}:")
        print("-" * 50)
        print(response['answer'])
        
        # Save to file
        response_type = f"obligations_{category}" if category else "obligations_all"
        question = f"Find key obligations in {category if category else 'all'} documents"
        filepath = self.save_response_to_file(question, response, response_type)
        print(f"\n💾 Response saved to: {filepath}")
        
        return response
    
    def handle_termination_clauses(self):
        """Handle termination clauses request with category selection"""
        category = self.get_category_choice("🚪 Select category for termination clauses")
        
        if category is False:
            return
        
        print(f"\n🚪 Finding termination clauses{f' for {category}' if category else ' (all categories)'}...")
        response = self.pipeline.find_termination_clauses(category)
        
        category_text = f" - {category.upper()}" if category else " - ALL CATEGORIES"
        print(f"\n📚 TERMINATION CLAUSES{category_text}:")
        print("-" * 50)
        print(response['answer'])
        
        # Save to file
        response_type = f"termination_{category}" if category else "termination_all"
        question = f"Find termination clauses in {category if category else 'all'} documents"
        filepath = self.save_response_to_file(question, response, response_type)
        print(f"\n💾 Response saved to: {filepath}")
        
        return response
    
    def handle_custom_question(self):
        """Handle custom question input with category selection"""
        print("\n❓ Custom Question Mode")
        print("-" * 30)
        print("Enter your question about the legal documents.")
        print("Type 'back' to return to main menu.")
        
        while True:
            question = input("\n🤔 Your question: ").strip()
            
            if question.lower() == 'back':
                break
            
            if not question:
                print("Please enter a valid question.")
                continue
            
            # Get category choice
            category = self.get_category_choice("❓ Select category for your question")
            if category is False:
                continue
            
            print(f"\n🔍 Processing your question{f' for {category}' if category else ' (all categories)'}...")
            try:
                response = self.pipeline.query_documents(question, category)
                
                category_text = f" - {category.upper()}" if category else " - ALL CATEGORIES"
                print(f"\n💡 ANSWER{category_text}:")
                print("-" * 50)
                print(response['answer'])
                
                if response.get('sources'):
                    sources_count = len(response['sources']) if isinstance(response['sources'], list) else response.get('total_sources', 0)
                    print(f"\n📚 Sources: {sources_count} document(s) referenced")
                
                # Save to file
                response_type = f"custom_{category}" if category else "custom_all"
                filepath = self.save_response_to_file(question, response, response_type)
                print(f"\n💾 Response saved to: {filepath}")
                
                # Ask if user wants to ask another question
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
            comp_choice = input("\nEnter comparison type (1-4): ").strip()
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
            print(response.get('answer', str(response)))

            # Save to file
            filepath = self.save_response_to_file(question, response, f"{response_type}_{file1}_vs_{file2}")
            print(f"\n💾 Comparison saved to: {filepath}")

        except Exception as e:
            print(f"❌ Error during comparison: {str(e)}")
    
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
            print(f"   A: {entry['answer'][:200]}{'...' if len(entry['answer']) > 200 else ''}")
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
                choice = input("\n🎯 Choose an option (1-10): ").strip()
                
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
                    self.show_conversation_history()
                elif choice == '7':
                    self.show_response_files()
                elif choice == '8':
                    self.show_category_information()
                elif choice == '9':
                    # Reload documents
                    print("📄 Reloading documents...")
                    documents = self.find_legal_documents()
                    if documents and self.initialize_pipeline(documents):
                        print("✅ Documents reloaded successfully!")
                    else:
                        print("❌ Failed to reload documents.")
                elif choice == '10':
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
                    print("❌ Invalid choice. Please select 1-10.")
                    
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