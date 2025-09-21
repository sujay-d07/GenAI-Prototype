# main_pipeline.py - Main Pipeline with Category Support

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Import all our custom modules
from config import Config
from models import get_model_manager
from document_processor import DocumentProcessor
from category_vector_store_manager import CategoryVectorStoreManager
from retrieval_chain import LegalDocumentAnalyzer
from document_categorizer import DocumentCategorizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_legal_rag_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LegalRAGPipeline:
    def _get_file_content(self, file_path: str) -> str:
        """Utility to load and return the full text content of a file."""
        from document_processor import DocumentProcessor
        processor = DocumentProcessor()
        docs, _ = processor.load_single_document(file_path, categorize=False)
        return "\n\n".join([doc.page_content for doc in docs]) if docs else ""
    def compare_documents_by_file(self, question: str, file_path1: str, file_path2: str) -> Dict[str, Any]:
        """Compare two specific documents by file path (not by category)"""
        # Load both documents (no categorization needed)
        docs1, _ = self.document_processor.load_single_document(file_path1, categorize=False)
        docs2, _ = self.document_processor.load_single_document(file_path2, categorize=False)
        if not docs1 or not docs2:
            raise ValueError("Both documents must have content for comparison.")
        # Use only the main text (concatenate all chunks for each doc)
        text1 = "\n\n".join([doc.page_content for doc in docs1])
        text2 = "\n\n".join([doc.page_content for doc in docs2])
        # Call analyzer for document-level comparison
        return self.analyzer.compare_documents_by_text(question, text1, text2, file1=os.path.basename(file_path1), file2=os.path.basename(file_path2))
    """
    Enhanced Legal Document RAG Pipeline with Document Categorization,
    Category-specific Vector Stores, and Document Comparison Features
    """
    
    def __init__(self):
        self.config = Config()
        self.config.validate_config()
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.categorizer = DocumentCategorizer()
        self.category_store_manager = CategoryVectorStoreManager()
        self.analyzer = LegalDocumentAnalyzer()
        
        # Pipeline state
        self.processed_documents = []
        self.categorizations = []
        self.current_store_prefix = None
        self.pipeline_ready = False
        self.available_categories = []
        
        logger.info("Enhanced Legal RAG Pipeline initialized successfully")
    
    def process_new_documents_with_categories(self, file_paths: List[str], 
                                            store_prefix: str = None) -> Dict[str, Any]:
        """
        Complete enhanced pipeline: Load documents -> Categorize -> Create category-specific stores -> Setup RAG
        """
        
        if not file_paths:
            raise ValueError("No file paths provided")
        
        # Generate store prefix if not provided
        if not store_prefix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            store_prefix = f"legal_docs_{timestamp}"
        
        try:
            logger.info(f"Starting enhanced document processing pipeline for {len(file_paths)} files")
            
            # Step 1: Load and process documents with categorization
            logger.info("Step 1: Loading, processing and categorizing documents...")
            documents, categorizations = self.document_processor.load_multiple_documents(
                file_paths, categorize=True
            )
            
            # Step 2: Split documents into chunks
            logger.info("Step 2: Splitting documents into chunks...")
            chunks = self.document_processor.split_documents(documents)
            
            # Step 3: Group documents by category
            logger.info("Step 3: Grouping documents by category...")
            categorized_docs = self.document_processor.group_documents_by_category(chunks)
            
            # Step 4: Create category-specific vector stores
            logger.info("Step 4: Creating category-specific vector stores...")
            store_creation_results = self.category_store_manager.create_category_stores(
                categorized_docs, store_prefix
            )
            
            # Step 5: Save category stores
            logger.info("Step 5: Saving category vector stores...")
            save_results = self.category_store_manager.save_category_stores()
            
            # Step 6: Setup enhanced RAG analyzer
            logger.info("Step 6: Setting up enhanced RAG analyzer...")
            self.analyzer.setup_with_category_stores(store_prefix)
            
            # Update pipeline state
            self.processed_documents = documents
            self.categorizations = categorizations
            self.current_store_prefix = store_prefix
            self.available_categories = list(categorized_docs.keys())
            self.pipeline_ready = True
            
            # Generate comprehensive statistics
            doc_stats = self.document_processor.get_document_stats(chunks)
            category_stats = self.document_processor.get_categories_summary(categorizations)
            store_info = self.category_store_manager.get_category_info()
            
            result = {
                "success": True,
                "store_prefix": store_prefix,
                "documents_processed": len(documents),
                "chunks_created": len(chunks),
                "categories_found": list(categorized_docs.keys()),
                "categorizations": categorizations,
                "category_distribution": {
                    cat: len(docs) for cat, docs in categorized_docs.items()
                },
                "store_creation_results": store_creation_results,
                "store_save_results": save_results,
                "document_stats": doc_stats,
                "category_stats": category_stats,
                "store_info": store_info,
                "processing_timestamp": datetime.now().isoformat(),
                "features_enabled": {
                    "categorization": True,
                    "category_specific_stores": True,
                    "document_comparison": True
                }
            }
            
            logger.info(f"Enhanced pipeline processing completed successfully: {store_prefix}")
            logger.info(f"Categories found: {list(categorized_docs.keys())}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced document processing pipeline: {e}")
            raise
    
    def load_existing_category_stores(self, store_prefix: str) -> Dict[str, Any]:
        """Load documents from existing category-specific vector stores"""
        
        try:
            logger.info(f"Loading existing category stores with prefix: {store_prefix}")
            
            # Load category stores
            load_results = self.category_store_manager.load_category_stores(store_prefix)
            
            if not any(load_results.values()):
                raise ValueError(f"No category stores found with prefix: {store_prefix}")
            
            # Setup analyzer with loaded stores
            self.analyzer.setup_with_category_stores(store_prefix)
            
            # Update pipeline state
            self.current_store_prefix = store_prefix
            self.available_categories = [cat for cat, success in load_results.items() if success]
            self.pipeline_ready = True
            
            store_info = self.category_store_manager.get_category_info()
            
            result = {
                "success": True,
                "store_prefix": store_prefix,
                "loaded_categories": self.available_categories,
                "load_results": load_results,
                "store_info": store_info,
                "load_timestamp": datetime.now().isoformat(),
                "features_available": {
                    "category_queries": True,
                    "cross_category_queries": True,
                    "document_comparison": len(self.available_categories) >= 2
                }
            }
            
            logger.info(f"Successfully loaded category stores: {self.available_categories}")
            return result
            
        except Exception as e:
            logger.error(f"Error loading existing category stores: {e}")
            raise
    
    def query_documents(self, question: str, category: str = None) -> Dict[str, Any]:
        """Query documents, optionally within a specific category"""
        
        if not self.pipeline_ready:
            raise ValueError("Pipeline not ready. Process documents or load existing stores first.")
        
        try:
            response = self.analyzer.ask_question(question, category)
            
            # Add pipeline metadata
            response.update({
                "pipeline_info": {
                    "store_prefix": self.current_store_prefix,
                    "total_categories": len(self.available_categories),
                    "query_category": category,
                    "query_type": "category_specific" if category else "multi_category"
                }
            })
            
            logger.info(f"Query processed successfully: {question[:50]}... (Category: {category or 'All'})")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def query_category(self, question: str, category: str) -> Dict[str, Any]:
        """Query documents within a specific category"""
        
        if category not in self.available_categories:
            raise ValueError(f"Category '{category}' not available. Available: {self.available_categories}")
        
        return self.query_documents(question, category)
    
    def compare_documents(self, question: str, category1: str, category2: str) -> Dict[str, Any]:
        """Compare documents between two categories"""
        
        if not self.pipeline_ready:
            raise ValueError("Pipeline not ready. Process documents or load existing stores first.")
        
        if len(self.available_categories) < 2:
            raise ValueError("At least 2 categories required for comparison")
        
        if category1 not in self.available_categories or category2 not in self.available_categories:
            raise ValueError(f"Invalid categories. Available: {self.available_categories}")
        
        try:
            response = self.analyzer.compare_documents(question, category1, category2)
            
            # Add pipeline metadata
            response.update({
                "pipeline_info": {
                    "store_prefix": self.current_store_prefix,
                    "comparison_categories": [category1, category2],
                    "total_categories": len(self.available_categories)
                }
            })
            
            logger.info(f"Document comparison completed: {category1} vs {category2}")
            return response
            
        except Exception as e:
            logger.error(f"Error comparing documents: {e}")
            raise
    
    # Enhanced versions of standard methods
    def get_document_summary(self, category: str = None) -> Dict[str, Any]:
        """Get a comprehensive summary of documents, optionally by category"""
        
        if not self.pipeline_ready:
            raise ValueError("Pipeline not ready. Process documents first.")
        return self.analyzer.summarize_documents(category)

    def get_document_summary_by_file(self, file_path: str) -> Dict[str, Any]:
        """Get a summary of a specific file by passing its content directly to the LLM."""
        content = self._get_file_content(file_path)
        return self.analyzer.summarize_documents(context=content)
    
    def explain_specific_clause(self, clause_description: str, category: str = None) -> Dict[str, Any]:
        """Explain a specific clause, optionally within a category"""
        
        if not self.pipeline_ready:
            raise ValueError("Pipeline not ready. Process documents first.")
        return self.analyzer.explain_clause(clause_description, category)

    def explain_clause_by_file(self, clause_description: str, file_path: str) -> Dict[str, Any]:
        content = self._get_file_content(file_path)
        return self.analyzer.explain_clause(clause_description, context=content)
    
    def find_key_obligations(self, category: str = None) -> Dict[str, Any]:
        """Find key obligations, optionally within a category"""
        
        if not self.pipeline_ready:
            raise ValueError("Pipeline not ready. Process documents first.")
        return self.analyzer.find_obligations(category)

    def find_obligations_by_file(self, file_path: str) -> Dict[str, Any]:
        content = self._get_file_content(file_path)
        return self.analyzer.find_obligations(context=content)
    
    def find_termination_clauses(self, category: str = None) -> Dict[str, Any]:
        """Find termination clauses, optionally within a category"""
        
        if not self.pipeline_ready:
            raise ValueError("Pipeline not ready. Process documents first.")
        return self.analyzer.find_termination_terms(category)

    def find_termination_clauses_by_file(self, file_path: str) -> Dict[str, Any]:
        content = self._get_file_content(file_path)
        return self.analyzer.find_termination_terms(context=content)
    
    # New comparison methods
    def compare_obligations(self, category1: str, category2: str) -> Dict[str, Any]:
        """Compare key obligations between two categories"""
        
        if not self.pipeline_ready:
            raise ValueError("Pipeline not ready. Process documents first.")
        
        return self.analyzer.compare_obligations(category1, category2)
    
    def compare_termination_clauses(self, category1: str, category2: str) -> Dict[str, Any]:
        """Compare termination clauses between two categories"""
        
        if not self.pipeline_ready:
            raise ValueError("Pipeline not ready. Process documents first.")
        
        return self.analyzer.compare_termination_clauses(category1, category2)
    
    def compare_specific_clauses(self, clause_description: str, category1: str, category2: str) -> Dict[str, Any]:
        """Compare specific clauses between two categories"""
        
        if not self.pipeline_ready:
            raise ValueError("Pipeline not ready. Process documents first.")
        
        return self.analyzer.compare_clauses(clause_description, category1, category2)
    
    # Utility methods
    def get_available_categories(self) -> List[str]:
        """Get list of available categories"""
        return self.available_categories
    
    def get_category_info(self, category: str = None) -> Dict[str, Any]:
        """Get information about categories"""
        return self.category_store_manager.get_category_info(category)
    
    def get_categorizations(self) -> List[Dict[str, Any]]:
        """Get categorization results for processed documents"""
        return self.categorizations
    
    def get_enhanced_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive enhanced pipeline status"""
        
        status = {
            "pipeline_ready": self.pipeline_ready,
            "current_store_prefix": self.current_store_prefix,
            "available_categories": self.available_categories,
            "total_categories": len(self.available_categories),
            "documents_in_memory": len(self.processed_documents),
            "categorizations_count": len(self.categorizations),
            "features": {
                "categorization": True,
                "category_specific_stores": True,
                "document_comparison": len(self.available_categories) >= 2,
                "cross_category_queries": True
            },
            "analyzer_status": self.analyzer.get_status() if self.pipeline_ready else None,
            "category_info": self.get_category_info() if self.pipeline_ready else None
        }
        
        return status
    
    def export_categorization_report(self, filepath: str = None) -> str:
        """Export categorization report"""
        
        if not self.categorizations:
            raise ValueError("No categorizations available to export")
        
        return self.document_processor.export_categorization_report(self.categorizations, filepath)
    
    def clear_conversation_history(self):
        """Clear the conversation memory"""
        if self.pipeline_ready:
            self.analyzer.clear_conversation()
            logger.info("Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history"""
        if self.pipeline_ready:
            return self.analyzer.get_conversation_history()
        return []
    
    def delete_category_stores(self, store_prefix: str = None) -> Dict[str, bool]:
        """Delete category stores"""
        
        prefix = store_prefix or self.current_store_prefix
        if not prefix:
            raise ValueError("No store prefix provided")
        
        # Get categories to delete
        categories_to_delete = self.available_categories if not store_prefix else []
        
        if not categories_to_delete:
            # Find categories by scanning directory
            try:
                available_stores = os.listdir(self.config.CATEGORY_STORE_FOLDER)
                categories_to_delete = [
                    store.replace(f"{prefix}_", "") 
                    for store in available_stores 
                    if store.startswith(prefix)
                ]
            except Exception:
                categories_to_delete = []
        
        # Delete each category store
        results = {}
        for category in categories_to_delete:
            try:
                success = self.category_store_manager.delete_category_store(category)
                results[category] = success
            except Exception as e:
                logger.error(f"Error deleting category store {category}: {e}")
                results[category] = False
        
        # Reset pipeline if current stores were deleted
        if store_prefix == self.current_store_prefix:
            self.current_store_prefix = None
            self.pipeline_ready = False
            self.processed_documents = []
            self.categorizations = []
            self.available_categories = []
            logger.info("Current stores deleted, pipeline reset")
        
        return results

    def query_documents_by_file(self, question: str, file_path: str) -> Dict[str, Any]:
        """Query a specific file by passing its content directly to the LLM."""
        content = self._get_file_content(file_path)
        return self.analyzer.ask_question_with_context(question, content)

def main():
    """Example usage of the Enhanced Legal RAG Pipeline"""
    
    try:
        # Initialize the enhanced pipeline
        pipeline = LegalRAGPipeline()
        
        print("🏛️  Enhanced Legal Document AI Analyzer - RAG Pipeline with Categories")
        print("=" * 80)
        print("\n✅ Enhanced pipeline initialized successfully!")
        
        print("\n🆕 New Features:")
        print("   📂 Automatic document categorization using AI")
        print("   🗃️  Separate vector stores for each document category")
        print("   ⚖️  Document comparison between categories")
        print("   🎯 Category-specific queries and analysis")
        
        print("\n📋 Available Commands:")
        print("1. Process new documents with categorization:")
        print("   pipeline.process_new_documents_with_categories(['file1.pdf', 'file2.docx'], 'store_name')")
        
        print("\n2. Load existing category stores:")
        print("   pipeline.load_existing_category_stores('store_prefix')")
        
        print("\n3. Query within specific category:")
        print("   pipeline.query_category('What are the main terms?', 'contract')")
        
        print("\n4. Query across all categories:")
        print("   pipeline.query_documents('What are payment terms?')")
        
        print("\n5. Compare documents between categories:")
        print("   pipeline.compare_documents('Compare termination clauses', 'contract', 'policy')")
        
        print("\n6. Category-specific analysis:")
        print("   pipeline.get_document_summary('contract')")
        print("   pipeline.find_key_obligations('employment')")
        print("   pipeline.find_termination_clauses('policy')")
        
        print("\n7. Compare specific aspects:")
        print("   pipeline.compare_obligations('contract', 'employment')")
        print("   pipeline.compare_termination_clauses('contract', 'policy')")
        
        print("\n8. Get category information:")
        print("   pipeline.get_available_categories()")
        print("   pipeline.get_category_info()")
        
        print("\n9. Export categorization report:")
        print("   pipeline.export_categorization_report()")
        
        # Show current status
        status = pipeline.get_enhanced_pipeline_status()
        print(f"\n📊 Current Status:")
        print(f"   Pipeline Ready: {status['pipeline_ready']}")
        print(f"   Available Categories: {len(status['available_categories'])}")
        print(f"   Document Categorization: {status['features']['categorization']}")
        print(f"   Document Comparison: {status['features']['document_comparison']}")
        
        if status['available_categories']:
            print(f"   Categories: {', '.join(status['available_categories'])}")
        
        print(f"\n🏷️  Supported Document Categories:")
        for category, description in pipeline.config.LEGAL_CATEGORIES.items():
            print(f"   • {category}: {description}")
        
        print("\n🚀 Ready to process legal documents with enhanced categorization!")
        print("   Remember to set your GEMINI_API_KEY in the .env file")
        
        # Example with dummy files (uncomment and modify for actual use):
        """
        # Example: Process legal documents with categorization
        file_paths = [
            "documents/service_contract.pdf",
            "documents/privacy_policy.docx", 
            "documents/employment_agreement.pdf",
            "documents/terms_of_service.txt"
        ]
        
        # Process documents with categorization
        result = pipeline.process_new_documents_with_categories(file_paths, "my_legal_docs_2024")
        print(f"Processing result: {result}")
        print(f"Categories found: {result['categories_found']}")
        
        # Query specific category
        contract_summary = pipeline.get_document_summary("contract")
        print(f"Contract Summary: {contract_summary['answer']}")
        
        # Compare between categories
        if len(result['categories_found']) >= 2:
            cat1, cat2 = result['categories_found'][:2]
            comparison = pipeline.compare_obligations(cat1, cat2)
            print(f"Obligation Comparison ({cat1} vs {cat2}): {comparison['answer']}")
        
        # Export categorization report
        report_path = pipeline.export_categorization_report()
        print(f"Categorization report saved to: {report_path}")
        """
        
    except Exception as e:
        print(f"❌ Error initializing enhanced pipeline: {e}")
        logger.error(f"Enhanced pipeline initialization failed: {e}")

if __name__ == "__main__":
    main()