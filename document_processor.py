import os
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    PyPDFLoader, 
    TextLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import Config
from document_categorizer import DocumentCategorizer
from r2_storage_manager import R2StorageManager

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, preprocessing, categorization, and text splitting"""
    
    def __init__(self):
        self.config = Config()
        self.categorizer = DocumentCategorizer()
        
        # Initialize R2 storage manager if enabled
        if self.config.USE_R2_STORAGE:
            self.r2_storage = R2StorageManager()
            logger.info("R2 storage enabled for document processing")
        else:
            self.r2_storage = None
            logger.info("Using local storage for document processing")
        
        self._setup_text_splitter()
    
    def _setup_text_splitter(self):
        """Initialize the text splitter with configured parameters"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing and cleaning for legal documents"""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Clean up common PDF artifacts and special characters
        text = re.sub(r'[^\w\s.,;:!?()"\'-]', ' ', text)
        
        # Fix common OCR issues in legal documents
        text = re.sub(r'\b(\w)\s+(\w)\b', r'\1\2', text)  # Fix scattered letters
        
        # Normalize legal document formatting
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
        text = re.sub(r'\s*\n\s*', '\n', text)  # Clean up newline spacing
        
        return text.strip()
    
    def _extract_metadata(self, file_path: str, doc_type: str) -> Dict[str, Any]:
        """Extract metadata from document"""
        file_stat = os.stat(file_path)
        
        return {
            "source": os.path.basename(file_path),
            "document_type": doc_type,
            "upload_date": datetime.now().isoformat(),
            "file_path": file_path,
            "file_size": file_stat.st_size,
            "modification_date": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
        }
    
    def _detect_document_type(self, file_path: str) -> str:
        """Detect document type from file extension"""
        extension = Path(file_path).suffix.lower()
        
        type_mapping = {
            '.pdf': 'PDF Document',
            '.docx': 'Word Document (DOCX)',
            '.doc': 'Word Document (DOC)', 
            '.txt': 'Text File'
        }
        
        return type_mapping.get(extension, 'Unknown Document Type')
    
    def upload_document_to_r2(self, local_file_path: str, add_timestamp: bool = True, 
                             metadata: Dict[str, str] = None) -> Dict[str, Any]:
        """Upload a document to R2 storage and return upload information"""
        if not self.config.USE_R2_STORAGE or not self.r2_storage:
            raise ValueError("R2 storage is not enabled or configured")
        
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"File not found: {local_file_path}")
        
        # Validate file type
        file_extension = Path(local_file_path).suffix.lower()
        if file_extension not in self.config.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        try:
            # Prepare additional metadata
            doc_metadata = {
                'processor': 'DocumentProcessor',
                'upload_source': 'legal_rag_system',
                'file_type': self._detect_document_type(local_file_path)
            }
            
            if metadata:
                doc_metadata.update(metadata)
            
            # Upload to R2
            upload_result = self.r2_storage.upload_file(
                local_file_path=local_file_path,
                add_timestamp=add_timestamp,
                metadata=doc_metadata
            )
            
            logger.info(f"Document uploaded to R2: {local_file_path} -> {upload_result['r2_key']}")
            return upload_result
            
        except Exception as e:
            logger.error(f"Failed to upload document to R2: {e}")
            raise
    
    def _download_document_from_r2(self, r2_key: str, local_dir: str = None) -> str:
        """Download a document from R2 storage to local temporary location"""
        if not self.config.USE_R2_STORAGE or not self.r2_storage:
            raise ValueError("R2 storage is not enabled or configured")
        
        try:
            # Use upload folder as default download location
            if not local_dir:
                local_dir = self.config.UPLOAD_FOLDER
            
            # Ensure directory exists
            os.makedirs(local_dir, exist_ok=True)
            
            # Generate local file path
            filename = os.path.basename(r2_key)
            local_file_path = os.path.join(local_dir, filename)
            
            # Download from R2
            download_result = self.r2_storage.download_file(r2_key, local_file_path)
            
            logger.info(f"Document downloaded from R2: {r2_key} -> {local_file_path}")
            return local_file_path
            
        except Exception as e:
            logger.error(f"Failed to download document from R2: {e}")
            raise
    
    def load_single_document(self, file_path: str, categorize: bool = True) -> Tuple[List[Document], Dict[str, Any]]:
        """Load a single document and optionally categorize it"""
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in self.config.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        try:
            documents = []
            doc_type = self._detect_document_type(file_path)
            
            if file_extension == '.pdf':
                # Try UnstructuredPDFLoader first, fallback to PyPDFLoader
                try:
                    loader = UnstructuredPDFLoader(file_path)
                    documents = loader.load()
                    logger.info(f"Loaded PDF using UnstructuredPDFLoader: {file_path}")
                except Exception as e:
                    logger.warning(f"UnstructuredPDFLoader failed, trying PyPDFLoader: {e}")
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    logger.info(f"Loaded PDF using PyPDFLoader: {file_path}")
                
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
                logger.info(f"Loaded Word document: {file_path}")
                
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
                logger.info(f"Loaded text file: {file_path}")
            
            # Process and clean documents
            metadata = self._extract_metadata(file_path, doc_type)
            processed_docs = []
            
            for i, doc in enumerate(documents):
                # Preprocess content
                doc.page_content = self._preprocess_text(doc.page_content)
                
                # Update metadata
                doc.metadata.update(metadata)
                doc.metadata['page_number'] = i + 1
                doc.metadata['total_pages'] = len(documents)
                
                # Only keep documents with meaningful content
                if len(doc.page_content.strip()) > 50:  # Minimum content threshold
                    processed_docs.append(doc)
            
            # Categorize the document if requested
            categorization_result = None
            if categorize and processed_docs:
                # Use the first meaningful document for categorization
                categorization_result = self.categorizer.categorize_document(processed_docs[0])
                
                # Add category information to all document chunks
                for doc in processed_docs:
                    doc.metadata.update({
                        'category': categorization_result['category'],
                        'category_confidence': categorization_result['confidence'],
                        'category_explanation': categorization_result['explanation']
                    })
                
                logger.info(f"Document categorized as: {categorization_result['category']} "
                          f"(confidence: {categorization_result['confidence']:.2f})")
            
            logger.info(f"Processed {len(processed_docs)} pages from {file_path}")
            return processed_docs, categorization_result
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
    def load_document_from_r2(self, r2_key: str, categorize: bool = True, cleanup_local: bool = True) -> Tuple[List[Document], Dict[str, Any]]:
        """Load a document from R2 storage and process it"""
        
        if not self.config.USE_R2_STORAGE or not self.r2_storage:
            raise ValueError("R2 storage is not enabled or configured")
        
        if not self.r2_storage.file_exists(r2_key):
            raise FileNotFoundError(f"File not found in R2 storage: {r2_key}")
        
        local_file_path = None
        try:
            # Download file from R2 to temporary location
            local_file_path = self._download_document_from_r2(r2_key)
            
            # Process the downloaded file
            documents, categorization_result = self.load_single_document(local_file_path, categorize=categorize)
            
            # Add R2-specific metadata to all documents
            r2_metadata = self.r2_storage.get_file_metadata(r2_key)
            
            for doc in documents:
                doc.metadata.update({
                    'r2_key': r2_key,
                    'r2_bucket': self.config.R2_BUCKET_NAME,
                    'storage_type': 'r2',
                    'r2_metadata': r2_metadata
                })
            
            logger.info(f"Successfully loaded document from R2: {r2_key}")
            return documents, categorization_result
            
        except Exception as e:
            logger.error(f"Error loading document from R2 {r2_key}: {e}")
            raise
            
        finally:
            # Cleanup local temporary file if requested
            if cleanup_local and local_file_path and os.path.exists(local_file_path):
                try:
                    os.remove(local_file_path)
                    logger.debug(f"Cleaned up temporary file: {local_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary file {local_file_path}: {e}")
    
    def load_multiple_documents_from_r2(self, r2_keys: List[str], categorize: bool = True, 
                                       cleanup_local: bool = True) -> Tuple[List[Document], List[Dict[str, Any]]]:
        """Load multiple documents from R2 storage and process them"""
        
        if not self.config.USE_R2_STORAGE or not self.r2_storage:
            raise ValueError("R2 storage is not enabled or configured")
        
        all_documents = []
        all_categorizations = []
        failed_keys = []
        
        for r2_key in r2_keys:
            try:
                docs, categorization = self.load_document_from_r2(r2_key, categorize=categorize, cleanup_local=cleanup_local)
                all_documents.extend(docs)
                
                if categorization:
                    all_categorizations.append(categorization)
                
                logger.info(f"Successfully loaded from R2: {r2_key} "
                          f"(Category: {categorization['category'] if categorization else 'N/A'})")
                
            except Exception as e:
                logger.error(f"Failed to load from R2 {r2_key}: {e}")
                failed_keys.append(r2_key)
                continue
        
        if failed_keys:
            logger.warning(f"Failed to load {len(failed_keys)} files from R2: {failed_keys}")
        
        if not all_documents:
            raise ValueError("No documents were successfully loaded from R2")
        
        logger.info(f"Total documents loaded from R2: {len(all_documents)}")
        logger.info(f"Documents categorized: {len(all_categorizations)}")
        
        return all_documents, all_categorizations
    
    def load_documents_hybrid(self, file_sources: List[str], categorize: bool = True, 
                             upload_local_to_r2: bool = False, cleanup_local: bool = True) -> Tuple[List[Document], List[Dict[str, Any]]]:
        """
        Load documents from mixed sources (local files and R2 keys)
        
        Args:
            file_sources: List of local file paths or R2 keys
            categorize: Whether to categorize documents
            upload_local_to_r2: Whether to upload local files to R2 after processing
            cleanup_local: Whether to cleanup temporary local files
            
        Returns:
            Tuple of (all_documents, all_categorizations)
        """
        
        all_documents = []
        all_categorizations = []
        failed_sources = []
        
        for source in file_sources:
            try:
                # Determine if source is local file or R2 key
                if self._is_r2_key(source):
                    # Load from R2
                    docs, categorization = self.load_document_from_r2(source, categorize=categorize, cleanup_local=cleanup_local)
                    logger.info(f"Loaded from R2: {source}")
                    
                else:
                    # Load local file
                    docs, categorization = self.load_single_document(source, categorize=categorize)
                    
                    # Upload to R2 if requested and R2 is enabled
                    if upload_local_to_r2 and self.config.USE_R2_STORAGE and self.r2_storage:
                        try:
                            upload_result = self.upload_document_to_r2(source)
                            logger.info(f"Uploaded local file to R2: {source} -> {upload_result['r2_key']}")
                            
                            # Add R2 metadata to documents
                            for doc in docs:
                                doc.metadata.update({
                                    'r2_key': upload_result['r2_key'],
                                    'r2_bucket': self.config.R2_BUCKET_NAME,
                                    'storage_type': 'r2_uploaded'
                                })
                                
                        except Exception as e:
                            logger.warning(f"Failed to upload {source} to R2: {e}")
                    
                    logger.info(f"Loaded from local: {source}")
                
                all_documents.extend(docs)
                
                if categorization:
                    all_categorizations.append(categorization)
                
            except Exception as e:
                logger.error(f"Failed to load from source {source}: {e}")
                failed_sources.append(source)
                continue
        
        if failed_sources:
            logger.warning(f"Failed to load {len(failed_sources)} sources: {failed_sources}")
        
        if not all_documents:
            raise ValueError("No documents were successfully loaded from any source")
        
        logger.info(f"Total documents loaded (hybrid): {len(all_documents)}")
        logger.info(f"Documents categorized: {len(all_categorizations)}")
        
        return all_documents, all_categorizations
    
    def _is_r2_key(self, source: str) -> bool:
        """Determine if a source string is an R2 key or local file path"""
        # R2 keys typically start with the documents prefix and don't exist as local files
        if source.startswith(self.config.R2_DOCUMENTS_PREFIX):
            return True
        
        # If R2 is enabled and file doesn't exist locally, assume it's an R2 key
        if self.config.USE_R2_STORAGE and not os.path.exists(source):
            return True
        
        # Otherwise assume it's a local file path
        return False
    
    def list_available_documents(self) -> Dict[str, Any]:
        """List all available documents from both local and R2 storage"""
        
        result = {
            'local_documents': [],
            'r2_documents': [],
            'total_local': 0,
            'total_r2': 0
        }
        
        # List local documents
        if os.path.exists(self.config.UPLOAD_FOLDER):
            try:
                for filename in os.listdir(self.config.UPLOAD_FOLDER):
                    file_path = os.path.join(self.config.UPLOAD_FOLDER, filename)
                    if os.path.isfile(file_path):
                        file_ext = Path(filename).suffix.lower()
                        if file_ext in self.config.ALLOWED_EXTENSIONS:
                            file_stat = os.stat(file_path)
                            result['local_documents'].append({
                                'filename': filename,
                                'path': file_path,
                                'size': file_stat.st_size,
                                'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                                'type': self._detect_document_type(file_path)
                            })
                
                result['total_local'] = len(result['local_documents'])
                logger.info(f"Found {result['total_local']} local documents")
                
            except Exception as e:
                logger.warning(f"Failed to list local documents: {e}")
        
        # List R2 documents
        if self.config.USE_R2_STORAGE and self.r2_storage:
            try:
                r2_docs = self.r2_storage.list_documents()
                result['r2_documents'] = r2_docs
                result['total_r2'] = len(r2_docs)
                logger.info(f"Found {result['total_r2']} R2 documents")
                
            except Exception as e:
                logger.warning(f"Failed to list R2 documents: {e}")
        
        return result
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get comprehensive storage information"""
        
        info = {
            'r2_enabled': self.config.USE_R2_STORAGE,
            'local_storage_path': self.config.UPLOAD_FOLDER,
        }
        
        if self.config.USE_R2_STORAGE and self.r2_storage:
            try:
                r2_stats = self.r2_storage.get_storage_stats()
                info.update({
                    'r2_stats': r2_stats,
                    'r2_bucket': self.config.R2_BUCKET_NAME,
                    'r2_prefix': self.config.R2_DOCUMENTS_PREFIX
                })
            except Exception as e:
                info['r2_error'] = str(e)
        
        # Get local storage stats
        local_stats = {'total_files': 0, 'total_size': 0}
        if os.path.exists(self.config.UPLOAD_FOLDER):
            try:
                for filename in os.listdir(self.config.UPLOAD_FOLDER):
                    file_path = os.path.join(self.config.UPLOAD_FOLDER, filename)
                    if os.path.isfile(file_path):
                        file_ext = Path(filename).suffix.lower()
                        if file_ext in self.config.ALLOWED_EXTENSIONS:
                            local_stats['total_files'] += 1
                            local_stats['total_size'] += os.path.getsize(file_path)
            except Exception as e:
                local_stats['error'] = str(e)
        
        info['local_stats'] = local_stats
        
        return info
    
    def load_multiple_documents(self, file_paths: List[str], categorize: bool = True) -> Tuple[List[Document], List[Dict[str, Any]]]:
        """Load multiple documents and categorize them"""
        all_documents = []
        all_categorizations = []
        failed_files = []
        
        for file_path in file_paths:
            try:
                docs, categorization = self.load_single_document(file_path, categorize=categorize)
                all_documents.extend(docs)
                
                if categorization:
                    all_categorizations.append(categorization)
                
                logger.info(f"Successfully loaded: {file_path} "
                          f"(Category: {categorization['category'] if categorization else 'N/A'})")
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                failed_files.append(file_path)
                continue
        
        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")
        
        if not all_documents:
            raise ValueError("No documents were successfully loaded")
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        logger.info(f"Documents categorized: {len(all_categorizations)}")
        
        return all_documents, all_categorizations
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for processing"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                chunk.metadata['chunk_size'] = len(chunk.page_content)
                chunk.metadata['total_chunks'] = len(chunks)
            
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise
    
    def group_documents_by_category(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Group documents by their categories"""
        
        categorized_docs = {}
        
        for doc in documents:
            category = doc.metadata.get('category', 'other')
            
            if category not in categorized_docs:
                categorized_docs[category] = []
            
            categorized_docs[category].append(doc)
        
        # Log category distribution
        for category, docs in categorized_docs.items():
            logger.info(f"Category '{category}': {len(docs)} documents")
        
        return categorized_docs
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about the loaded documents including categories"""
        if not documents:
            return {}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chunk_size = total_chars / len(documents) if documents else 0
        
        # Count by document type
        doc_types = {}
        sources = set()
        categories = {}
        
        for doc in documents:
            # Document types
            doc_type = doc.metadata.get('document_type', 'Unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            # Sources
            sources.add(doc.metadata.get('source', 'Unknown'))
            
            # Categories
            category = doc.metadata.get('category', 'other')
            categories[category] = categories.get(category, 0) + 1
        
        return {
            'total_documents': len(documents),
            'total_characters': total_chars,
            'average_chunk_size': avg_chunk_size,
            'document_types': doc_types,
            'unique_sources': len(sources),
            'source_files': list(sources),
            'categories': categories,
            'category_distribution': {
                cat: {
                    'count': count,
                    'percentage': round((count / len(documents)) * 100, 2)
                }
                for cat, count in categories.items()
            }
        }
    
    def filter_documents_by_category(self, documents: List[Document], category: str) -> List[Document]:
        """Filter documents by a specific category"""
        
        if category not in self.config.LEGAL_CATEGORIES:
            raise ValueError(f"Unknown category: {category}. Valid categories: {list(self.config.LEGAL_CATEGORIES.keys())}")
        
        filtered_docs = [
            doc for doc in documents 
            if doc.metadata.get('category') == category
        ]
        
        logger.info(f"Filtered {len(filtered_docs)} documents for category: {category}")
        return filtered_docs
    
    def get_categories_summary(self, categorizations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get a summary of all document categories"""
        
        if not categorizations:
            return {}
        
        return self.categorizer.get_category_statistics(categorizations)
    
    def export_categorization_report(self, categorizations: List[Dict[str, Any]], filepath: str = None) -> str:
        """Export categorization report"""
        
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.config.LOGS_FOLDER, f"categorization_report_{timestamp}.json")
        
        return self.categorizer.export_categorizations(categorizations, filepath)

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        processor = DocumentProcessor()
        
        # Test with sample files (uncomment to test with actual files)
        """
        file_paths = [
            "documents/contract.pdf",
            "documents/privacy_policy.docx",
            "documents/terms_of_service.txt"
        ]
        
        # Load and categorize documents
        documents, categorizations = processor.load_multiple_documents(file_paths)
        
        # Get statistics
        stats = processor.get_document_stats(documents)
        print(f"Document Statistics: {stats}")
        
        # Group by category
        grouped_docs = processor.group_documents_by_category(documents)
        print(f"Documents by category: {list(grouped_docs.keys())}")
        
        # Split documents
        chunks = processor.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Export categorization report
        report_path = processor.export_categorization_report(categorizations)
        print(f"Categorization report exported to: {report_path}")
        """
        
        print("DocumentProcessor with R2 and categorization initialized successfully!")
        print("Available methods:")
        print("- load_single_document(file_path, categorize=True)")
        print("- load_multiple_documents(file_paths, categorize=True)")
        print("- load_document_from_r2(r2_key, categorize=True)")
        print("- load_multiple_documents_from_r2(r2_keys, categorize=True)")
        print("- load_documents_hybrid(file_sources, categorize=True)")
        print("- upload_document_to_r2(local_file_path)")
        print("- list_available_documents()")
        print("- get_storage_info()")
        print("- group_documents_by_category(documents)")
        print("- filter_documents_by_category(documents, category)")
        print("- get_document_stats(documents)")
        print("- export_categorization_report(categorizations)")
        
        # Show storage configuration
        storage_info = processor.get_storage_info()
        print(f"\n📊 Storage Configuration:")
        print(f"R2 Storage Enabled: {storage_info['r2_enabled']}")
        if storage_info['r2_enabled']:
            if 'r2_stats' in storage_info:
                print(f"R2 Documents: {storage_info['r2_stats'].get('total_documents', 0)}")
            if 'r2_bucket' in storage_info:
                print(f"R2 Bucket: {storage_info['r2_bucket']}")
        
        print(f"Local Documents: {storage_info['local_stats']['total_files']}")
        
        # List available documents
        available_docs = processor.list_available_documents()
        print(f"\n📁 Available Documents:")
        print(f"Local: {available_docs['total_local']}, R2: {available_docs['total_r2']}")
        
        # Example usage with R2
        """
        # Upload local file to R2
        if os.path.exists("documents/sample.pdf"):
            upload_result = processor.upload_document_to_r2("documents/sample.pdf")
            print(f"Uploaded to R2: {upload_result['r2_key']}")
            
            # Load from R2
            documents, categorizations = processor.load_document_from_r2(upload_result['r2_key'])
            print(f"Loaded from R2: {len(documents)} documents")
        
        # Mixed source loading
        file_sources = [
            "local_documents/contract.pdf",  # Local file
            "documents/policy_20240101_120000.pdf",  # R2 key
            "local_documents/agreement.docx"  # Local file
        ]
        
        docs, cats = processor.load_documents_hybrid(
            file_sources, 
            upload_local_to_r2=True  # Upload local files to R2
        )
        print(f"Hybrid loading: {len(docs)} documents from mixed sources")
        """
        
    except Exception as e:
        print(f"Error during testing: {e}")
        logger.exception("Error during document processing test")