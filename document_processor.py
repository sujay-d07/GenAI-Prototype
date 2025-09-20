import os
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

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

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, preprocessing, categorization, and text splitting"""
    
    def __init__(self):
        self.config = Config()
        self.categorizer = DocumentCategorizer()
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
        
        print("DocumentProcessor with categorization initialized successfully!")
        print("Available methods:")
        print("- load_single_document(file_path, categorize=True)")
        print("- load_multiple_documents(file_paths, categorize=True)")
        print("- group_documents_by_category(documents)")
        print("- filter_documents_by_category(documents, category)")
        print("- get_document_stats(documents)")
        print("- export_categorization_report(categorizations)")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        logger.exception("Error during document processing test")