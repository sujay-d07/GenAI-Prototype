# category_vector_store_manager.py - Category-based Vector Store Management

import os
# Fix OpenMP library conflict before importing FAISS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pickle
import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

# Updated imports for latest LangChain versions
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from config import Config
from models import get_model_manager

logger = logging.getLogger(__name__)

class CategoryVectorStoreManager:
    """Manages separate FAISS vector stores for each document category"""
    
    def __init__(self):
        self.config = Config()
        self.model_manager = get_model_manager()
        self.embeddings = self.model_manager.get_embeddings()
        
        # Dictionary to store vector stores by category
        self.category_stores = {}  # {category: FAISS_store}
        self.category_paths = {}   # {category: store_path}
        
        # Ensure category store directory exists
        os.makedirs(self.config.CATEGORY_STORE_FOLDER, exist_ok=True)
    
    def create_category_stores(self, categorized_documents: Dict[str, List[Document]], 
                             store_prefix: str = "legal_docs") -> Dict[str, bool]:
        """Create separate vector stores for each category"""
        
        if not categorized_documents:
            raise ValueError("No categorized documents provided")
        
        results = {}
        
        for category, documents in categorized_documents.items():
            if not documents:
                logger.warning(f"No documents for category: {category}")
                results[category] = False
                continue
            
            try:
                logger.info(f"Creating vector store for category '{category}' with {len(documents)} documents")
                
                # Create FAISS vector store for this category
                vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                
                # Store the vector store
                self.category_stores[category] = vector_store
                
                # Set store path
                store_name = f"{store_prefix}_{category}"
                store_path = os.path.join(self.config.CATEGORY_STORE_FOLDER, store_name)
                self.category_paths[category] = store_path
                
                results[category] = True
                logger.info(f"Successfully created vector store for category: {category}")
                
            except Exception as e:
                logger.error(f"Error creating vector store for category '{category}': {e}")
                results[category] = False
        
        logger.info(f"Created vector stores for {sum(results.values())} out of {len(categorized_documents)} categories")
        return results
    
    def save_category_stores(self) -> Dict[str, bool]:
        """Save all category vector stores to disk"""
        
        if not self.category_stores:
            raise ValueError("No category stores to save")
        
        results = {}
        
        for category, vector_store in self.category_stores.items():
            try:
                store_path = self.category_paths.get(category)
                if not store_path:
                    store_path = os.path.join(self.config.CATEGORY_STORE_FOLDER, f"default_{category}")
                    self.category_paths[category] = store_path
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(store_path), exist_ok=True)
                
                # Save the vector store
                vector_store.save_local(store_path)
                
                # Save category metadata
                metadata = {
                    'category': category,
                    'category_description': self.config.LEGAL_CATEGORIES.get(category, 'Unknown'),
                    'store_name': os.path.basename(store_path),
                    'creation_date': str(np.datetime64('now')),
                    'document_count': self.get_category_document_count(category),
                    'embedding_model': self.config.EMBEDDING_MODEL
                }
                
                metadata_path = f"{store_path}_metadata.pkl"
                with open(metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)
                
                results[category] = True
                logger.info(f"Saved vector store for category '{category}' to: {store_path}")
                
            except Exception as e:
                logger.error(f"Error saving vector store for category '{category}': {e}")
                results[category] = False
        
        return results
    
    def load_category_stores(self, store_prefix: str = "legal_docs") -> Dict[str, bool]:
        """Load all available category vector stores"""
        
        results = {}
        
        # Look for category stores in the category store folder
        if not os.path.exists(self.config.CATEGORY_STORE_FOLDER):
            logger.warning(f"Category store folder not found: {self.config.CATEGORY_STORE_FOLDER}")
            return results
        
        # Find all category store directories
        for item in os.listdir(self.config.CATEGORY_STORE_FOLDER):
            item_path = os.path.join(self.config.CATEGORY_STORE_FOLDER, item)
            
            if os.path.isdir(item_path) and item.startswith(store_prefix):
                # Extract category from store name
                category = item.replace(f"{store_prefix}_", "")
                
                try:
                    # Load the vector store
                    vector_store = FAISS.load_local(
                        item_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    self.category_stores[category] = vector_store
                    self.category_paths[category] = item_path
                    
                    results[category] = True
                    logger.info(f"Loaded vector store for category: {category}")
                    
                except Exception as e:
                    logger.error(f"Error loading vector store for category '{category}': {e}")
                    results[category] = False
        
        logger.info(f"Loaded {sum(results.values())} category vector stores")
        return results
    
    def load_specific_category_store(self, category: str, store_prefix: str = "legal_docs") -> bool:
        """Load vector store for a specific category"""
        
        try:
            store_name = f"{store_prefix}_{category}"
            store_path = os.path.join(self.config.CATEGORY_STORE_FOLDER, store_name)
            
            if not os.path.exists(store_path):
                logger.warning(f"Vector store not found for category '{category}' at: {store_path}")
                return False
            
            # Load the vector store
            vector_store = FAISS.load_local(
                store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            self.category_stores[category] = vector_store
            self.category_paths[category] = store_path
            
            logger.info(f"Successfully loaded vector store for category: {category}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store for category '{category}': {e}")
            return False
    
    def add_documents_to_category(self, category: str, documents: List[Document]) -> bool:
        """Add documents to an existing category vector store"""
        
        if category not in self.category_stores:
            logger.error(f"Category '{category}' not found in loaded stores")
            return False
        
        if not documents:
            logger.warning("No documents provided to add")
            return False
        
        try:
            # Filter documents to ensure they belong to the correct category
            category_docs = [
                doc for doc in documents 
                if doc.metadata.get('category') == category
            ]
            
            if not category_docs:
                logger.warning(f"No documents found for category '{category}'")
                return False
            
            # Add documents to existing store
            self.category_stores[category].add_documents(category_docs)
            logger.info(f"Added {len(category_docs)} documents to category '{category}'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to category '{category}': {e}")
            return False
    
    def get_category_store(self, category: str) -> Optional[FAISS]:
        """Get vector store for a specific category"""
        return self.category_stores.get(category)
    
    def get_category_retriever(self, category: str, search_type: str = "similarity", 
                             search_kwargs: Dict = None) -> Any:
        """Get a retriever for a specific category"""
        
        if category not in self.category_stores:
            raise ValueError(f"Category '{category}' not found in loaded stores")
        
        default_kwargs = {"k": self.config.TOP_K}
        if search_kwargs:
            default_kwargs.update(search_kwargs)
        
        try:
            retriever = self.category_stores[category].as_retriever(
                search_type=search_type,
                search_kwargs=default_kwargs
            )
            
            logger.info(f"Created retriever for category '{category}' with search_type={search_type}")
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever for category '{category}': {e}")
            raise
    
    def similarity_search_category(self, category: str, query: str, k: int = None) -> List[Document]:
        """Perform similarity search within a specific category"""
        
        if category not in self.category_stores:
            raise ValueError(f"Category '{category}' not found in loaded stores")
        
        k = k or self.config.TOP_K
        
        try:
            results = self.category_stores[category].similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents in category '{category}'")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search in category '{category}': {e}")
            raise
    
    def similarity_search_with_score_category(self, category: str, query: str, 
                                            k: int = None) -> List[tuple]:
        """Perform similarity search with scores within a specific category"""
        
        if category not in self.category_stores:
            raise ValueError(f"Category '{category}' not found in loaded stores")
        
        k = k or self.config.TOP_K
        
        try:
            results = self.category_stores[category].similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents with scores in category '{category}'")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search with scores in category '{category}': {e}")
            raise
    
    def get_category_document_count(self, category: str) -> int:
        """Get the number of documents in a specific category store"""
        
        if category not in self.category_stores:
            return 0
        
        try:
            return self.category_stores[category].index.ntotal
        except Exception as e:
            logger.warning(f"Could not get document count for category '{category}': {e}")
            return 0
    
    def get_all_categories(self) -> List[str]:
        """Get list of all loaded categories"""
        return list(self.category_stores.keys())
    
    def delete_category_store(self, category: str) -> bool:
        """Delete a category vector store"""
        
        try:
            # Remove from memory
            if category in self.category_stores:
                del self.category_stores[category]
            
            # Remove from disk
            if category in self.category_paths:
                store_path = self.category_paths[category]
                
                if os.path.exists(store_path):
                    import shutil
                    shutil.rmtree(store_path)
                    
                    # Remove metadata file
                    metadata_path = f"{store_path}_metadata.pkl"
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                
                del self.category_paths[category]
            
            logger.info(f"Deleted category store: {category}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting category store '{category}': {e}")
            return False
    
    def get_category_info(self, category: str = None) -> Dict[str, Any]:
        """Get information about category stores"""
        
        if category:
            # Get info for specific category
            if category not in self.category_stores:
                return {'error': f'Category {category} not found'}
            
            info = {
                'category': category,
                'loaded': True,
                'document_count': self.get_category_document_count(category),
                'store_path': self.category_paths.get(category)
            }
            
            # Load metadata if available
            store_path = self.category_paths.get(category)
            if store_path:
                metadata_path = f"{store_path}_metadata.pkl"
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'rb') as f:
                            metadata = pickle.load(f)
                        info.update(metadata)
                    except Exception as e:
                        logger.warning(f"Could not load metadata for category '{category}': {e}")
            
            return info
        
        else:
            # Get info for all categories
            all_info = {
                'total_categories': len(self.category_stores),
                'loaded_categories': list(self.category_stores.keys()),
                'total_documents': sum(self.get_category_document_count(cat) for cat in self.category_stores.keys()),
                'category_details': {}
            }
            
            for cat in self.category_stores.keys():
                all_info['category_details'][cat] = self.get_category_info(cat)
            
            return all_info
    
    def compare_categories(self, category1: str, category2: str, query: str, k: int = 3) -> Dict[str, Any]:
        """Compare search results between two categories"""
        
        if category1 not in self.category_stores:
            raise ValueError(f"Category '{category1}' not found")
        
        if category2 not in self.category_stores:
            raise ValueError(f"Category '{category2}' not found")
        
        try:
            # Search in both categories
            results1 = self.similarity_search_with_score_category(category1, query, k=k)
            results2 = self.similarity_search_with_score_category(category2, query, k=k)
            
            comparison = {
                'query': query,
                'categories': [category1, category2],
                'results': {
                    category1: [
                        {
                            'content': doc.page_content[:200] + "...",
                            'score': score,
                            'source': doc.metadata.get('source', 'Unknown'),
                            'page': doc.metadata.get('page_number', 1)
                        }
                        for doc, score in results1
                    ],
                    category2: [
                        {
                            'content': doc.page_content[:200] + "...",
                            'score': score,
                            'source': doc.metadata.get('source', 'Unknown'),
                            'page': doc.metadata.get('page_number', 1)
                        }
                        for doc, score in results2
                    ]
                },
                'summary': {
                    f'{category1}_count': len(results1),
                    f'{category2}_count': len(results2),
                    f'{category1}_avg_score': sum(score for _, score in results1) / len(results1) if results1 else 0,
                    f'{category2}_avg_score': sum(score for _, score in results2) / len(results2) if results2 else 0
                }
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing categories '{category1}' and '{category2}': {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        manager = CategoryVectorStoreManager()
        
        print("CategoryVectorStoreManager initialized successfully!")
        print("\nAvailable methods:")
        print("- create_category_stores(categorized_documents, store_prefix)")
        print("- save_category_stores()")
        print("- load_category_stores(store_prefix)")
        print("- load_specific_category_store(category, store_prefix)")
        print("- get_category_store(category)")
        print("- get_category_retriever(category, search_type, search_kwargs)")
        print("- similarity_search_category(category, query, k)")
        print("- get_category_info(category)")
        print("- compare_categories(category1, category2, query, k)")
        
        # Example with dummy data (uncomment to test with actual data)
        """
        from langchain.schema import Document
        
        # Create test documents with categories
        test_docs = {
            'contract': [
                Document(
                    page_content="This is a service agreement between parties...",
                    metadata={'source': 'contract.pdf', 'category': 'contract'}
                )
            ],
            'policy': [
                Document(
                    page_content="Company policy regarding data protection...",
                    metadata={'source': 'policy.pdf', 'category': 'policy'}
                )
            ]
        }
        
        # Create category stores
        results = manager.create_category_stores(test_docs, "test_docs")
        print(f"Creation results: {results}")
        
        # Save stores
        save_results = manager.save_category_stores()
        print(f"Save results: {save_results}")
        
        # Get info
        info = manager.get_category_info()
        print(f"Store info: {info}")
        """
        
    except Exception as e:
        print(f"Error during testing: {e}")
        logger.exception("Error during category vector store testing")