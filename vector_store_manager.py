# vector_store_manager.py - FAISS Vector Store Management

import os
# Fix OpenMP library conflict before importing FAISS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pickle
import logging
from typing import List, Optional, Dict, Any
import numpy as np

# Updated imports for latest LangChain versions
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from config import Config
from models import get_model_manager

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages FAISS vector store operations"""
    
    def __init__(self):
        self.config = Config()
        self.model_manager = get_model_manager()
        self.embeddings = self.model_manager.get_embeddings()
        self.vector_store = None
        self._store_path = None
    
    def create_vector_store(self, documents: List[Document], store_name: str = "legal_docs") -> FAISS:
        """Create a new FAISS vector store from documents"""
        
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        try:
            logger.info(f"Creating vector store with {len(documents)} documents...")
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Set store path
            self._store_path = os.path.join(self.config.VECTOR_STORE_FOLDER, store_name)
            
            logger.info(f"Vector store created successfully with {len(documents)} documents")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add new documents to existing vector store"""
        
        if not self.vector_store:
            raise ValueError("No vector store exists. Create one first.")
        
        if not documents:
            logger.warning("No documents provided to add")
            return False
        
        try:
            # Add documents to existing store
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    def save_vector_store(self, store_name: str = None) -> bool:
        """Save vector store to disk"""
        
        if not self.vector_store:
            raise ValueError("No vector store to save")
        
        try:
            if store_name:
                self._store_path = os.path.join(self.config.VECTOR_STORE_FOLDER, store_name)
            
            if not self._store_path:
                self._store_path = os.path.join(self.config.VECTOR_STORE_FOLDER, "default_legal_store")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self._store_path), exist_ok=True)
            
            # Save the vector store
            self.vector_store.save_local(self._store_path)
            
            # Save additional metadata
            metadata = {
                'store_name': os.path.basename(self._store_path),
                'creation_date': str(np.datetime64('now')),
                'document_count': self.get_document_count(),
                'embedding_model': self.config.EMBEDDING_MODEL
            }
            
            metadata_path = f"{self._store_path}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Vector store saved to: {self._store_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            return False
    
    def load_vector_store(self, store_path: str) -> bool:
        """Load vector store from disk"""
        
        try:
            full_path = os.path.join(self.config.VECTOR_STORE_FOLDER, store_path) if not os.path.isabs(store_path) else store_path
            
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Vector store not found at: {full_path}")
            
            # Load the vector store
            self.vector_store = FAISS.load_local(
                full_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            self._store_path = full_path
            
            # Load metadata if available
            metadata_path = f"{full_path}_metadata.pkl"
            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    logger.info(f"Loaded metadata: {metadata}")
                except Exception as e:
                    logger.warning(f"Could not load metadata: {e}")
            
            logger.info(f"Vector store loaded from: {full_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Perform similarity search in vector store"""
        
        if not self.vector_store:
            raise ValueError("No vector store loaded")
        
        k = k or self.config.TOP_K
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = None) -> List[tuple]:
        """Perform similarity search with relevance scores"""
        
        if not self.vector_store:
            raise ValueError("No vector store loaded")
        
        k = k or self.config.TOP_K
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search with scores: {e}")
            raise
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: Dict = None) -> Any:
        """Get a retriever for the vector store"""
        
        if not self.vector_store:
            raise ValueError("No vector store loaded")
        
        default_kwargs = {"k": self.config.TOP_K}
        if search_kwargs:
            default_kwargs.update(search_kwargs)
        
        try:
            retriever = self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs=default_kwargs
            )
            
            logger.info(f"Created retriever with search_type={search_type}, kwargs={default_kwargs}")
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector store"""
        
        if not self.vector_store:
            return 0
        
        try:
            # FAISS vector store doesn't have a direct document count method
            # We'll estimate from the index size
            return self.vector_store.index.ntotal
            
        except Exception as e:
            logger.warning(f"Could not get document count: {e}")
            return 0
    
    def delete_vector_store(self, store_name: str = None) -> bool:
        """Delete a vector store from disk"""
        
        try:
            if store_name:
                store_path = os.path.join(self.config.VECTOR_STORE_FOLDER, store_name)
            else:
                store_path = self._store_path
            
            if not store_path or not os.path.exists(store_path):
                logger.warning(f"Vector store not found: {store_path}")
                return False
            
            # Remove directory and all files
            import shutil
            shutil.rmtree(store_path)
            
            # Remove metadata file if it exists
            metadata_path = f"{store_path}_metadata.pkl"
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            logger.info(f"Deleted vector store: {store_path}")
            
            # Clear current store if it was deleted
            if store_path == self._store_path:
                self.vector_store = None
                self._store_path = None
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting vector store: {e}")
            return False
    
    def list_available_stores(self) -> List[str]:
        """List all available vector stores"""
        
        try:
            if not os.path.exists(self.config.VECTOR_STORE_FOLDER):
                return []
            
            stores = []
            for item in os.listdir(self.config.VECTOR_STORE_FOLDER):
                item_path = os.path.join(self.config.VECTOR_STORE_FOLDER, item)
                if os.path.isdir(item_path) and not item.endswith('_metadata.pkl'):
                    stores.append(item)
            
            return sorted(stores)
            
        except Exception as e:
            logger.error(f"Error listing vector stores: {e}")
            return []
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get information about the current vector store"""
        
        info = {
            'loaded': self.vector_store is not None,
            'store_path': self._store_path,
            'document_count': self.get_document_count() if self.vector_store else 0
        }
        
        # Load metadata if available
        if self._store_path:
            metadata_path = f"{self._store_path}_metadata.pkl"
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    info.update(metadata)
                except Exception as e:
                    logger.warning(f"Could not load store metadata: {e}")
        
        return info

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the vector store manager
    try:
        manager = VectorStoreManager()
        
        # Create sample documents for testing
        test_docs = [
            Document(
                page_content="This is a sample legal contract with terms and conditions.",
                metadata={"source": "contract1.pdf", "page": 1}
            ),
            Document(
                page_content="Agreement contains payment terms and termination clauses.",
                metadata={"source": "contract1.pdf", "page": 2}
            )
        ]
        
        # Create vector store
        vector_store = manager.create_vector_store(test_docs, "test_store")
        
        # Test similarity search
        results = manager.similarity_search("payment terms", k=1)
        print(f"Search results: {len(results)} documents found")
        
        # Save vector store
        success = manager.save_vector_store("test_store")
        print(f"Save successful: {success}")
        
        # List available stores
        stores = manager.list_available_stores()
        print(f"Available stores: {stores}")
        
        # Get store info
        info = manager.get_store_info()
        print(f"Store info: {info}")
        
    except Exception as e:
        print(f"Error during testing: {e}")