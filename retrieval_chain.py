# retrieval_chain.py - RAG Chain with Category Support and Comparison

import logging
from typing import Dict, Any, List, Optional, Tuple

# Updated imports for latest LangChain versions
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.messages import HumanMessage, AIMessage

from config import Config
from models import get_model_manager
from category_vector_store_manager import CategoryVectorStoreManager

logger = logging.getLogger(__name__)

class CategoryAwareLegalRAGChain:
    """Enhanced RAG chain with category-specific retrieval and document comparison"""
    
    def __init__(self):
        self.config = Config()
        self.model_manager = get_model_manager()
        self.category_store_manager = None  # Will be set from outside
        
        # Initialize components
        self.llm = self.model_manager.get_llm()
        self.memory = None
        self.category_chains = {}  # {category: ConversationalRetrievalChain}
        
        self._setup_memory()
        self._create_legal_prompts()
    
    def _setup_memory(self):
        """Setup conversation memory"""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )
        logger.info("Conversation memory initialized")
    
    def _create_legal_prompts(self):
        """Create specialized prompt templates for different use cases"""
        
        # Standard legal analysis prompt
        self.standard_prompt_template = """You are a specialized AI assistant for legal document analysis. Your expertise lies in helping users understand complex legal documents by providing clear, accurate summaries and explanations.

        **Your Role:**
        - Analyze legal documents and provide accessible explanations
        - Break down complex legal jargon into simple, understandable language  
        - Help users understand their rights, obligations, and important terms
        - Provide accurate information while avoiding legal advice

        **Context:**
        {context}

        **Previous Conversation:**
        {chat_history}

        **Current Question:** {question}

        **Instructions:**
        1. **Accuracy First**: Base your response strictly on the provided document context
        2. **Clear Communication**: Use plain language to explain legal concepts
        3. **Source Attribution**: Always cite specific documents when referencing information
        4. **Structure**: Organize your response in the following format:
        - **Summary**: Provide a concise overview of the relevant document sections
        - **Flaws**: Highlight any risks, ambiguities, or problematic clauses
        - **How to Prevent**: Suggest ways to mitigate risks or clarify issues
        5. **Quotes**: When explaining clauses, quote the relevant text and then explain it
        6. **Limitations**: If information isn't in the provided context, clearly state this
        7. **No Legal Advice**: Provide information and explanations, not legal advice

        **Response Format:**
        - **Summary**: 
        - **Flaws**: 
        - **How to Prevent**: 

        **Answer:**"""

        self.standard_prompt = PromptTemplate(
            template=self.standard_prompt_template,
            input_variables=["context", "chat_history", "question"]
        )

        # Comparison prompt template
        self.comparison_prompt_template = """You are a specialized AI assistant for comparing legal documents. Your task is to analyze and compare documents from different categories or sources to identify similarities, differences, and potential conflicts.

        **Comparison Task:** Compare documents from {category1} and {category2} categories

        **Context from {category1} Documents:**
        {context1}

        **Context from {category2} Documents:**
        {context2}

        **Previous Conversation:**
        {chat_history}

        **Comparison Question:** {question}

        **Instructions:**
        1. **Compare Systematically**: Analyze both document sets for the requested information
        2. **Identify Differences**: Highlight key differences between the categories
        3. **Find Similarities**: Note common elements or overlapping provisions
        4. **Risk Assessment**: Identify potential conflicts or inconsistencies
        5. **Clear Structure**: Organize your comparison clearly

        **Response Format:**
        - **Summary**: Brief overview of what was compared
        - **Key Similarities**: Common elements between the document categories
        - **Key Differences**: Major differences and variations
        - **Potential Issues**: Conflicts, gaps, or inconsistencies identified
        - **Recommendations**: Suggestions for addressing any issues found

        **Your Comparison:**"""

        self.comparison_prompt = PromptTemplate(
            template=self.comparison_prompt_template,
            input_variables=["context1", "context2", "chat_history", "question", "category1", "category2"]
        )

        logger.info("Legal prompt templates created")
    
    def setup_category_chains(self, category_store_manager: CategoryVectorStoreManager, categories: List[str] = None):
        """Setup retrieval chains for specific categories"""
        
        # Store the category store manager reference
        self.category_store_manager = category_store_manager
        
        if categories is None:
            categories = self.category_store_manager.get_all_categories()
        
        if not categories:
            raise ValueError("No categories available. Load category stores first.")
        
        try:
            logger.info(f"Setting up category chains for categories: {categories}")
            
            for category in categories:
                if category in self.category_chains:
                    logger.info(f"Category chain for '{category}' already exists, skipping")
                    continue  # Already set up
                
                # Check if category store exists
                category_store = self.category_store_manager.get_category_store(category)
                if category_store is None:
                    logger.warning(f"No vector store found for category: {category}")
                    continue
                
                # Get retriever for this category
                try:
                    retriever = self.category_store_manager.get_category_retriever(
                        category,
                        search_type="similarity",
                        search_kwargs={
                            "k": self.config.TOP_K,
                            "score_threshold": 0.7
                        }
                    )
                    
                    # Create category-specific prompt
                    category_prompt = PromptTemplate(
                        template=self.standard_prompt_template,
                        input_variables=["context", "chat_history", "question"]
                    )
                    
                    # Create conversational retrieval chain for this category
                    chain = ConversationalRetrievalChain.from_llm(
                        llm=self.llm,
                        retriever=retriever,
                        memory=self.memory,
                        combine_docs_chain_kwargs={
                            "prompt": category_prompt
                        },
                        return_source_documents=True,
                        verbose=True,
                        max_tokens_limit=4000
                    )
                    
                    self.category_chains[category] = chain
                    logger.info(f"Successfully setup retrieval chain for category: {category}")
                    
                except Exception as e:
                    logger.error(f"Failed to setup retrieval chain for category '{category}': {e}")
                    continue
            
            if not self.category_chains:
                raise ValueError("No category chains could be created")
            
            logger.info(f"Setup completed for {len(self.category_chains)} categories: {list(self.category_chains.keys())}")
            
        except Exception as e:
            logger.error(f"Error setting up category chains: {e}")
            raise
    
    def query_category(self, question: str, category: str, include_sources: bool = True) -> Dict[str, Any]:
        """Query documents within a specific category"""
        
        if category not in self.category_chains:
            available_categories = list(self.category_chains.keys())
            raise ValueError(f"Category '{category}' not available. Available categories: {available_categories}")
        
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            logger.info(f"Processing query for category '{category}': {question[:100]}...")
            
            # Execute the category-specific retrieval chain
            response = self.category_chains[category].invoke({"question": question})
            logger.info(f"Raw chain response: {response}")

            # Handle both dict and string responses
            if isinstance(response, dict):
                answer = response.get("answer", "")
                sources = response.get("source_documents", [])
            elif isinstance(response, str):
                answer = response
                sources = []
            else:
                answer = str(response)
                sources = []

            result = {
                "question": question,
                "category": category,
                "category_description": self.config.LEGAL_CATEGORIES.get(category, category),
                "answer": answer,
                "sources": [],
                "chat_history_length": len(self.memory.chat_memory.messages)
            }

            # Extract and format source documents
            if include_sources and sources:
                result["sources"] = self._format_source_documents(sources)

            logger.info(f"Query processed successfully for category: {category}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query for category '{category}': {e}")
            raise
    
    def query_all_categories(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """Query across all loaded categories and aggregate results"""
        
        if not self.category_chains:
            raise ValueError("No category chains available")
        
        try:
            all_results = {}
            combined_sources = []
            
            for category in self.category_chains.keys():
                try:
                    result = self.query_category(question, category, include_sources)
                    all_results[category] = result
                    
                    if include_sources:
                        combined_sources.extend(result.get("sources", []))
                        
                except Exception as e:
                    logger.warning(f"Failed to query category '{category}': {e}")
                    all_results[category] = {
                        "error": str(e),
                        "category": category
                    }
            
            # Create aggregated response
            aggregated_result = {
                "question": question,
                "query_type": "multi_category",
                "categories_queried": list(self.category_chains.keys()),
                "category_results": all_results,
                "combined_sources": combined_sources,
                "total_sources": len(combined_sources)
            }
            
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Error querying all categories: {e}")
            raise
    
    def compare_documents(self, question: str, category1: str, category2: str, 
                         include_sources: bool = True) -> Dict[str, Any]:
        """Compare documents between two categories"""
        
        if category1 not in self.category_chains or category2 not in self.category_chains:
            available_cats = list(self.category_chains.keys())
            raise ValueError(f"One or both categories not available. Available: {available_cats}")
        
        try:
            logger.info(f"Comparing categories '{category1}' vs '{category2}' for question: {question[:100]}...")
            
            # Get relevant documents from both categories
            retriever1 = self.category_store_manager.get_category_retriever(category1)
            retriever2 = self.category_store_manager.get_category_retriever(category2)
            
            # Retrieve relevant documents
            docs1 = retriever1.get_relevant_documents(question)
            docs2 = retriever2.get_relevant_documents(question)
            
            # Prepare contexts
            context1 = "\n\n".join([doc.page_content for doc in docs1])
            context2 = "\n\n".join([doc.page_content for doc in docs2])
            
            # Format comparison prompt
            chat_history_str = self._format_chat_history()
            
            prompt = self.comparison_prompt.format(
                context1=context1,
                context2=context2,
                chat_history=chat_history_str,
                question=question,
                category1=self.config.LEGAL_CATEGORIES.get(category1, category1),
                category2=self.config.LEGAL_CATEGORIES.get(category2, category2)
            )
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Add to memory
            self.memory.save_context(
                {"question": f"Compare {category1} and {category2}: {question}"},
                {"answer": answer}
            )
            
            # Format result
            result = {
                "question": question,
                "comparison_type": "category_comparison",
                "category1": category1,
                "category2": category2,
                "category1_description": self.config.LEGAL_CATEGORIES.get(category1, category1),
                "category2_description": self.config.LEGAL_CATEGORIES.get(category2, category2),
                "answer": answer,
                "sources": [],
                "document_counts": {
                    category1: len(docs1),
                    category2: len(docs2)
                }
            }
            
            # Format sources if requested
            if include_sources:
                sources1 = self._format_source_documents(docs1, category1)
                sources2 = self._format_source_documents(docs2, category2)
                
                result["sources"] = {
                    category1: sources1,
                    category2: sources2,
                    "total": len(sources1) + len(sources2)
                }
            
            logger.info(f"Document comparison completed between '{category1}' and '{category2}'")
            return result
            
        except Exception as e:
            logger.error(f"Error comparing categories '{category1}' and '{category2}': {e}")
            raise
    
    def _format_source_documents(self, source_docs: List[Document], category: str = None) -> List[Dict[str, Any]]:
        """Format source documents for response"""
        
        formatted_sources = []
        seen_sources = set()
        
        for doc in source_docs:
            source_key = f"{doc.metadata.get('source', 'Unknown')}_{doc.metadata.get('page_number', 0)}"
            
            if source_key in seen_sources:
                continue
            
            seen_sources.add(source_key)
            
            source_info = {
                "document": doc.metadata.get("source", "Unknown Document"),
                "document_type": doc.metadata.get("document_type", "Unknown"),
                "category": doc.metadata.get("category", category or "Unknown"),
                "page_number": doc.metadata.get("page_number"),
                "content_preview": self._create_content_preview(doc.page_content),
                "upload_date": doc.metadata.get("upload_date"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "confidence": doc.metadata.get("category_confidence")
            }
            
            formatted_sources.append(source_info)
        
        return formatted_sources
    
    def _create_content_preview(self, content: str, max_length: int = 200) -> str:
        """Create a preview of document content"""
        
        if len(content) <= max_length:
            return content
        
        # Try to cut at sentence boundary
        preview = content[:max_length]
        last_period = preview.rfind('. ')
        
        if last_period > max_length * 0.7:
            preview = preview[:last_period + 1]
        else:
            preview = preview + "..."
        
        return preview.strip()
    
    def _format_chat_history(self) -> str:
        """Format chat history for prompt inclusion"""
        
        if not self.memory or not self.memory.chat_memory:
            return ""
        
        history_parts = []
        for message in self.memory.chat_memory.messages[-4:]:  # Last 4 messages
            if isinstance(message, HumanMessage):
                history_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                history_parts.append(f"AI: {message.content[:200]}...")
        
        return "\n".join(history_parts)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history"""
        
        history = []
        if self.memory and self.memory.chat_memory:
            for message in self.memory.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    history.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    history.append({"role": "assistant", "content": message.content})
        
        return history
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories"""
        return list(self.category_chains.keys())
    
    def get_category_status(self) -> Dict[str, Any]:
        """Get status of all category chains"""
        
        status = {
            "total_categories": len(self.category_chains),
            "available_categories": list(self.category_chains.keys()),
            "memory_stats": self.get_memory_stats(),
            "category_details": {}
        }
        
        if self.category_store_manager:
            for category in self.category_chains.keys():
                doc_count = self.category_store_manager.get_category_document_count(category)
                status["category_details"][category] = {
                    "description": self.config.LEGAL_CATEGORIES.get(category, category),
                    "document_count": doc_count,
                    "chain_ready": True
                }
        
        return status
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about current memory usage"""
        
        stats = {
            "total_messages": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "memory_buffer_length": 0
        }
        
        if self.memory and self.memory.chat_memory:
            messages = self.memory.chat_memory.messages
            stats["total_messages"] = len(messages)
            stats["user_messages"] = sum(1 for msg in messages if isinstance(msg, HumanMessage))
            stats["assistant_messages"] = sum(1 for msg in messages if isinstance(msg, AIMessage))
            
            total_chars = sum(len(msg.content) for msg in messages)
            stats["memory_buffer_length"] = total_chars
        
        return stats

class LegalDocumentAnalyzer:
    def compare_documents_by_text(self, question: str, text1: str, text2: str, file1: str = "Document 1", file2: str = "Document 2") -> Dict[str, Any]:
        """Compare two documents by their text content (not by category)"""
        if not text1 or not text2:
            raise ValueError("Both documents must have text content for comparison.")
        # Prepare prompt for document-level comparison
        chat_history_str = self._format_chat_history() if hasattr(self, '_format_chat_history') else ""
        prompt = f"""
Compare the following two legal documents for the question below.

Document 1 ({file1}):\n{text1[:2000]}{'...' if len(text1) > 2000 else ''}

Document 2 ({file2}):\n{text2[:2000]}{'...' if len(text2) > 2000 else ''}

Question: {question}

{chat_history_str}
Provide a detailed, structured comparison, highlighting similarities, differences, and any important legal implications.
"""
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        result = {
            "question": question,
            "comparison_type": "document_file_comparison",
            "file1": file1,
            "file2": file2,
            "answer": answer
        }
        return result
    """High-level interface for category-aware legal document analysis with comparison features"""
    
    def __init__(self):
        self.config = Config()
        self.rag_chain = CategoryAwareLegalRAGChain()
        self.category_store_manager = CategoryVectorStoreManager()
        self._is_ready = False
        self._available_categories = []
        # Use the same LLM as the RAG chain for direct document comparison
        self.llm = self.rag_chain.llm
    
    def setup_with_category_stores(self, store_prefix: str = "legal_docs"):
        """Setup analyzer with category-based vector stores"""
        try:
            # Load category stores
            load_results = self.category_store_manager.load_category_stores(store_prefix)
            
            if not any(load_results.values()):
                raise ValueError("No category stores could be loaded")
            
            # Get successfully loaded categories
            self._available_categories = [cat for cat, success in load_results.items() if success]
            
            if not self._available_categories:
                raise ValueError("No categories were successfully loaded")
            
            # Setup RAG chains for loaded categories - FIXED: Pass category store manager
            self.rag_chain.setup_category_chains(self.category_store_manager, self._available_categories)
            
            self._is_ready = True
            logger.info(f"Analyzer setup completed with {len(self._available_categories)} categories: {self._available_categories}")
            
        except Exception as e:
            logger.error(f"Error setting up analyzer with category stores: {e}")
            raise
    
    def ask_question(self, question: str, category: str = None) -> Dict[str, Any]:
        """Ask a question about legal documents, optionally within a specific category"""
        
        if not self._is_ready:
            raise ValueError("Analyzer not ready. Setup category stores first.")
        
        if category:
            if category not in self._available_categories:
                raise ValueError(f"Category '{category}' not available. Available: {self._available_categories}")
            return self.rag_chain.query_category(question, category)
        else:
            return self.rag_chain.query_all_categories(question)
    
    def ask_question_category(self, question: str, category: str) -> Dict[str, Any]:
        """Ask a question within a specific category"""
        return self.ask_question(question, category)
    
    def compare_documents(self, question: str, category1: str, category2: str) -> Dict[str, Any]:
        """Compare documents between two categories"""
        
        if not self._is_ready:
            raise ValueError("Analyzer not ready. Setup category stores first.")
        
        return self.rag_chain.compare_documents(question, category1, category2)
    
    def summarize_documents(self, category: str = None) -> Dict[str, Any]:
        """Get a summary of documents, optionally within a specific category"""
        
        summary_question = (
            "Please provide a comprehensive summary of all the legal documents, "
            "including the main parties involved, key terms and conditions, "
            "important obligations, rights, and any critical deadlines or clauses."
        )
        
        return self.ask_question(summary_question, category)
    
    def explain_clause(self, clause_description: str, category: str = None) -> Dict[str, Any]:
        """Explain a specific clause, optionally within a specific category"""
        
        explanation_question = (
            f"Please find and explain the clause or section related to '{clause_description}'. "
            "Quote the relevant text and then explain what it means in simple terms, "
            "including any implications or important details."
        )
        
        return self.ask_question(explanation_question, category)
    
    def find_obligations(self, category: str = None) -> Dict[str, Any]:
        """Find obligations, optionally within a specific category"""
        
        obligations_question = (
            "What are the key obligations and responsibilities for each party "
            "mentioned in these legal documents? Please organize by party and "
            "include specific requirements, deadlines, and consequences."
        )
        
        return self.ask_question(obligations_question, category)
    
    def find_termination_terms(self, category: str = None) -> Dict[str, Any]:
        """Find termination clauses, optionally within a specific category"""
        
        termination_question = (
            "What are the termination conditions and procedures outlined in these documents? "
            "Include information about notice periods, conditions that trigger termination, "
            "and any penalties or procedures that must be followed."
        )
        
        return self.ask_question(termination_question, category)
    
    def compare_obligations(self, category1: str, category2: str) -> Dict[str, Any]:
        """Compare obligations between two document categories"""
        
        comparison_question = (
            "Compare the key obligations and responsibilities outlined in these document categories. "
            "Identify similarities, differences, and potential conflicts in the obligations."
        )
        
        return self.compare_documents(comparison_question, category1, category2)
    
    def compare_termination_clauses(self, category1: str, category2: str) -> Dict[str, Any]:
        """Compare termination clauses between two document categories"""
        
        comparison_question = (
            "Compare the termination conditions and procedures between these document categories. "
            "Highlight differences in notice periods, termination triggers, and procedures."
        )
        
        return self.compare_documents(comparison_question, category1, category2)
    
    def compare_clauses(self, clause_description: str, category1: str, category2: str) -> Dict[str, Any]:
        """Compare specific clauses between two document categories"""
        
        comparison_question = (
            f"Compare the clauses related to '{clause_description}' between these document categories. "
            "Identify key differences, similarities, and potential inconsistencies."
        )
        
        return self.compare_documents(comparison_question, category1, category2)
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories"""
        return self._available_categories
    
    def get_category_info(self, category: str = None) -> Dict[str, Any]:
        """Get information about categories"""
        return self.category_store_manager.get_category_info(category)
    
    def is_ready(self) -> bool:
        """Check if analyzer is ready"""
        return self._is_ready
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        
        status = {
            "ready": self._is_ready,
            "available_categories": self._available_categories,
            "total_categories": len(self._available_categories),
            "category_info": self.get_category_info(),
            "rag_chain_status": self.rag_chain.get_category_status() if self._is_ready else None
        }
        
        return status
    
    def clear_conversation(self):
        """Clear conversation history"""
        if self._is_ready:
            self.rag_chain.clear_memory()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        if self._is_ready:
            return self.rag_chain.get_conversation_history()
        return []

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        analyzer = LegalDocumentAnalyzer()
        
        print("Enhanced Legal Document Analyzer initialized!")
        print("\nNew Features:")
        print("- Category-specific document analysis")
        print("- Document comparison between categories")
        print("- Multi-category querying")
        
        print("\nAvailable methods:")
        print("- setup_with_category_stores(store_prefix)")
        print("- ask_question(question, category=None)")
        print("- ask_question_category(question, category)")
        print("- compare_documents(question, category1, category2)")
        print("- summarize_documents(category=None)")
        print("- find_obligations(category=None)")
        print("- compare_obligations(category1, category2)")
        print("- compare_termination_clauses(category1, category2)")
        print("- get_available_categories()")
        print("- get_status()")
        
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        logger.exception("Error during initialization")