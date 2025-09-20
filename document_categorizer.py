# document_categorizer.py - Comprehensive Document Categorization Module

import os
import json
import logging
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path

from langchain.schema import Document
from langchain.prompts import PromptTemplate

from config import Config
from models import get_model_manager

logger = logging.getLogger(__name__)

class DocumentCategorizer:
    """
    Advanced legal document categorization system using LLM analysis
    with fallback mechanisms and comprehensive reporting
    """
    
    def __init__(self):
        self.config = Config()
        self.model_manager = get_model_manager()
        self.llm = self.model_manager.get_llm()
        
        # Initialize categorization components
        self._setup_categorization_prompt()
        self.categorization_cache = {}
        self.categorization_stats = {
            'total_categorizations': 0,
            'cache_hits': 0,
            'llm_categorizations': 0,
            'fallback_categorizations': 0,
            'failed_categorizations': 0
        }
        
        # Load existing cache
        self._load_categorization_cache()
        
        logger.info("DocumentCategorizer initialized successfully")
    
    def _setup_categorization_prompt(self):
        """Setup the comprehensive prompt template for document categorization"""
        
        # Create detailed category descriptions
        category_details = []
        for category, name in self.config.LEGAL_CATEGORIES.items():
            description = self.config.CATEGORY_DESCRIPTIONS.get(category, "")
            keywords = ", ".join(self.config.CATEGORY_KEYWORDS.get(category, [])[:5])  # First 5 keywords
            category_details.append(f"• **{category}**: {name}\n  Description: {description}\n  Key indicators: {keywords}")
        
        category_info_text = "\n\n".join(category_details)
        
        categorization_template = f"""You are an expert legal document classifier with deep knowledge of legal document types and structures. Your task is to analyze the provided document content and classify it into the most appropriate category.

**AVAILABLE CATEGORIES:**

{category_info_text}

**DOCUMENT CONTENT TO ANALYZE:**
{{content}}

**CLASSIFICATION INSTRUCTIONS:**
1. **Read Carefully**: Analyze the document content, structure, and legal language
2. **Identify Key Elements**: Look for specific legal terms, document structure, parties involved, and purpose
3. **Match to Category**: Choose the MOST APPROPRIATE category from the list above
4. **Primary Focus**: If the document contains multiple types, choose the PRIMARY/DOMINANT category
5. **Confidence Assessment**: Rate your confidence in the classification (0.0 to 1.0)
6. **Explanation Required**: Provide a clear explanation for your choice
7. **Key Indicators**: List specific terms or phrases that influenced your decision

**CLASSIFICATION CRITERIA:**
- **Content Analysis**: What is the main purpose and subject matter?
- **Legal Structure**: What type of legal framework does it follow?
- **Parties Involved**: Who are the main parties and their relationships?
- **Legal Obligations**: What types of obligations, rights, or procedures are outlined?
- **Document Format**: What is the typical format and structure?

**RESPONSE FORMAT (MUST BE VALID JSON):**
{{
    "category": "category_key_from_list",
    "confidence": 0.85,
    "explanation": "Clear explanation of why this category was chosen based on document analysis",
    "key_indicators": ["specific", "terms", "or", "phrases", "found"],
    "alternative_category": "second_best_option_if_any",
    "document_type_detected": "specific type within category",
    "analysis_notes": "Additional observations about the document"
}}

**IMPORTANT NOTES:**
- Only use category keys from the provided list
- Confidence should reflect how certain you are (higher = more certain)
- Explanation should be specific and reference actual content
- Key indicators should be actual words/phrases found in the document
- If unsure between categories, choose the most dominant one and mention alternatives

**YOUR CLASSIFICATION:**"""

        self.categorization_prompt = PromptTemplate(
            template=categorization_template,
            input_variables=["content"]
        )
        
        logger.info("Comprehensive categorization prompt template created")
    
    def _load_categorization_cache(self):
        """Load previously cached categorizations with error handling"""
        cache_file = os.path.join(self.config.LOGS_FOLDER, self.config.CATEGORIZATION_SETTINGS['cache_file'])
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                # Validate cache format
                if isinstance(cache_data, dict):
                    self.categorization_cache = cache_data.get('categorizations', {})
                    self.categorization_stats.update(cache_data.get('stats', {}))
                    logger.info(f"Loaded {len(self.categorization_cache)} cached categorizations")
                else:
                    # Legacy format support
                    self.categorization_cache = cache_data
                    logger.info(f"Loaded {len(self.categorization_cache)} cached categorizations (legacy format)")
                    
            except Exception as e:
                logger.warning(f"Could not load categorization cache: {e}")
                self.categorization_cache = {}
        else:
            self.categorization_cache = {}
            logger.info("No existing categorization cache found, starting fresh")
    
    def _save_categorization_cache(self):
        """Save categorizations to cache with statistics"""
        if not self.config.CATEGORIZATION_SETTINGS['use_cache']:
            return
        
        cache_file = os.path.join(self.config.LOGS_FOLDER, self.config.CATEGORIZATION_SETTINGS['cache_file'])
        
        try:
            cache_data = {
                'version': '2.0',
                'last_updated': datetime.now().isoformat(),
                'total_entries': len(self.categorization_cache),
                'categorizations': self.categorization_cache,
                'stats': self.categorization_stats
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Categorization cache saved with {len(self.categorization_cache)} entries")
            
        except Exception as e:
            logger.error(f"Could not save categorization cache: {e}")
    
    def _create_content_hash(self, content: str) -> str:
        """Create a hash of document content for caching"""
        # Use first N characters for consistent hashing
        max_length = self.config.CATEGORIZATION_SETTINGS.get('max_content_length', 3000)
        content_sample = content[:max_length].strip().lower()
        return hashlib.md5(content_sample.encode('utf-8')).hexdigest()
    
    def categorize_document(self, document: Document) -> Dict[str, Any]:
        """
        Categorize a single document using LLM analysis with fallback
        
        Args:
            document: LangChain Document object to categorize
            
        Returns:
            Dictionary containing categorization results
        """
        
        try:
            # Check cache first if enabled
            content_hash = self._create_content_hash(document.page_content)
            
            if (self.config.CATEGORIZATION_SETTINGS['use_cache'] and 
                content_hash in self.categorization_cache):
                
                self.categorization_stats['cache_hits'] += 1
                logger.info("Using cached categorization")
                return self.categorization_cache[content_hash]
            
            # Prepare content for analysis
            max_length = self.config.CATEGORIZATION_SETTINGS['max_content_length']
            content = document.page_content[:max_length].strip()
            
            if not content or len(content) < 50:
                logger.warning("Document content too short for reliable categorization")
                return self._create_default_categorization(document, "Content too short", content_hash)
            
            # Try LLM categorization first
            try:
                categorization_result = self._llm_categorize(content, document)
                self.categorization_stats['llm_categorizations'] += 1
                
            except Exception as llm_error:
                logger.warning(f"LLM categorization failed: {llm_error}, using fallback")
                categorization_result = self._fallback_categorization(content, document)
                self.categorization_stats['fallback_categorizations'] += 1
            
            # Validate and enhance result
            categorization_result = self._validate_and_enhance_result(
                categorization_result, document, content_hash
            )
            
            # Cache the result if enabled
            if self.config.CATEGORIZATION_SETTINGS['use_cache']:
                self.categorization_cache[content_hash] = categorization_result
                self._save_categorization_cache()
            
            self.categorization_stats['total_categorizations'] += 1
            
            logger.info(f"Document categorized as: {categorization_result['category']} "
                       f"(confidence: {categorization_result['confidence']:.2f})")
            
            return categorization_result
            
        except Exception as e:
            logger.error(f"Error categorizing document: {e}")
            self.categorization_stats['failed_categorizations'] += 1
            return self._create_default_categorization(document, f"Error: {str(e)}", 
                                                     self._create_content_hash(document.page_content))
    
    def _llm_categorize(self, content: str, document: Document) -> Dict[str, Any]:
        """Perform LLM-based categorization"""
        
        # Format the prompt with content
        prompt = self.categorization_prompt.format(content=content)
        
        # Get LLM response
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                categorization_result = json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON found in response", response_text, 0)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse LLM JSON response: {e}")
            # Try alternative parsing or fallback
            categorization_result = self._parse_non_json_response(response_text, content)
        
        return categorization_result
    
    def _parse_non_json_response(self, response_text: str, content: str) -> Dict[str, Any]:
        """Parse non-JSON LLM responses as fallback"""
        
        logger.info("Attempting to parse non-JSON LLM response")
        
        # Extract category from response text
        category = 'other'
        confidence = 0.5
        explanation = "Parsed from non-JSON response"
        
        # Look for category mentions in response
        response_lower = response_text.lower()
        for cat_key in self.config.LEGAL_CATEGORIES.keys():
            if cat_key in response_lower or self.config.LEGAL_CATEGORIES[cat_key].lower() in response_lower:
                category = cat_key
                confidence = 0.6  # Moderate confidence for parsed response
                explanation = f"Category '{cat_key}' mentioned in response"
                break
        
        # If still 'other', try keyword matching
        if category == 'other':
            fallback_result = self._fallback_categorization(content)
            category = fallback_result['category']
            confidence = min(fallback_result['confidence'], 0.5)  # Lower confidence
            explanation = f"Fallback categorization: {fallback_result['explanation']}"
        
        return {
            'category': category,
            'confidence': confidence,
            'explanation': explanation,
            'key_indicators': [],
            'method': 'parsed_response'
        }
    
    def _fallback_categorization(self, content: str, document: Document = None) -> Dict[str, Any]:
        """
        Fallback categorization using keyword matching and heuristics
        
        Args:
            content: Document content to analyze
            document: Optional document object for metadata
            
        Returns:
            Categorization result dictionary
        """
        
        content_lower = content.lower()
        
        # Score each category based on keyword matches
        category_scores = {}
        found_keywords = {}
        
        for category, keywords in self.config.CATEGORY_KEYWORDS.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                # Count occurrences of each keyword
                keyword_count = content_lower.count(keyword.lower())
                if keyword_count > 0:
                    score += keyword_count
                    matched_keywords.append(keyword)
            
            if score > 0:
                category_scores[category] = score
                found_keywords[category] = matched_keywords
        
        # Determine best category
        if category_scores:
            # Get category with highest score
            best_category = max(category_scores, key=category_scores.get)
            max_score = category_scores[best_category]
            
            # Calculate confidence based on score and keyword matches
            confidence = min(0.8, (max_score / 10) + 0.3)  # Max 0.8 for fallback
            
            explanation = (f"Keyword-based categorization. Found {max_score} relevant terms: "
                          f"{', '.join(found_keywords[best_category][:3])}")
            
            key_indicators = found_keywords[best_category][:5]  # Top 5 matched keywords
            
        else:
            # No keywords matched, use document metadata if available
            best_category = self._analyze_document_metadata(document) if document else 'other'
            confidence = 0.3
            explanation = "No keywords matched, used metadata analysis or default classification"
            key_indicators = []
        
        return {
            'category': best_category,
            'confidence': confidence,
            'explanation': explanation,
            'key_indicators': key_indicators,
            'method': 'fallback_keywords',
            'keyword_scores': category_scores
        }
    
    def _analyze_document_metadata(self, document: Document) -> str:
        """Analyze document metadata for categorization hints"""
        
        if not document or not document.metadata:
            return 'other'
        
        metadata = document.metadata
        
        # Check filename for hints
        filename = metadata.get('source', '').lower()
        
        filename_hints = {
            'contract': ['contract', 'agreement', 'service'],
            'policy': ['policy', 'procedure', 'guideline'],
            'employment': ['employment', 'hr', 'job', 'employee'],
            'privacy_policy': ['privacy', 'data_protection'],
            'terms_of_service': ['terms', 'tos', 'conditions'],
            'financial': ['financial', 'payment', 'invoice'],
            'license': ['license', 'eula', 'copyright']
        }
        
        for category, hints in filename_hints.items():
            if any(hint in filename for hint in hints):
                return category
        
        return 'other'
    
    def _validate_and_enhance_result(self, result: Dict[str, Any], document: Document, 
                                   content_hash: str) -> Dict[str, Any]:
        """Validate and enhance categorization result"""
        
        # Ensure required fields exist
        if 'category' not in result:
            result['category'] = 'other'
        
        # Validate category
        if result['category'] not in self.config.LEGAL_CATEGORIES:
            logger.warning(f"Invalid category: {result['category']}, defaulting to 'other'")
            result['category'] = 'other'
        
        # Ensure confidence is within valid range
        confidence = result.get('confidence', 0.5)
        result['confidence'] = max(0.0, min(1.0, float(confidence)))
        
        # Apply minimum confidence threshold
        min_threshold = self.config.CATEGORIZATION_SETTINGS['min_confidence_threshold']
        if result['confidence'] < min_threshold:
            result['category'] = 'other'
            result['confidence'] = min_threshold
            result['explanation'] = f"Confidence below threshold, defaulted to 'other'. Original: {result.get('explanation', '')}"
        
        # Add standard metadata
        result.update({
            'categorization_date': datetime.now().isoformat(),
            'source_document': document.metadata.get('source', 'Unknown') if document else 'Unknown',
            'content_hash': content_hash,
            'category_name': self.config.LEGAL_CATEGORIES.get(result['category'], result['category']),
            'method': result.get('method', 'llm_analysis')
        })
        
        # Ensure lists exist
        if 'key_indicators' not in result:
            result['key_indicators'] = []
        
        return result
    
    def _create_default_categorization(self, document: Document, reason: str, 
                                     content_hash: str) -> Dict[str, Any]:
        """Create a default categorization result"""
        
        return {
            'category': 'other',
            'confidence': 0.1,
            'explanation': f'Default categorization used: {reason}',
            'key_indicators': [],
            'categorization_date': datetime.now().isoformat(),
            'source_document': document.metadata.get('source', 'Unknown') if document else 'Unknown',
            'content_hash': content_hash,
            'category_name': 'Other Legal Document',
            'method': 'default'
        }
    
    def categorize_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Categorize multiple documents with progress tracking
        
        Args:
            documents: List of Document objects to categorize
            
        Returns:
            List of categorization results
        """
        
        categorizations = []
        total_docs = len(documents)
        
        logger.info(f"Starting categorization of {total_docs} documents")
        
        for i, document in enumerate(documents, 1):
            try:
                logger.info(f"Categorizing document {i}/{total_docs}: "
                           f"{document.metadata.get('source', 'Unknown')}")
                
                categorization = self.categorize_document(document)
                
                # Add categorization info to document metadata
                document.metadata.update({
                    'category': categorization['category'],
                    'category_confidence': categorization['confidence'],
                    'category_explanation': categorization['explanation'],
                    'category_name': categorization.get('category_name', 'Unknown')
                })
                
                categorizations.append(categorization)
                
                # Log progress every 10 documents
                if i % 10 == 0 or i == total_docs:
                    logger.info(f"Progress: {i}/{total_docs} documents categorized")
                
            except Exception as e:
                logger.error(f"Failed to categorize document {i}: {e}")
                
                # Add error categorization
                error_categorization = self._create_default_categorization(
                    document, f"Categorization error: {str(e)}", 
                    self._create_content_hash(document.page_content if document.page_content else "")
                )
                categorizations.append(error_categorization)
                
                # Update document metadata with error info
                document.metadata.update({
                    'category': 'other',
                    'category_confidence': 0.1,
                    'category_explanation': f'Error during categorization: {str(e)}',
                    'category_name': 'Other Legal Document'
                })
        
        logger.info(f"Categorization completed: {len(categorizations)} results")
        self._log_categorization_summary(categorizations)
        
        return categorizations
    
    def _log_categorization_summary(self, categorizations: List[Dict[str, Any]]):
        """Log a summary of categorization results"""
        
        if not categorizations:
            return
        
        # Count categories
        category_counts = {}
        total_confidence = 0
        
        for cat in categorizations:
            category = cat.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
            total_confidence += cat.get('confidence', 0)
        
        avg_confidence = total_confidence / len(categorizations)
        
        logger.info("Categorization Summary:")
        logger.info(f"  Total documents: {len(categorizations)}")
        logger.info(f"  Average confidence: {avg_confidence:.3f}")
        logger.info(f"  Categories found: {len(category_counts)}")
        
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(categorizations)) * 100
            category_name = self.config.LEGAL_CATEGORIES.get(category, category)
            logger.info(f"    {category_name}: {count} docs ({percentage:.1f}%)")
    
    def get_category_statistics(self, categorizations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about document categories
        
        Args:
            categorizations: List of categorization results
            
        Returns:
            Dictionary containing detailed statistics
        """
        
        if not categorizations:
            return {
                'total_documents': 0,
                'error': 'No categorizations provided'
            }
        
        # Basic counts
        total_docs = len(categorizations)
        category_counts = {}
        confidence_scores = []
        methods_used = {}
        
        # Analyze each categorization
        for cat in categorizations:
            category = cat.get('category', 'unknown')
            confidence = cat.get('confidence', 0)
            method = cat.get('method', 'unknown')
            
            category_counts[category] = category_counts.get(category, 0) + 1
            confidence_scores.append(confidence)
            methods_used[method] = methods_used.get(method, 0) + 1
        
        # Calculate statistics
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        min_confidence = min(confidence_scores) if confidence_scores else 0
        max_confidence = max(confidence_scores) if confidence_scores else 0
        
        # Confidence distribution
        high_confidence = sum(1 for c in confidence_scores if c >= 0.8)
        medium_confidence = sum(1 for c in confidence_scores if 0.5 <= c < 0.8)
        low_confidence = sum(1 for c in confidence_scores if c < 0.5)
        
        # Category distribution with percentages
        category_distribution = {}
        for category, count in category_counts.items():
            percentage = (count / total_docs) * 100
            category_name = self.config.LEGAL_CATEGORIES.get(category, category)
            
            category_distribution[category] = {
                'count': count,
                'percentage': round(percentage, 2),
                'name': category_name
            }
        
        # Most and least common categories
        most_common = max(category_counts, key=category_counts.get) if category_counts else None
        least_common = min(category_counts, key=category_counts.get) if category_counts else None
        
        return {
            'total_documents': total_docs,
            'unique_categories': len(category_counts),
            'category_distribution': category_distribution,
            'confidence_statistics': {
                'average': round(avg_confidence, 3),
                'minimum': round(min_confidence, 3),
                'maximum': round(max_confidence, 3),
                'high_confidence_docs': high_confidence,
                'medium_confidence_docs': medium_confidence,
                'low_confidence_docs': low_confidence
            },
            'methods_used': methods_used,
            'most_common_category': {
                'key': most_common,
                'name': self.config.LEGAL_CATEGORIES.get(most_common, most_common) if most_common else None,
                'count': category_counts.get(most_common, 0) if most_common else 0
            },
            'least_common_category': {
                'key': least_common,
                'name': self.config.LEGAL_CATEGORIES.get(least_common, least_common) if least_common else None,
                'count': category_counts.get(least_common, 0) if least_common else 0
            },
            'categorizer_stats': self.categorization_stats.copy()
        }
    
    def export_categorizations(self, categorizations: List[Dict[str, Any]], 
                              filepath: str = None) -> Optional[str]:
        """
        Export categorization results to a comprehensive JSON report
        
        Args:
            categorizations: List of categorization results
            filepath: Optional custom filepath
            
        Returns:
            Path to exported file or None if failed
        """
        
        if not self.config.CATEGORIZATION_SETTINGS['export_reports']:
            logger.info("Export reports disabled in configuration")
            return None
        
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.config.LOGS_FOLDER, f"categorization_report_{timestamp}.json")
        
        try:
            # Generate comprehensive report
            export_data = {
                'report_metadata': {
                    'export_date': datetime.now().isoformat(),
                    'total_documents': len(categorizations),
                    'categorizer_version': '2.0',
                    'config_summary': {
                        'categories_available': len(self.config.LEGAL_CATEGORIES),
                        'fallback_enabled': self.config.CATEGORIZATION_SETTINGS['fallback_enabled'],
                        'cache_enabled': self.config.CATEGORIZATION_SETTINGS['use_cache']
                    }
                },
                'statistics': self.get_category_statistics(categorizations),
                'categorizations': categorizations,
                'category_definitions': {
                    category: {
                        'name': self.config.LEGAL_CATEGORIES[category],
                        'description': self.config.CATEGORY_DESCRIPTIONS.get(category, ''),
                        'keywords': self.config.CATEGORY_KEYWORDS.get(category, [])
                    }
                    for category in self.config.LEGAL_CATEGORIES.keys()
                }
            }
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Categorization report exported to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting categorizations: {e}")
            return None
    
    def get_categorizer_stats(self) -> Dict[str, Any]:
        """Get current categorizer statistics"""
        
        cache_size = len(self.categorization_cache)
        
        return {
            'cache_statistics': {
                'cache_size': cache_size,
                'cache_enabled': self.config.CATEGORIZATION_SETTINGS['use_cache']
            },
            'performance_stats': self.categorization_stats.copy(),
            'configuration': {
                'total_categories': len(self.config.LEGAL_CATEGORIES),
                'min_confidence_threshold': self.config.CATEGORIZATION_SETTINGS['min_confidence_threshold'],
                'max_content_length': self.config.CATEGORIZATION_SETTINGS['max_content_length'],
                'fallback_enabled': self.config.CATEGORIZATION_SETTINGS['fallback_enabled']
            }
        }
    
    def clear_cache(self) -> bool:
        """Clear categorization cache"""
        
        try:
            self.categorization_cache.clear()
            
            # Remove cache file if it exists
            cache_file = os.path.join(self.config.LOGS_FOLDER, 
                                    self.config.CATEGORIZATION_SETTINGS['cache_file'])
            
            if os.path.exists(cache_file):
                os.remove(cache_file)
            
            # Reset stats
            self.categorization_stats = {
                'total_categorizations': 0,
                'cache_hits': 0,
                'llm_categorizations': 0,
                'fallback_categorizations': 0,
                'failed_categorizations': 0
            }
            
            logger.info("Categorization cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

# Example usage and comprehensive testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        print("🏷️  Document Categorizer - Comprehensive Testing")
        print("=" * 60)
        
        # Initialize categorizer
        categorizer = DocumentCategorizer()
        
        # Create comprehensive test documents
        test_documents = [
            Document(
                page_content="""SERVICE AGREEMENT
                This Service Agreement is entered into between TechCorp Inc. and ClientCorp Ltd. 
                for the provision of software development services. The parties agree to the 
                following terms and conditions regarding the scope of work, payment terms, 
                intellectual property rights, and termination procedures. The contractor shall
                provide services as outlined in Exhibit A attached hereto.""",
                metadata={"source": "service_agreement.pdf", "page": 1}
            ),
            Document(
                page_content="""PRIVACY POLICY - DATA PROTECTION NOTICE
                This privacy policy explains how we collect, use, store, and protect your 
                personal information when you use our website and services. We are committed 
                to protecting your privacy and ensuring compliance with GDPR and other data 
                protection regulations. We collect personal data including name, email, 
                and usage analytics through cookies.""",
                metadata={"source": "privacy_policy.html", "page": 1}
            ),
            Document(
                page_content="""EMPLOYMENT CONTRACT
                This Employment Agreement is between ABC Corporation and John Doe for the 
                position of Software Engineer. Terms include salary of $75,000 annually, 
                benefits package, non-disclosure agreement, and termination procedures. 
                Employee agrees to confidentiality and non-compete clauses during and 
                after employment.""",
                metadata={"source": "employment_contract.docx", "page": 1}
            ),
            Document(
                page_content="""COMPANY POLICY MANUAL
                This document outlines corporate policies and procedures for all employees.
                Includes guidelines for workplace conduct, IT usage policies, vacation 
                procedures, and compliance requirements. All employees must acknowledge 
                and follow these internal policies and procedures.""",
                metadata={"source": "policy_manual.pdf", "page": 1}
            )
        ]
        
        # Test single document categorization
        print("\n📄 Testing Single Document Categorization:")
        print("-" * 50)
        
        for i, doc in enumerate(test_documents, 1):
            result = categorizer.categorize_document(doc)
            print(f"\nDocument {i}: {doc.metadata.get('source', 'Unknown')}")
            print(f"Category: {result['category']} ({result.get('category_name', '')})")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Method: {result.get('method', 'unknown')}")
            print(f"Explanation: {result['explanation'][:100]}...")
            if result.get('key_indicators'):
                print(f"Key indicators: {', '.join(result['key_indicators'][:3])}")
        
        # Test batch categorization
        print(f"\n📚 Testing Batch Categorization:")
        print("-" * 50)
        
        categorizations = categorizer.categorize_documents(test_documents)
        
        # Generate and display statistics
        print(f"\n📊 Categorization Statistics:")
        print("-" * 50)
        
        stats = categorizer.get_category_statistics(categorizations)
        
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Unique Categories: {stats['unique_categories']}")
        print(f"Average Confidence: {stats['confidence_statistics']['average']:.3f}")
        
        print(f"\nCategory Distribution:")
        for category, info in stats['category_distribution'].items():
            print(f"  • {info['name']}: {info['count']} docs ({info['percentage']}%)")
        
        print(f"\nConfidence Distribution:")
        conf_stats = stats['confidence_statistics']
        print(f"  • High confidence (≥0.8): {conf_stats['high_confidence_docs']} docs")
        print(f"  • Medium confidence (0.5-0.8): {conf_stats['medium_confidence_docs']} docs")
        print(f"  • Low confidence (<0.5): {conf_stats['low_confidence_docs']} docs")
        
        print(f"\nMethods Used:")
        for method, count in stats['methods_used'].items():
            print(f"  • {method}: {count} docs")
        
        # Test categorizer statistics
        print(f"\n🔧 Categorizer Performance:")
        print("-" * 50)
        
        categorizer_stats = categorizer.get_categorizer_stats()
        perf_stats = categorizer_stats['performance_stats']
        
        print(f"Total categorizations: {perf_stats['total_categorizations']}")
        print(f"Cache hits: {perf_stats['cache_hits']}")
        print(f"LLM categorizations: {perf_stats['llm_categorizations']}")
        print(f"Fallback categorizations: {perf_stats['fallback_categorizations']}")
        print(f"Failed categorizations: {perf_stats['failed_categorizations']}")
        
        cache_stats = categorizer_stats['cache_statistics']
        print(f"Cache size: {cache_stats['cache_size']} entries")
        print(f"Cache enabled: {cache_stats['cache_enabled']}")
        
        # Test export functionality
        print(f"\n💾 Testing Export Functionality:")
        print("-" * 50)
        
        export_path = categorizer.export_categorizations(categorizations)
        if export_path:
            print(f"✅ Report exported to: {export_path}")
            
            # Check file size
            if os.path.exists(export_path):
                file_size = os.path.getsize(export_path)
                print(f"   File size: {file_size:,} bytes")
        else:
            print("❌ Export failed or disabled")
        
        # Test fallback categorization
        print(f"\n🔄 Testing Fallback Categorization:")
        print("-" * 50)
        
        # Create a document that should trigger fallback
        fallback_doc = Document(
            page_content="This document contains payment terms and financial obligations for loan repayment.",
            metadata={"source": "test_fallback.txt", "page": 1}
        )
        
        # Force fallback by testing directly
        fallback_result = categorizer._fallback_categorization(fallback_doc.page_content, fallback_doc)
        print(f"Fallback result:")
        print(f"  Category: {fallback_result['category']}")
        print(f"  Confidence: {fallback_result['confidence']:.3f}")
        print(f"  Method: {fallback_result['method']}")
        print(f"  Keywords found: {', '.join(fallback_result.get('key_indicators', []))}")
        
        # Test category validation
        print(f"\n✅ Testing Category Validation:")
        print("-" * 50)
        
        # Test valid categories
        valid_categories = ['contract', 'policy', 'employment', 'privacy_policy']
        print("Valid categories test:")
        for cat in valid_categories:
            is_valid = categorizer.config.is_valid_category(cat)
            cat_name = categorizer.config.get_category_name(cat)
            print(f"  • {cat} -> {is_valid} ({cat_name})")
        
        # Test invalid categories
        invalid_categories = ['invalid_cat', 'nonexistent', '']
        print("\nInvalid categories test:")
        for cat in invalid_categories:
            is_valid = categorizer.config.is_valid_category(cat)
            print(f"  • '{cat}' -> {is_valid}")
        
        # Test cache operations
        print(f"\n💽 Testing Cache Operations:")
        print("-" * 50)
        
        initial_cache_size = len(categorizer.categorization_cache)
        print(f"Initial cache size: {initial_cache_size}")
        
        # Test cache clearing
        if initial_cache_size > 0:
            cache_cleared = categorizer.clear_cache()
            final_cache_size = len(categorizer.categorization_cache)
            print(f"Cache cleared: {cache_cleared}")
            print(f"Final cache size: {final_cache_size}")
        else:
            print("Cache already empty, skipping clear test")
        
        # Display available categories
        print(f"\n🏷️  Available Categories ({len(categorizer.config.LEGAL_CATEGORIES)}):")
        print("-" * 50)
        
        for key, name in categorizer.config.LEGAL_CATEGORIES.items():
            description = categorizer.config.CATEGORY_DESCRIPTIONS.get(key, "")
            keyword_count = len(categorizer.config.CATEGORY_KEYWORDS.get(key, []))
            print(f"• {key}: {name}")
            print(f"  Description: {description[:100]}...")
            print(f"  Keywords available: {keyword_count}")
        
        print(f"\n✅ Document Categorizer testing completed successfully!")
        print(f"📊 Final Statistics:")
        print(f"   Documents processed: {len(test_documents)}")
        print(f"   Categories identified: {len(set(cat['category'] for cat in categorizations))}")
        print(f"   Average confidence: {sum(cat['confidence'] for cat in categorizations) / len(categorizations):.3f}")
        
        # Performance summary
        total_operations = sum(categorizer.categorization_stats.values())
        if total_operations > 0:
            print(f"   Cache hit rate: {categorizer.categorization_stats['cache_hits'] / total_operations * 100:.1f}%")
            print(f"   Success rate: {(total_operations - categorizer.categorization_stats['failed_categorizations']) / total_operations * 100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        logger.exception("Error during categorizer testing")
        
    finally:
        print(f"\n🏁 Testing completed!")