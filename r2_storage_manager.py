"""
Cloudflare R2 Storage Manager for Legal Document AI System
Handles document upload, download, listing, and deletion operations with R2 bucket
"""

import os
import io
import logging
from typing import List, Dict, Any, Optional, BinaryIO
from datetime import datetime
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config as BotoConfig

from config import Config

logger = logging.getLogger(__name__)

class R2StorageManager:
    """Manages Cloudflare R2 storage operations for document handling"""
    
    def __init__(self):
        self.config = Config()
        self._validate_r2_config()
        self._setup_r2_client()
    
    def _validate_r2_config(self):
        """Validate R2 configuration settings"""
        if not self.config.USE_R2_STORAGE:
            logger.warning("R2 storage is disabled in configuration")
            return
        
        required_vars = [
            self.config.R2_ACCESS_KEY_ID,
            self.config.R2_SECRET_ACCESS_KEY,
            self.config.R2_BUCKET_NAME,
            self.config.R2_ENDPOINT_URL
        ]
        
        if not all(required_vars):
            raise ValueError("Missing required R2 configuration. Please check your .env file.")
        
        logger.info(f"R2 configuration validated for bucket: {self.config.R2_BUCKET_NAME}")
    
    def _setup_r2_client(self):
        """Initialize the R2/S3 client"""
        if not self.config.USE_R2_STORAGE:
            self.client = None
            return
        
        try:
            # Configure boto3 for Cloudflare R2
            self.client = boto3.client(
                's3',
                endpoint_url=self.config.R2_ENDPOINT_URL,
                aws_access_key_id=self.config.R2_ACCESS_KEY_ID,
                aws_secret_access_key=self.config.R2_SECRET_ACCESS_KEY,
                config=BotoConfig(
                    region_name='auto',  # R2 uses 'auto' region
                    retries={'max_attempts': 3, 'mode': 'standard'},
                    max_pool_connections=50
                )
            )
            
            # Test connection
            self._test_connection()
            logger.info("R2 client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize R2 client: {e}")
            raise
    
    def _test_connection(self):
        """Test R2 connection and bucket access"""
        if not self.client:
            return False
        
        try:
            # Try to list objects (this will verify both connection and bucket access)
            response = self.client.list_objects_v2(
                Bucket=self.config.R2_BUCKET_NAME,
                MaxKeys=1
            )
            logger.info("R2 connection test successful")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                raise ValueError(f"R2 bucket '{self.config.R2_BUCKET_NAME}' does not exist")
            elif error_code == 'AccessDenied':
                raise ValueError("Access denied to R2 bucket. Please check your credentials and permissions")
            else:
                raise ValueError(f"R2 connection failed: {e}")
        except Exception as e:
            raise ValueError(f"R2 connection test failed: {e}")
    
    def _get_document_key(self, filename: str) -> str:
        """Generate R2 key for a document"""
        # Clean filename
        clean_filename = Path(filename).name
        return f"{self.config.R2_DOCUMENTS_PREFIX}{clean_filename}"
    
    def _get_timestamped_key(self, filename: str) -> str:
        """Generate timestamped R2 key for a document to avoid conflicts"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = Path(filename).name
        name, ext = os.path.splitext(clean_filename)
        timestamped_filename = f"{name}_{timestamp}{ext}"
        return f"{self.config.R2_DOCUMENTS_PREFIX}{timestamped_filename}"
    
    def upload_file(self, local_file_path: str, r2_key: str = None, 
                   add_timestamp: bool = False, metadata: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Upload a local file to R2 storage
        
        Args:
            local_file_path: Path to the local file
            r2_key: Custom R2 key (optional, will generate if not provided)
            add_timestamp: Whether to add timestamp to avoid filename conflicts
            metadata: Additional metadata to store with the file
            
        Returns:
            Dict containing upload result information
        """
        if not self.config.USE_R2_STORAGE:
            raise ValueError("R2 storage is disabled")
        
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file not found: {local_file_path}")
        
        try:
            # Generate R2 key
            if not r2_key:
                if add_timestamp:
                    r2_key = self._get_timestamped_key(local_file_path)
                else:
                    r2_key = self._get_document_key(local_file_path)
            
            # Prepare metadata
            file_metadata = {
                'original_filename': os.path.basename(local_file_path),
                'upload_timestamp': datetime.now().isoformat(),
                'file_size': str(os.path.getsize(local_file_path)),
                'content_type': self._get_content_type(local_file_path)
            }
            
            if metadata:
                file_metadata.update(metadata)
            
            # Upload file
            with open(local_file_path, 'rb') as file:
                self.client.upload_fileobj(
                    file,
                    self.config.R2_BUCKET_NAME,
                    r2_key,
                    ExtraArgs={
                        'Metadata': file_metadata,
                        'ContentType': file_metadata['content_type']
                    }
                )
            
            result = {
                'success': True,
                'r2_key': r2_key,
                'bucket': self.config.R2_BUCKET_NAME,
                'original_filename': os.path.basename(local_file_path),
                'file_size': os.path.getsize(local_file_path),
                'upload_timestamp': datetime.now().isoformat(),
                'url': f"{self.config.R2_ENDPOINT_URL}/{self.config.R2_BUCKET_NAME}/{r2_key}"
            }
            
            logger.info(f"File uploaded successfully: {local_file_path} -> {r2_key}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to upload file {local_file_path}: {e}")
            raise
    
    def upload_file_content(self, content: bytes, filename: str, 
                          add_timestamp: bool = False, metadata: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Upload file content directly to R2 storage
        
        Args:
            content: File content as bytes
            filename: Original filename
            add_timestamp: Whether to add timestamp to avoid filename conflicts
            metadata: Additional metadata to store with the file
            
        Returns:
            Dict containing upload result information
        """
        if not self.config.USE_R2_STORAGE:
            raise ValueError("R2 storage is disabled")
        
        try:
            # Generate R2 key
            if add_timestamp:
                r2_key = self._get_timestamped_key(filename)
            else:
                r2_key = self._get_document_key(filename)
            
            # Prepare metadata
            file_metadata = {
                'original_filename': filename,
                'upload_timestamp': datetime.now().isoformat(),
                'file_size': str(len(content)),
                'content_type': self._get_content_type(filename)
            }
            
            if metadata:
                file_metadata.update(metadata)
            
            # Upload content
            file_obj = io.BytesIO(content)
            self.client.upload_fileobj(
                file_obj,
                self.config.R2_BUCKET_NAME,
                r2_key,
                ExtraArgs={
                    'Metadata': file_metadata,
                    'ContentType': file_metadata['content_type']
                }
            )
            
            result = {
                'success': True,
                'r2_key': r2_key,
                'bucket': self.config.R2_BUCKET_NAME,
                'original_filename': filename,
                'file_size': len(content),
                'upload_timestamp': datetime.now().isoformat(),
                'url': f"{self.config.R2_ENDPOINT_URL}/{self.config.R2_BUCKET_NAME}/{r2_key}"
            }
            
            logger.info(f"Content uploaded successfully: {filename} -> {r2_key}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to upload content for {filename}: {e}")
            raise
    
    def download_file(self, r2_key: str, local_file_path: str = None) -> Dict[str, Any]:
        """
        Download a file from R2 storage
        
        Args:
            r2_key: R2 key of the file to download
            local_file_path: Local path to save the file (optional)
            
        Returns:
            Dict containing download result information
        """
        if not self.config.USE_R2_STORAGE:
            raise ValueError("R2 storage is disabled")
        
        try:
            # Check if file exists
            if not self.file_exists(r2_key):
                raise FileNotFoundError(f"File not found in R2: {r2_key}")
            
            # Generate local file path if not provided
            if not local_file_path:
                filename = os.path.basename(r2_key)
                local_file_path = os.path.join(self.config.UPLOAD_FOLDER, filename)
            
            # Ensure directory exists
            local_dir = os.path.dirname(local_file_path)
            if local_dir:  # Only create if directory path is not empty
                os.makedirs(local_dir, exist_ok=True)
            
            # Download file
            self.client.download_file(
                self.config.R2_BUCKET_NAME,
                r2_key,
                local_file_path
            )
            
            # Get file metadata
            metadata = self.get_file_metadata(r2_key)
            
            result = {
                'success': True,
                'r2_key': r2_key,
                'local_path': local_file_path,
                'file_size': os.path.getsize(local_file_path),
                'download_timestamp': datetime.now().isoformat(),
                'metadata': metadata
            }
            
            logger.info(f"File downloaded successfully: {r2_key} -> {local_file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to download file {r2_key}: {e}")
            raise
    
    def download_file_content(self, r2_key: str) -> bytes:
        """
        Download file content from R2 storage as bytes
        
        Args:
            r2_key: R2 key of the file to download
            
        Returns:
            File content as bytes
        """
        if not self.config.USE_R2_STORAGE:
            raise ValueError("R2 storage is disabled")
        
        try:
            # Check if file exists
            if not self.file_exists(r2_key):
                raise FileNotFoundError(f"File not found in R2: {r2_key}")
            
            # Download file content
            response = self.client.get_object(
                Bucket=self.config.R2_BUCKET_NAME,
                Key=r2_key
            )
            
            content = response['Body'].read()
            logger.info(f"File content downloaded successfully: {r2_key} ({len(content)} bytes)")
            return content
            
        except Exception as e:
            logger.error(f"Failed to download content for {r2_key}: {e}")
            raise
    
    def list_documents(self, prefix: str = None, max_keys: int = 1000) -> List[Dict[str, Any]]:
        """
        List documents in R2 storage
        
        Args:
            prefix: Filter documents by prefix (optional)
            max_keys: Maximum number of keys to return
            
        Returns:
            List of document information dictionaries
        """
        if not self.config.USE_R2_STORAGE:
            raise ValueError("R2 storage is disabled")
        
        try:
            # Use default document prefix if none provided
            if prefix is None:
                prefix = self.config.R2_DOCUMENTS_PREFIX
            
            response = self.client.list_objects_v2(
                Bucket=self.config.R2_BUCKET_NAME,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            documents = []
            for obj in response.get('Contents', []):
                # Get additional metadata
                metadata = self.get_file_metadata(obj['Key'])
                
                doc_info = {
                    'r2_key': obj['Key'],
                    'filename': os.path.basename(obj['Key']),
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'etag': obj['ETag'].strip('"'),
                    'metadata': metadata,
                    'url': f"{self.config.R2_ENDPOINT_URL}/{self.config.R2_BUCKET_NAME}/{obj['Key']}"
                }
                documents.append(doc_info)
            
            logger.info(f"Listed {len(documents)} documents from R2 with prefix: {prefix}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise
    
    def delete_file(self, r2_key: str) -> bool:
        """
        Delete a file from R2 storage
        
        Args:
            r2_key: R2 key of the file to delete
            
        Returns:
            True if deletion was successful
        """
        if not self.config.USE_R2_STORAGE:
            raise ValueError("R2 storage is disabled")
        
        try:
            self.client.delete_object(
                Bucket=self.config.R2_BUCKET_NAME,
                Key=r2_key
            )
            
            logger.info(f"File deleted successfully: {r2_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {r2_key}: {e}")
            return False
    
    def file_exists(self, r2_key: str) -> bool:
        """
        Check if a file exists in R2 storage
        
        Args:
            r2_key: R2 key to check
            
        Returns:
            True if file exists
        """
        if not self.config.USE_R2_STORAGE:
            return False
        
        try:
            self.client.head_object(
                Bucket=self.config.R2_BUCKET_NAME,
                Key=r2_key
            )
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise
    
    def get_file_metadata(self, r2_key: str) -> Dict[str, Any]:
        """
        Get metadata for a file in R2 storage
        
        Args:
            r2_key: R2 key of the file
            
        Returns:
            Dictionary containing file metadata
        """
        if not self.config.USE_R2_STORAGE:
            raise ValueError("R2 storage is disabled")
        
        try:
            response = self.client.head_object(
                Bucket=self.config.R2_BUCKET_NAME,
                Key=r2_key
            )
            
            return {
                'content_type': response.get('ContentType', ''),
                'content_length': response.get('ContentLength', 0),
                'last_modified': response.get('LastModified', '').isoformat() if response.get('LastModified') else '',
                'etag': response.get('ETag', '').strip('"'),
                'metadata': response.get('Metadata', {}),
                'server_side_encryption': response.get('ServerSideEncryption', ''),
            }
            
        except Exception as e:
            logger.error(f"Failed to get metadata for {r2_key}: {e}")
            raise
    
    def generate_presigned_url(self, r2_key: str, expiration: int = 3600, method: str = 'get_object') -> str:
        """
        Generate a presigned URL for a file in R2 storage
        
        Args:
            r2_key: R2 key of the file
            expiration: URL expiration time in seconds (default: 1 hour)
            method: HTTP method ('get_object' or 'put_object')
            
        Returns:
            Presigned URL string
        """
        if not self.config.USE_R2_STORAGE:
            raise ValueError("R2 storage is disabled")
        
        try:
            url = self.client.generate_presigned_url(
                method,
                Params={'Bucket': self.config.R2_BUCKET_NAME, 'Key': r2_key},
                ExpiresIn=expiration
            )
            
            logger.info(f"Generated presigned URL for {r2_key} (expires in {expiration}s)")
            return url
            
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {r2_key}: {e}")
            raise
    
    def _get_content_type(self, filename: str) -> str:
        """Get MIME content type based on file extension"""
        extension = Path(filename).suffix.lower()
        
        content_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.txt': 'text/plain',
        }
        
        return content_types.get(extension, 'application/octet-stream')
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for documents"""
        if not self.config.USE_R2_STORAGE:
            return {'storage_enabled': False}
        
        try:
            documents = self.list_documents()
            
            total_size = sum(doc['size'] for doc in documents)
            file_types = {}
            
            for doc in documents:
                ext = Path(doc['filename']).suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
            
            return {
                'storage_enabled': True,
                'total_documents': len(documents),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'file_types': file_types,
                'bucket_name': self.config.R2_BUCKET_NAME,
                'documents_prefix': self.config.R2_DOCUMENTS_PREFIX
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {'storage_enabled': True, 'error': str(e)}

# Example usage and testing
if __name__ == "__main__":
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize R2 storage manager
        storage = R2StorageManager()
        
        print("🌩️  Cloudflare R2 Storage Manager")
        print("=" * 50)
        
        # Test connection
        print("Testing R2 connection...")
        
        # Get storage statistics
        stats = storage.get_storage_stats()
        print(f"\n📊 Storage Statistics:")
        print(json.dumps(stats, indent=2))
        
        # List existing documents
        documents = storage.list_documents()
        print(f"\n📁 Documents in Storage: {len(documents)}")
        
        for doc in documents[:5]:  # Show first 5
            print(f"  • {doc['filename']} ({doc['size']} bytes)")
        
        print("\n✅ R2 Storage Manager initialized successfully!")
        print("\nAvailable methods:")
        print("- upload_file(local_path, r2_key=None, add_timestamp=False)")
        print("- upload_file_content(content, filename, add_timestamp=False)")
        print("- download_file(r2_key, local_path=None)")
        print("- download_file_content(r2_key)")
        print("- list_documents(prefix=None)")
        print("- delete_file(r2_key)")
        print("- file_exists(r2_key)")
        print("- generate_presigned_url(r2_key, expiration=3600)")
        
    except Exception as e:
        print(f"❌ Error initializing R2 Storage Manager: {e}")
        logging.exception("R2 Storage Manager initialization failed")