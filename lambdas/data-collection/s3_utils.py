"""
S3 utility functions for YouTube ingestion Lambda.
Handles all S3 operations for storing video data.
"""

import boto3
import json
import logging
from datetime import datetime
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# S3 client
s3_client = boto3.client('s3')


def save_to_s3(channel_name, data, bucket_name):
    """
    Save channel data to S3 as JSON.
    
    Args:
        channel_name (str): Name of the channel
        data (list): List of video data dictionaries
        bucket_name (str): S3 bucket name
        
    Returns:
        str: S3 key of the uploaded object
        
    Raises:
        ClientError: If S3 upload fails
    """
    try:
        timestamp = datetime.now().isoformat()
        s3_key = f"youtube-data/{channel_name.lower()}/{timestamp.replace(':', '-')}.json"
        
        payload = {
            "channel": channel_name,
            "fetched_at": timestamp,
            "videos": data
        }
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=json.dumps(payload, ensure_ascii=False),
            ContentType='application/json'
        )
        
        logger.info(f"Uploaded to S3: s3://{bucket_name}/{s3_key}")
        return s3_key
        
    except ClientError as e:
        logger.error(f"Error uploading to S3: {e}", exc_info=True)
        raise


def list_s3_videos(bucket_name, channel_name=None, limit=100):
    """
    List videos stored in S3 for a specific channel or all channels.
    
    Args:
        bucket_name (str): S3 bucket name
        channel_name (str, optional): Specific channel to filter by
        limit (int): Maximum number of objects to return
        
    Returns:
        list: List of S3 objects with metadata
    """
    try:
        prefix = f"youtube-data/"
        if channel_name:
            prefix = f"youtube-data/{channel_name.lower()}/"
        
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            MaxKeys=limit
        )
        
        objects = response.get('Contents', [])
        logger.info(f"Listed {len(objects)} objects in S3 (prefix: {prefix})")
        
        return objects
        
    except ClientError as e:
        logger.error(f"Error listing S3 objects: {e}", exc_info=True)
        raise


def get_s3_object(bucket_name, key):
    """
    Retrieve a specific object from S3.
    
    Args:
        bucket_name (str): S3 bucket name
        key (str): S3 object key
        
    Returns:
        dict: Parsed JSON content of the object
        
    Raises:
        ClientError: If S3 get fails
    """
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        content = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"Retrieved object from S3: s3://{bucket_name}/{key}")
        return content
        
    except ClientError as e:
        logger.error(f"Error retrieving object from S3: {e}", exc_info=True)
        raise


def delete_s3_object(bucket_name, key):
    """
    Delete an object from S3.
    
    Args:
        bucket_name (str): S3 bucket name
        key (str): S3 object key
        
    Returns:
        bool: True if deletion successful
    """
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=key)
        logger.info(f"Deleted object from S3: s3://{bucket_name}/{key}")
        return True
        
    except ClientError as e:
        logger.error(f"Error deleting object from S3: {e}", exc_info=True)
        raise
