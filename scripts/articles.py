
"""
articles.py - DynamoDB Articles Table Creation

Creates the 'articles' table in DynamoDB with the following schema:
  - article_id (String, PK)
  - cluster_id (Number, foreign key to narrative-clusters)
  - title (String)
  - overview (String)
  - body (String)
  - weekNumber (String)

 Note: In DynamoDB, you only define the primary key and indexed fields in the table schema.
 Other fields (like title, overview, body, weekNumber) are added when you insert items.
 Example of adding an article with all fields:

	 client.put_item(
		 TableName="articles",
		 Item={
			 "article_id": {"S": "123"},
			 "cluster_id": {"N": "1"},
			 "title": {"S": "My Title"},
			 "overview": {"S": "Short summary"},
			 "body": {"S": "Full article text"},
			 "weekNumber": {"S": "2026-W13"}
		 }
	 )

You do not need to change the table definition to store these fields. Just include them in your data when you write to DynamoDB.

Requirements:
  pip install boto3 python-dotenv

Run:
  python scripts/articles.py
"""

import os
import sys
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

load_dotenv()

REGION = os.getenv("AWS_REGION", "us-east-2")
ARTICLES_TABLE = "articles"
NARRATIVES_TABLE = "narrative-clusters"

def get_client():
	return boto3.client("dynamodb", region_name=REGION)

def cluster_exists(client, cluster_id: int) -> bool:
	"""Return True if cluster_id exists as a PK in narrative-clusters."""
	response = client.get_item(
		TableName=NARRATIVES_TABLE,
		Key={"cluster_id": {"N": str(cluster_id)}},
		ProjectionExpression="cluster_id",
	)
	return "Item" in response

def create_articles_table():
	print(f"Creating table '{ARTICLES_TABLE}'...")
	client = get_client()
	try:
		client.create_table(
			TableName=ARTICLES_TABLE,
			KeySchema=[
				{"AttributeName": "article_id", "KeyType": "HASH"},
			],
			AttributeDefinitions=[
				{"AttributeName": "article_id", "AttributeType": "S"},
			],
			BillingMode="PAY_PER_REQUEST",
		)
		print(f"  Table creation initiated. Waiting for ACTIVE status...")
		_wait_for_table(client, ARTICLES_TABLE)
		print(f"  Table '{ARTICLES_TABLE}' is ACTIVE")

	except ClientError as e:
		if e.response["Error"]["Code"] == "ResourceInUseException":
			print(f"  Table '{ARTICLES_TABLE}' already exists")
		else:
			raise

def insert_article(article_id: str, cluster_id: int, title: str, overview: str, body: str, week_number: str):
	client = get_client()
	if not cluster_exists(client, cluster_id):
		raise ValueError(f"cluster_id={cluster_id} not found in '{NARRATIVES_TABLE}'")
	client.put_item(
		TableName=ARTICLES_TABLE,
		Item={
			"article_id": {"S": article_id},
			"cluster_id": {"N": str(cluster_id)},
			"title": {"S": title},
			"overview": {"S": overview},
			"body": {"S": body},
			"weekNumber": {"S": week_number},
		}
	)

def _wait_for_table(client, table_name, timeout=120):
	import time
	elapsed = 0
	while elapsed < timeout:
		desc = client.describe_table(TableName=table_name)
		status = desc["Table"]["TableStatus"]
		if status == "ACTIVE":
			return
		print(f"    {table_name}: {status} ({elapsed}s)")
		time.sleep(5)
		elapsed += 5
	raise TimeoutError(f"Table '{table_name}' did not become ACTIVE within {timeout}s")

if __name__ == "__main__":
	#REPLACE WITH YOUR OWN ITEM
	insert_article(
		article_id="test-article-1",
		cluster_id=1,
		title="Test Article Title",
		overview="This is a test overview.",
		body="This is the body of the test article.",
		week_number="2026-W13",
	)
