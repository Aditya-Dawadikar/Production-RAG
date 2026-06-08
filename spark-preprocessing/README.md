# Spark Preprocessing for Data Chunking

This folder is responsible for generating chunks from the wiki dataset.

`preprocess_wiki.py` uses the Dockerfile to run a sample script on local to test the chunking logic

`process_wiki_s3.py` is meant for running the same logic on EC2 instance that already has Spark installed. The chunking data is read from and written back to S3.
