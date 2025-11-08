import os
import psycopg2
from psycopg2 import sql
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL Connection Setup
def init_postgres():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB", "face_metadata"),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", "admin123"),
    )
    cur = conn.cursor()

    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS person_metadata (
            id SERIAL PRIMARY KEY,
            person_id VARCHAR(255) UNIQUE,
            name VARCHAR(255),
            group_id VARCHAR(255),
            image_paths TEXT[],
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    print("✅ PostgreSQL connected and table ready.")
    return conn, cur


# Milvus Connection Setup
def init_milvus():
    connections.connect(
        alias="default",
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530")
    )

    collection_name = "face_embeddings"

    fields = [
        FieldSchema(name="person_id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),  # InsightFace uses 512-dim
    ]

    schema = CollectionSchema(fields=fields, description="Face embedding collection")

    if collection_name not in [c.name for c in Collection.list_collections()]:
        Collection(name=collection_name, schema=schema)
        print("✅ Milvus collection created.")
    else:
        print("ℹ️ Milvus collection already exists.")

    return Collection(collection_name)
