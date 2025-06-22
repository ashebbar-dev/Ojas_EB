-- This script sets up your complete database schema for a high-performance,
-- secure, and scalable RAG system using Supabase and pgvector.

-- Step 1: Enable the pgvector extension if it's not already enabled.
-- This gives your database the ability to understand and work with vectors.
CREATE EXTENSION IF NOT EXISTS vector;


-- Step 2: Create the main table to store your chunked documentation.
-- This structure promotes key metadata to dedicated columns for clarity and efficient querying.
CREATE TABLE dementia_chunks (
    id bigserial PRIMARY KEY,
    source_url text NOT NULL,
    page_title text NOT NULL,
    topic_heading text NOT NULL,
    content text NOT NULL,
    processing_info jsonb, -- Stores flexible data like {"contents": "Part 1 of 2"}
    embedding vector(1024), -- IMPORTANT: Set for Voyage AI's 1024 dimensions
    created_at timestamp with time zone DEFAULT timezone('utc'::text, now()) NOT NULL
);


-- Step 3: Create performance indexes. This is critical for speed.
-- This first index (IVFFLAT) is for the vector search. It groups similar vectors
-- together, so search operations don't have to scan every single row.
CREATE INDEX ON dementia_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- This is ENHANCEMENT #2.
-- These next indexes (B-Tree) are for traditional filtering on metadata columns.
-- They make the `WHERE` clause in your search function instantaneous.
CREATE INDEX ON dementia_chunks (source_url);
CREATE INDEX ON dementia_chunks (page_title);


-- Step 4: Create a function to perform the search.
-- This function combines fast metadata pre-filtering with vector similarity search.
CREATE OR REPLACE FUNCTION match_dementia_chunks (
  query_embedding vector(1024),
  match_count int,
  p_source_url text DEFAULT NULL, -- Optional filter for a specific URL
  p_page_title text DEFAULT NULL  -- Optional filter for a specific page title
) RETURNS TABLE (
  id bigint,
  source_url text,
  page_title text,
  topic_heading text,
  content text,
  processing_info jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    dc.id,
    dc.source_url,
    dc.page_title,
    dc.topic_heading,
    dc.content,
    dc.processing_info,
    1 - (dc.embedding <=> query_embedding) AS similarity
  FROM dementia_chunks AS dc
  WHERE
    (p_source_url IS NULL OR dc.source_url = p_source_url) AND
    (p_page_title IS NULL OR dc.page_title = p_page_title)
  ORDER BY dc.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;


-- Step 5: Set up Row Level Security (RLS) for data protection.
-- This is a crucial security step for any public-facing application.
ALTER TABLE dementia_chunks ENABLE ROW LEVEL SECURITY;

-- This policy allows anyone (e.g., your public web app) to read data from the table.
CREATE POLICY "Allow public read-only access"
  ON dementia_chunks
  FOR SELECT
  TO public
  USING (true);

-- This policy allows your backend script (using the SERVICE_ROLE_KEY) to
-- perform any action (insert, update, delete), bypassing the RLS above.
CREATE POLICY "Allow full access for service role"
  ON dementia_chunks
  FOR ALL
  TO service_role
  USING (true);