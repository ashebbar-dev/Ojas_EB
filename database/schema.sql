-- Function 1: Simple Hybrid Search (Keyword-Dominant)
CREATE OR REPLACE FUNCTION simple_hybrid_search (
  query_embedding vector(1024),
  keyword text,
  match_count int,
  similarity_threshold float DEFAULT 0.6
) RETURNS TABLE (
  id bigint,
  content text,
  source_url text,
  page_title text,
  topic_heading text,
  similarity float,
  search_type text
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  WITH vector_matches AS (
    SELECT
      dc.id,
      dc.content,
      dc.source_url,
      dc.page_title,
      dc.topic_heading,
      (1 - (dc.embedding <=> query_embedding)) AS similarity,
      'vector'::text AS search_type
    FROM dementia_chunks dc
    WHERE (1 - (dc.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY similarity DESC
    LIMIT match_count
  ),
  keyword_matches AS (
    SELECT
      dc.id,
      dc.content,
      dc.source_url,
      dc.page_title,
      dc.topic_heading,
      -- Weight keyword matches higher to make them dominant
      (ts_rank(to_tsvector('english', dc.content), websearch_to_tsquery('english', keyword)) * 1.5) AS similarity,
      'keyword'::text AS search_type
    FROM dementia_chunks dc
    WHERE keyword IS NOT NULL
      AND keyword <> ''
      AND to_tsvector('english', dc.content) @@ websearch_to_tsquery('english', keyword)
    ORDER BY similarity DESC
    LIMIT match_count
  )
  SELECT * FROM keyword_matches  -- Keyword matches first (dominant)
  UNION ALL
  SELECT * FROM vector_matches
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;

-- Function 2: Title-Filtered Hybrid Search with Reranking
CREATE OR REPLACE FUNCTION title_filtered_search (
  query_embedding vector(1024),
  keyword text,
  match_count int,
  title_match_count int DEFAULT 10,
  similarity_threshold float DEFAULT 0.6
) RETURNS TABLE (
  id bigint,
  content text,
  source_url text,
  page_title text,
  topic_heading text,
  similarity float,
  search_type text
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  WITH top_matching_titles AS (
    -- Step 1: Find top matching titles based on keyword
    SELECT DISTINCT 
      dc.page_title,
      ts_rank(to_tsvector('english', dc.page_title), websearch_to_tsquery('english', keyword)) AS title_rank
    FROM dementia_chunks dc
    WHERE keyword IS NOT NULL
      AND keyword <> ''
      AND to_tsvector('english', dc.page_title) @@ websearch_to_tsquery('english', keyword)
    ORDER BY title_rank DESC
    LIMIT title_match_count
  ),
  chunks_from_top_titles AS (
    -- Step 2: Get all chunks from top matching titles
    SELECT
      dc.id,
      dc.content,
      dc.source_url,
      dc.page_title,
      dc.topic_heading,
      dc.embedding
    FROM dementia_chunks dc
    INNER JOIN top_matching_titles tmt ON dc.page_title = tmt.page_title
  ),
  reranked_chunks AS (
    -- Step 3: Rerank chunks by query relevance
    SELECT
      cftt.id,
      cftt.content,
      cftt.source_url,
      cftt.page_title,
      cftt.topic_heading,
      (1 - (cftt.embedding <=> query_embedding)) AS similarity,
      'title_filtered'::text AS search_type
    FROM chunks_from_top_titles cftt
    WHERE (1 - (cftt.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY similarity DESC
  )
  SELECT * FROM reranked_chunks
  LIMIT match_count;
END;
$$;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS dementia_chunks_content_gin_idx 
  ON dementia_chunks USING GIN (to_tsvector('english', content));

CREATE INDEX IF NOT EXISTS dementia_chunks_title_gin_idx 
  ON dementia_chunks USING GIN (to_tsvector('english', page_title));