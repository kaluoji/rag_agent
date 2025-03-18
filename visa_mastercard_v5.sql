-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documentation chunks table
create table visa_mastercard_v5 (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,  -- Added content column
    metadata jsonb not null default '{}'::jsonb,  -- Added metadata column
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number)
);

-- Create an index for better vector similarity search performance
-- create index on visa_mastercard_v5 using ivfflat (embedding vector_cosine_ops);

DROP INDEX IF EXISTS visa_mastercard_v5_embedding_idx;
CREATE INDEX visa_mastercard_v5_embedding_hnsw_idx 
  ON visa_mastercard_v5 
  USING hnsw (embedding vector_cosine_ops) 
  WITH (ef_construction = 200);

-- Create an index on metadata for faster filtering
create index idx_visa_mastercard_v5_metadata on visa_mastercard_v5 using gin (metadata);

-- Create a function to search for documentation chunks
create or replace function match_visa_mastercard_v5 (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
-- variable_conflict use_column
begin
  return query
  select
    visa_mastercard_v5.id,  -- Calificado con el nombre de la tabla
    visa_mastercard_v5.url,
    visa_mastercard_v5.chunk_number,
    visa_mastercard_v5.title,
    visa_mastercard_v5.summary,
    visa_mastercard_v5.content,
    visa_mastercard_v5.metadata,
    1 - (visa_mastercard_v5.embedding <=> query_embedding) as similarity
  from visa_mastercard_v5
  --WHERE 1 - (visa_mastercard_v5.embedding <=> query_embedding) >= COALESCE((filter->>'similarity_threshold')::float, 0.0)
  order by visa_mastercard_v5.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Everything above will work for any PostgreSQL database. The below commands are for Supabase security

-- Enable RLS on the table
alter table visa_mastercard_v5 enable row level security;

-- Create a policy that allows anyone to read
create policy "Allow public read access"
  on visa_mastercard_v5
  for select
  to public
  using (true);

-- Crear una función para buscar chunks dentro del mismo cluster para visa_mastercard_v5
CREATE OR REPLACE FUNCTION match_visa_mastercard_v5_by_cluster (
  cluster_id int,
  match_count int default 10
) RETURNS TABLE (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  cluster_similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    visa_mastercard_v5.id,  -- Calificado con el nombre de la tabla
    visa_mastercard_v5.url,
    visa_mastercard_v5.chunk_number,
    visa_mastercard_v5.title,
    visa_mastercard_v5.summary,
    visa_mastercard_v5.content,
    visa_mastercard_v5.metadata,
    (visa_mastercard_v5.metadata->>'cluster_size')::float AS cluster_similarity
  FROM visa_mastercard_v5
  WHERE (visa_mastercard_v5.metadata->>'cluster_id')::int = cluster_id
  ORDER BY visa_mastercard_v5.chunk_number
  LIMIT match_count;
END;
$$;

