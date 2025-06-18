import json
import tiktoken
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
MIN_TOKENS = 300
MAX_TOKENS = 600
ENCODING_NAME = "cl100k_base"

def get_tokenizer():
    """Initializes and returns the tiktoken tokenizer."""
    try:
        return tiktoken.get_encoding(ENCODING_NAME)
    except Exception as e:
        print(f"Error getting tokenizer: {e}")
        return tiktoken.encoding_for_model("gpt-3.5-turbo")

def refine_chunks(input_file: str, output_file: str):
    """
    Loads structural chunks, refines them to be within a specific token range,
    and adds a 'contents' field to the metadata to describe the processing.
    """
    print(f"[*] Loading chunks from '{input_file}'...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            structural_chunks = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[!] Error loading file: {e}")
        return

    tokenizer = get_tokenizer()
    final_chunks = []
    
    merging_buffer_text = []
    merging_buffer_headings = []
    buffer_metadata_base = {}

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING_NAME,
        chunk_size=MAX_TOKENS,
        chunk_overlap=50
    )

    print(f"[*] Refining {len(structural_chunks)} structural chunks...")

    def flush_merging_buffer():
        nonlocal merging_buffer_text, merging_buffer_headings, buffer_metadata_base
        if merging_buffer_text:
            merged_text = ' '.join(merging_buffer_text)
            new_metadata = buffer_metadata_base.copy()
            new_metadata['contents'] = merging_buffer_headings
            
            final_chunks.append({
                'text': merged_text,
                'metadata': new_metadata
            })
            merging_buffer_text = []
            merging_buffer_headings = []
            buffer_metadata_base = {}

    for chunk in structural_chunks:
        token_count = len(tokenizer.encode(chunk['text']))
        
        if token_count > MAX_TOKENS:
            flush_merging_buffer()
            sub_chunks_text = text_splitter.split_text(chunk['text'])
            num_sub_chunks = len(sub_chunks_text)
            for i, sub_chunk_text in enumerate(sub_chunks_text):
                new_metadata = chunk['metadata'].copy()
                new_metadata['contents'] = f"Part {i + 1} of {num_sub_chunks}"
                final_chunks.append({'text': sub_chunk_text, 'metadata': new_metadata})

        elif token_count >= MIN_TOKENS:
            flush_merging_buffer()
            new_metadata = chunk['metadata'].copy()
            new_metadata['contents'] = None 
            chunk['metadata'] = new_metadata
            final_chunks.append(chunk)

        else: # Chunk is TOO SMALL
            # **THE FIX**: Check for overshoot before adding to the buffer
            if merging_buffer_text:
                # Predict the size of the buffer IF we add the new chunk
                predicted_buffer_text = ' '.join(merging_buffer_text) + ' ' + chunk['text']
                predicted_token_count = len(tokenizer.encode(predicted_buffer_text))
                
                if predicted_token_count > MAX_TOKENS:
                    # The new chunk would make the buffer too big. Flush the existing buffer first.
                    flush_merging_buffer()
            
            # Now, proceed with adding the small chunk to the (potentially new) buffer
            if not merging_buffer_text:
                buffer_metadata_base = chunk['metadata']
            
            merging_buffer_text.append(chunk['text'])
            merging_buffer_headings.append(chunk['metadata']['topic_heading'])
            
            buffer_text = ' '.join(merging_buffer_text)
            if len(tokenizer.encode(buffer_text)) >= MIN_TOKENS:
                flush_merging_buffer()

    flush_merging_buffer()

    print(f"[*] Refinement complete. Original chunks: {len(structural_chunks)}, Final chunks: {len(final_chunks)}")
    
    print(f"[*] Saving {len(final_chunks)} refined chunks to '{output_file}'...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_chunks, f, indent=2, ensure_ascii=False)
        print(f"\n[***] All done! The final knowledge base is saved in '{output_file}'. [***]")
    except IOError as e:
        print(f"[!] Error writing to file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Refine structural chunks and add a 'contents' metadata field.")
    parser.add_argument("input_file", help="The input JSON file from the chunker.")
    parser.add_argument("-o", "--output", default="final_knowledge_base.json", help="The output file for refined chunks.")
    args = parser.parse_args()
    refine_chunks(args.input_file, args.output)

if __name__ == "__main__":
    main()