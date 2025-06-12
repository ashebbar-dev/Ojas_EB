# extract_pdf_text.py

import fitz  # PyMuPDF library
import sys
import os
import argparse

def extract_text_from_pages(pdf_path, pages_specifier):
    """
    Extracts text from specified pages of a PDF file.

    Args:
        pdf_path (str): The full path to the PDF file.
        pages_specifier (list of int or str): A list of 1-based page numbers
                                               or the string 'all'.

    Returns:
        str: The concatenated text from the specified pages, or None on error.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' was not found.", file=sys.stderr)
        return None

    try:
        with fitz.open(pdf_path) as doc:
            if doc.is_encrypted and not doc.authenticate(""):
                print(f"Error: PDF '{pdf_path}' is encrypted.", file=sys.stderr)
                return None
            
            total_pages = doc.page_count
            pages_to_extract = []

            if isinstance(pages_specifier, str) and pages_specifier.lower() == 'all':
                pages_to_extract = range(1, total_pages + 1)
            else:
                pages_to_extract = sorted(pages_specifier)
            
            extracted_texts = []
            
            for page_num in pages_to_extract:
                if 1 <= page_num <= total_pages:
                    page = doc.load_page(page_num - 1)
                    header = f"\n--- TEXT FROM PAGE {page_num} ---\n"
                    text = page.get_text("text")
                    extracted_texts.append(header)
                    extracted_texts.append(text)
                else:
                    print(f"Warning: Page {page_num} is invalid. PDF has {total_pages} pages. Skipping.", file=sys.stderr)
            
            if not extracted_texts:
                return "" # Return empty string if valid pages had no text
            
            return "".join(extracted_texts)

    except Exception as e:
        print(f"An error occurred while processing the PDF: {e}", file=sys.stderr)
        return None

def write_to_file(filepath, content):
    """Writes content to a specified file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except IOError as e:
        print(f"Error writing to file '{filepath}': {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Extracts text from pages of a PDF and prints to console or saves to a file.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example Usage:
-------------
# Extract from page 2 and print to console
python extract_pdf_text.py report.pdf 2

# Extract from pages 1, 5, 10 and save to a file
python extract_pdf_text.py document.pdf 1 5 10 -o extracted_content.txt

# Extract from ALL pages and save to a file
python extract_pdf_text.py manual.pdf all --output full_manual.txt
"""
    )
    
    parser.add_argument("pdf_file", type=str, help="The path to the PDF file.")
    parser.add_argument("pages", type=str, nargs='+', help="Page numbers (e.g., 1 3 5) or 'all'.")
    parser.add_argument("-o", "--output", type=str, help="Optional: Path to the output text file.")
    
    args = parser.parse_args()
    
    pages_to_process = []
    if len(args.pages) == 1 and args.pages[0].lower() == 'all':
        pages_to_process = 'all'
    else:
        try:
            pages_to_process = [int(p) for p in args.pages]
        except ValueError:
            print("Error: Pages must be integers or the single keyword 'all'.", file=sys.stderr)
            sys.exit(1)

    extracted_text = extract_text_from_pages(args.pdf_file, pages_to_process)
    
    # If extraction failed (returned None) or no text was found, exit.
    if extracted_text is None:
        sys.exit(1) # Exit with an error code
    if not extracted_text.strip():
        print("No text was extracted. The specified pages might be empty or contain only images.")
        sys.exit(0)

    # --- Output Handling ---
    if args.output:
        # User specified an output file
        print(f"Attempting to save extracted text to '{args.output}'...")
        if write_to_file(args.output, extracted_text):
            print("Successfully saved.")
    else:
        # No output file, print to console
        print(extracted_text)

if __name__ == "__main__":
    main()