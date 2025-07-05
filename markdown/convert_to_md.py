from markitdown import MarkItDown
import os

# --- Configuration ---
# Use a raw string (r"...") to handle Windows paths correctly
PDF_INPUT_PATH = r"D:\ai_assistant\VGU_AI_Assistant\Data_rag\20250120 Bachelor Admission Regulation 2025_VN_Hanh draft (1).pdf"
MARKDOWN_OUTPUT_PATH = "VGU_Regulations.md"

def convert_pdf_to_markdown(pdf_path, md_path):
    """
    Converts a PDF file to a clean Markdown file using the corrected
    method for the markitdown library.
    """
    print(f"Converting '{pdf_path}' to Markdown...")
    
    # Check if the input file exists
    if not os.path.exists(pdf_path):
        print(f"🛑 ERROR: Input PDF file not found at '{pdf_path}'")
        return False

    try:
        # Initialize the converter
        md = MarkItDown()
        
        # Call convert() with only the input path. It returns a result object.
        result_object = md.convert(pdf_path)
        
        # --- THE FIX IS HERE ---
        # The result object itself can be converted to a string to get the content.
        # We explicitly convert the result object to a string before writing.
        markdown_content = str(result_object)
        
        # Now, we write the extracted string content to our output file.
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        # --- END OF FIX ---

        print(f"✅ Successfully converted and saved to '{md_path}'")
        return True
        
    except Exception as e:
        print(f"An error occurred during Markdown conversion: {e}")
        return False

if __name__ == "__main__":
    convert_pdf_to_markdown(PDF_INPUT_PATH, MARKDOWN_OUTPUT_PATH)