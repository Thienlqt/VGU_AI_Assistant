import google.generativeai as genai
import pandas as pd
import numpy as np
import textwrap
import re
import os
import fitz  # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

# --- Secure Configuration for Local Development ---
load_dotenv() # This loads the variables from your .env file

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("🛑 ERROR: GOOGLE_API_KEY not found in .env file.")
    print("Please create a .env file and add your key to it.")
    exit() # Stop if key is not found

try:
    genai.configure(api_key=API_KEY)
    print("✅ API key configured successfully from .env file!")
except Exception as e:
    print(f"🛑 API key might be invalid or billing is not enabled. Error: {e}")
    exit()

PDF_FILE_PATH = "D:\\ai_assistant\VGU_AI_Assistant\\Data_rag\\20250120 Bachelor Admission Regulation 2025_VN_Hanh draft (1).pdf"
EMBEDDING_MODEL = 'models/text-embedding-004'
#GENERATIVE_MODEL = 'gemini-2.0-flash-001'

GENERATIVE_MODEL = ChatGoogleGenerativeAI(
    google_api_key=API_KEY,
    model="gemini-2.0-flash-thinking-exp-01-21",
    temperature=0.7
)

# --- PART 1: PDF PROCESSING & KNOWLEDGE BASE CREATION ---

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a given PDF file using PyMuPDF.
    """
    print(f"Extracting text from '{pdf_path}'...")
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text("text")
            full_text += "\n\n" # Add a separator between pages
        print("Text extraction successful.")
        return full_text
    except Exception as e:
        print(f"Error reading or processing PDF file: {e}")
        return None

def preprocess_and_chunk_text(full_text):
    """
    Takes a string of text and applies a two-level hierarchical chunking strategy.
    1. Splits the document into major sections (by 'Điều' and 'Phụ lục').
    2. Further splits each section into paragraphs.
    3. Prepends the section title to each paragraph for context.
    """
    if not full_text:
        return None

    # First, split into major sections based on 'Điều' and 'Phụ lục'
    major_sections = re.split(r'(?=Điều \d+\.|Phụ lục \d+\.)', full_text)
    
    final_chunks = []
    for section in major_sections:
        if not section.strip():
            continue

        # Find the title of the section (the first line)
        lines = section.strip().split('\n')
        section_title = lines[0].strip()
        
        # Split the rest of the section content by one or more newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', section)

        for para in paragraphs:
            cleaned_para = para.strip()
            if cleaned_para:
                # Create a chunk by combining the section title with the paragraph.
                # This gives each paragraph essential context.
                # We check if the paragraph already starts with the title to avoid duplication.
                if not cleaned_para.startswith(section_title):
                    contextual_chunk = f"{section_title}\n\n{cleaned_para}"
                else:
                    contextual_chunk = cleaned_para
                
                final_chunks.append(contextual_chunk)

    if not final_chunks:
        print("Warning: No chunks were created. The document might be empty or in an unexpected format.")
        # Fallback to simple paragraph splitting if the main logic fails
        final_chunks = [p.strip() for p in full_text.split('\n\n') if p.strip()]

    df = pd.DataFrame(final_chunks, columns=['text_chunk'])
    print(f"Successfully created {len(df)} contextual chunks from the document.")
    return df


def embed_knowledge_base(df):
    """
    Takes a DataFrame of text chunks, embeds them, and adds the embeddings to the DataFrame.
    """
    if df is None or df.empty:
        print("DataFrame is empty. Cannot perform embedding.")
        return None

    def embed_fn(text):
        return genai.embed_content(model=EMBEDDING_MODEL,
                                   content=text,
                                   task_type="RETRIEVAL_DOCUMENT")['embedding']

    print(f"Embedding knowledge base using '{EMBEDDING_MODEL}'... (This may take a moment)")
    df['embedding'] = df['text_chunk'].apply(embed_fn)
    print("Embedding complete.")
    return df

# --- PART 2: THE RAG PIPELINE (RETRIEVE & GENERATE) ---
# NEW, CLEARER, AND MORE ROBUST SEARCH FUNCTION

def semantic_search(query, knowledge_base, top_k=3):
    """
    Performs semantic search using cosine similarity and includes a crucial debugging step.
    
    Args:
        query (str): The user's question.
        knowledge_base (pd.DataFrame): The DataFrame containing text chunks and their embeddings.
        top_k (int): The number of top results to return.
        
    Returns:
        pd.DataFrame: A DataFrame containing the top_k most relevant chunks.
    """
    query_embedding = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="RETRIEVAL_QUERY"
    )['embedding']

    # Reshape the query embedding to be a 2D array for the function
    query_embedding_2d = np.array(query_embedding).reshape(1, -1)

    # Get all document embeddings as a 2D array
    doc_embeddings = np.stack(knowledge_base['embedding'])

    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding_2d, doc_embeddings)[0]

    # Get the indices of the top_k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # --- VITAL DEBUGGING STEP ---
    print("\n--- Top Retrieved Chunks (for debugging) ---")
    for i in top_indices:
        print(f"Similarity Score: {similarities[i]:.4f}")
        print(f"Chunk: {knowledge_base.iloc[i]['text_chunk'][:250]}...") # Print the start of the chunk
        print("-" * 20)
    print("--- End of Retrieved Chunks ---\n")
    # --- END OF DEBUGGING STEP ---

    # Return the top_k chunks
    return knowledge_base.iloc[top_indices]

def generate_answer_with_flash(query, context_chunks):
    """
    Generates an answer using Gemini 1.5 Flash.
    This version includes a more advanced prompt that encourages reasoning, counting, and synthesis.
    """
    context = "\n\n---\n\n".join(context_chunks['text_chunk'])
    
    # This is the new, more powerful prompt.
    prompt = f"""
    You are an intelligent and helpful assistant for the Vietnamese-German University (VGU).
    Your primary task is to analyze the provided context to answer the user's question accurately and concisely.

    **Instructions for Reasoning:**
    1.  **Synthesize, Don't Just Extract:** Do not just copy-paste from the context. Read and understand the information to form a complete, coherent answer.
    2.  **Count Lists:** If the user asks for a quantity (e.g., 'how many', 'có mấy'), you MUST scan the context for numbered or lettered lists (like a), b), c) or 1., 2., 3.). Count the items in the list to determine the total number, and state that number clearly in your answer before listing the items.
    3.  **Stay Grounded:** Base your answer STRICTLY on the provided context. Do not use any external knowledge.
    4.  **Handle Missing Information:** If the information to answer the question is truly not present in the context, state clearly: "Thông tin này không có trong tài liệu được cung cấp."
    5.  **Language:** Always answer in Vietnamese.

    **CONTEXT:**
    ---
    {context}
    ---

    **USER'S QUESTION:**
    {query}

    **ANALYSIS AND ANSWER:**
    """
    
    generative_model = genai.GenerativeModel(GENERATIVE_MODEL)
    # It's good practice to add safety settings
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    answer = generative_model.generate_content(prompt, safety_settings=safety_settings)
    
    return answer.text


# --- PART 3 & 4: FEEDBACK LOOP & CHATBOT INTERFACE ---

def main_chat_loop(pdf_path):
    """
    Main function to run the chatbot, including the feedback loop.
    """
    # --- Initialization ---
    feedback_log_path = 'D:\\ai_assistant\\VGU_AI_Assistant\\feedback\\feedback_log.csv'
    base_name = os.path.splitext(pdf_path)[0]
    knowledge_base_path = f"{base_name}_knowledge_base.pkl"

    if os.path.exists(knowledge_base_path):
        print(f"Loading existing knowledge base from '{knowledge_base_path}'...")
        knowledge_base = pd.read_pickle(knowledge_base_path)
    else:
        print("No existing knowledge base found. Processing PDF from scratch...")
        document_text = extract_text_from_pdf(pdf_path)
        if not document_text:
            return # Stop if PDF processing failed
            
        df_chunks = preprocess_and_chunk_text(document_text)
        if df_chunks is None:
            return
            
        knowledge_base = embed_knowledge_base(df_chunks)
        knowledge_base.to_pickle(knowledge_base_path)
        print(f"Knowledge base created and saved to '{knowledge_base_path}'.")

    if os.path.exists(feedback_log_path) and os.path.getsize(feedback_log_path) > 0:
        feedback_df = pd.read_csv(feedback_log_path)
    else:
        # If file doesn't exist or is empty, create a new DataFrame.
        feedback_df = pd.DataFrame(columns=['query', 'generated_answer', 'feedback', 'correction'])
        # Optional: create an empty file with headers so it's not empty next time
        feedback_df.to_csv(feedback_log_path, index=False)

    print(f"\n--- VGU Regulations Chatbot ---")
    print("Ask a question about VGU's undergraduate admission regulations. Type 'quit' to exit.")

    while True:
        query = input("\nYour Question: ")
        if query.lower() == 'quit':
            break

        # 1. Retrieve
        relevant_chunks = semantic_search(query, knowledge_base)
        
        # 2. Generate with Flash
        answer = generate_answer_with_flash(query, relevant_chunks)
        print("\nChatbot Answer:")
        print(textwrap.fill(answer, width=80))

        # 3. Collect Feedback
        print("\nWas this answer helpful?")
        feedback = input("Enter 'yes', 'no', or 'edit': ").lower()

        correction = ""
        if feedback == 'no':
            print("Sorry the answer was not helpful. Your feedback is logged.")
        elif feedback == 'edit':
            correction = input("Please provide the correct or a better answer: ")
            print("Thank you! Your correction will help improve the system.")
            
            # --- 4. The Feedback Loop: Create and add a "golden record" ---
            print("Updating knowledge base with your feedback...")
            new_record_text = f"User query: {query}\nCorrect answer: {correction}"
            
            new_embedding = genai.embed_content(model=EMBEDDING_MODEL,
                                               content=new_record_text,
                                               task_type="RETRIEVAL_DOCUMENT")['embedding']
            
            new_row = pd.DataFrame([{'text_chunk': new_record_text, 'embedding': new_embedding}])
            knowledge_base = pd.concat([knowledge_base, new_row], ignore_index=True)
            knowledge_base.to_pickle(knowledge_base_path) # Persist the improvement
            print("Knowledge base updated.")

        # Log the interaction for offline analysis
        new_log_entry = pd.DataFrame([{'query': query, 'generated_answer': answer, 'feedback': feedback, 'correction': correction}])
        feedback_df = pd.concat([feedback_df, new_log_entry], ignore_index=True)
        feedback_df.to_csv(feedback_log_path, index=False)


# --- Run the chatbot ---
if __name__ == "__main__":
    main_chat_loop(pdf_path=PDF_FILE_PATH)