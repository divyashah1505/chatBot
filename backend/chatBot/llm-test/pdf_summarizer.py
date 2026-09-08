import os
import sys
from pypdf import PdfReader
from openai import OpenAI

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using pypdf."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    print(f"📄 Reading PDF: {os.path.basename(pdf_path)}...")
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n--- Page {i+1} ---\n" + page_text
            
    print(f"✅ Extracted {len(reader.pages)} pages ({len(text)} characters).")
    return text.strip()

def chunk_text(text: str, max_chars_per_chunk: int = 4000) -> list[str]:
    """Splits long text into smaller chunks to handle LLM context token limits."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        if len(current_chunk) + len(p) <= max_chars_per_chunk:
            current_chunk += p + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def summarize_with_llama(text_to_summarize: str, is_chunk: bool = False, chunk_num: int = 1, total_chunks: int = 1) -> str:
    """Send text to Local Llama model via LM Studio."""
    client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio"
    )
    
    if is_chunk:
        print(f"🤖 Summarizing Chunk {chunk_num}/{total_chunks} with Llama...")
        prompt = f"Summarize the following section of a document into concise bullet points:\n\n{text_to_summarize}"
    else:
        print(f"🤖 Generating Final Summary with Llama...")
        prompt = f"""You are an AI Document Summarizer. 
Summarize the following document into a structured overview with the following sections:
1. Executive Summary / Overview
2. Key Highlights & Main Points
3. Core Skills / Technical Details (if applicable)
4. Action Items / Takeaways

Document Content:
{text_to_summarize}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.2-3b-instruct",
            messages=[
                {"role": "system", "content": "You are a professional AI document analyst. Provide clean, well-structured markdown summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Error connecting to LM Studio: {e}")
        print("💡 Make sure LM Studio Local Server is running at http://localhost:1234")
        return ""

def summarize_pdf(pdf_path: str) -> str:
    """Main pipeline for PDF summarization."""
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        return "⚠️ PDF appears to be empty or contains scanned images without OCR text."
        
    # Context window management
    chunks = chunk_text(text, max_chars_per_chunk=4000)
    print(f"🧩 Split document into {len(chunks)} chunk(s) for processing.")
    
    if len(chunks) == 1:
        final_summary = summarize_with_llama(chunks[0], is_chunk=False)
    else:
        chunk_summaries = []
        for idx, chunk in enumerate(chunks):
            chunk_summary = summarize_with_llama(chunk, is_chunk=True, chunk_num=idx+1, total_chunks=len(chunks))
            if chunk_summary:
                chunk_summaries.append(chunk_summary)
                
        combined_summaries = "\n\n".join(chunk_summaries)
        final_summary = summarize_with_llama(combined_summaries, is_chunk=False)
        
    return final_summary

if __name__ == "__main__":
    print("=" * 60)
    print(" 📚 PDF Summarizer using Local Llama Model (LM Studio) ")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        pdf_file = input("\nEnter path to your PDF file: ").strip().strip('"')
        
    if not pdf_file:
        print("No PDF path provided. Exiting.")
        sys.exit(1)
        
    summary = summarize_pdf(pdf_file)
    
    print("\n" + "=" * 60)
    print(" 📝 GENERATED SUMMARY ")
    print("=" * 60 + "\n")
    print(summary)
