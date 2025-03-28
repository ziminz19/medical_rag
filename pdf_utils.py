import fitz  # PyMuPDF for PDF reading
import re
from pathlib import Path
import spacy
import json

class MedicalPaperPreprocessor:
    
    def __init__(self, pdf_dir, chunk_size_tokens=500, overlap_tokens=50):
        """
        Initialize the preprocessor.
        
        Parameters:
        - pdf_dir (str): Path to the folder containing PDF files.
        - chunk_size_tokens (int): Number of tokens per chunk.
        - overlap_tokens (int): Number of tokens to overlap between chunks.
        """
        self.pdf_dir = Path(pdf_dir)
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens
        self.pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        # Load a spaCy model for tokenization
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")
    
    def extract_main_text_from_pdf(self, pdf_path):
        """
        Extract the main body text from a PDF file.
        
        This method extracts text from all pages, skips lines that seem to be
        figure/table captions, and truncates the text at a "References" or "Bibliography" section.
        
        Parameters:
        - pdf_path (Path): Path to the PDF file.
        
        Returns:
        - text (str): The extracted main body text.
        """
        doc = fitz.open(pdf_path)
        full_text = []
        for page in doc:
            text = page.get_text("text")
            lines = text.splitlines()
            filtered_lines = []
            for line in lines:
                if not line.strip():
                    continue  # Skip empty lines
                # Skip lines starting with Figure, Fig., or Table (likely captions)
                if re.match(r'^\s*(Figure|Fig\.|Table)\s*\d+', line, flags=re.IGNORECASE):
                    continue
                # Skip lines that are just numbers (page numbers)
                if re.match(r'^\d+\s*$', line.strip()):
                    continue
                filtered_lines.append(line)
            page_text = " ".join(filtered_lines)
            full_text.append(page_text)
        doc.close()
        
        # Join all page texts into one string
        text = "\n".join(full_text)
        # Truncate text at the "References" or "Bibliography" section if found
        match = re.search(r'\bReferences\b', text, flags=re.IGNORECASE)
        if match:
            text = text[:match.start()]
        match = re.search(r'\bBibliography\b', text, flags=re.IGNORECASE)
        if match:
            text = text[:match.start()]
        return text
    
    def clean_text(self, text):
        """
        Clean and normalize text.
        
        - Fix hyphenated line breaks.
        - Remove excessive newlines and spaces.
        
        Parameters:
        - text (str): Raw text.
        
        Returns:
        - text (str): Cleaned text.
        """
        # Join words split by hyphenated line breaks (e.g., "experi-\nment" -> "experiment")
        text = re.sub(r'-\s*\n\s*', '', text)
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
        # Replace multiple spaces or tabs with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
    
    def chunk_text(self, text):
        """
        Split the text into overlapping chunks based on token counts.
        
        Uses spaCy to tokenize the text and then creates chunks of a specified size
        with an overlap between consecutive chunks.
        
        Parameters:
        - text (str): Cleaned text.
        
        Returns:
        - chunks (list of str): List of text chunks.
        """
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        chunks = []
        total_tokens = len(tokens)
        step = self.chunk_size_tokens - self.overlap_tokens
        if step <= 0:
            raise ValueError("Overlap must be smaller than the chunk size.")
        
        for start in range(0, total_tokens, step):
            end = start + self.chunk_size_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens).strip()
            chunks.append(chunk_text)
            if end >= total_tokens:
                break
        return chunks
    
    def process_pdf(self, pdf_path):
        """
        Process a single PDF: extract text, clean it, and split it into chunks.
        
        Parameters:
        - pdf_path (Path): Path to the PDF file.
        
        Returns:
        - chunks (list of str): List of text chunks from the PDF.
        """
        raw_text = self.extract_main_text_from_pdf(pdf_path)
        cleaned_text = self.clean_text(raw_text)
        return self.chunk_text(cleaned_text)
    
    def process_all_pdfs(self):
        """
        Process all PDFs in the provided directory.
        
        Returns:
        - all_chunks (list of dict): A list of dictionaries, each containing the source PDF name,
          chunk index, and text chunk.
        """
        all_chunks = []
        for pdf_path in self.pdf_files:
            chunks = self.process_pdf(pdf_path)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "source": pdf_path.name,
                    "chunk_index": i,
                    "text": chunk
                })
        return all_chunks
    
    def save_chunks_to_json(self, output_path, chunks):
        """
        Save the list of chunk dictionaries to a JSON file.
        
        Parameters:
        - output_path (str): File path for the JSON output.
        - chunks (list of dict): The list of chunk dictionaries to save.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

# Example usage:
if __name__ == "__main__":
    # Initialize the preprocessor with the local folder containing PDFs
    preprocessor = MedicalPaperPreprocessor(pdf_dir="papers", chunk_size_tokens=500, overlap_tokens=50)
    
    # Process all PDFs to get text chunks
    all_chunks = preprocessor.process_all_pdfs()
    
    # Save the chunks to a JSON file
    preprocessor.save_chunks_to_json("pdf_chunks.json", all_chunks)
    
    print(f"Processed {len(preprocessor.pdf_files)} PDFs and saved {len(all_chunks)} chunks to 'pdf_chunks.json'.")
