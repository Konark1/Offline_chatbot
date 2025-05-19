import os
import json
import fitz  # PyMuPDF
import pytesseract
import cv2
from PIL import Image
from gpt4all import GPT4All
import logging
from functools import lru_cache
import numpy as np
import time
import torch  
import sys
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Test Tesseract installation
try:
    test_text = pytesseract.get_tesseract_version()
    logging.info(f"Tesseract OCR version: {test_text}")
except Exception as e:
    logging.error(f"Tesseract OCR not properly configured: {str(e)}")
    logging.error("Please verify Tesseract installation and PATH")
    exit(1)

class StudyBot:
    def __init__(self):
        # Create required directories
        for directory in ['models', 'documents', 'images']:
            os.makedirs(directory, exist_ok=True)
            
        try:
            model_path = os.path.join("models", "mistral-7b-instruct-v0.1.Q4_0.gguf")
            if not os.path.exists(model_path):
                logging.error(f"Model file not found at: {model_path}")
                logging.error("Please download the model file and place it in the models folder")
                exit(1)

            # Detect available hardware
            device = self.detect_hardware()
            logging.info(f"Using device: {device}")
                
            # Load GPT4All model
            self.model = GPT4All(
                "mistral-7b-instruct-v0.1.Q4_0.gguf",
                model_path="models/",
                device=device,  # Will be either 'gpu' or 'cpu'
                n_threads=8 if device == 'cpu' else None,  # Only set threads for CPU
                allow_download=False
            )
        except Exception as e:
            logging.error(f"Failed to load GPT4All model: {str(e)}")
            exit(1)

        # Initialize formula storage (JSON file)
        self.formulas_file = "formulas.json"
        self.formulas_db = {}
        self.init_formulas_db()
        
        # Add cache initialization
        self.cache = {}
        self.status_callback = None
        self.progress_callback = None
        self.pdf_cache = {}
        self.chunk_index = {}  # Add semantic index for faster searching
        self.current_pdf = None

    # Add these new methods after __init__
    def set_callbacks(self, status_cb=None, progress_cb=None):
        """Set callback functions for status and progress updates."""
        self.status_callback = status_cb
        self.progress_callback = progress_cb

    def update_status(self, message):
        """Update status if callback is set."""
        if self.status_callback:
            self.status_callback(message)

    def update_progress(self, current, total, operation="Processing"):
        """Better progress tracking."""
        if self.progress_callback:
            progress = (current / total) * 100
            self.progress_callback(progress)
            if self.status_callback:
                self.status_callback(f"{operation}: {current}/{total} ({progress:.1f}%)")

    def validate_and_fix_json(self):
        try:
            # Attempt to load the JSON file
            with open(self.formulas_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if the "formulas" key exists and is a dictionary
            if not isinstance(data.get("formulas", {}), dict):
                raise ValueError("Invalid structure: 'formulas' key is not a dictionary.")
            
            logging.info("JSON file is valid.")
            return data  # Return the valid data

        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
            logging.error(f"Invalid JSON file: {str(e)}. Resetting to default.")
            # Reset the JSON file to a valid default state
            default_data = {"formulas": {}}
            with open(self.formulas_file, 'w', encoding='utf-8') as f:
                json.dump(default_data, f, ensure_ascii=False, indent=2)
            return default_data  # Return the default data

    def init_formulas_db(self):
        data = self.validate_and_fix_json()
        self.formulas_db = data.get("formulas", {})

    def normalize_query(self, query):
        """Normalize the query to handle similar variations."""
        filler_words = ['the', 'a', 'an', 'of', 'for', 'to', 'in', 'formula', 'equation']
        words = query.lower().split()
        normalized = ' '.join(word for word in words if word not in filler_words)
        return normalized.strip()

    @lru_cache(maxsize=100)
    def get_formula(self, query):
        """Get formula with caching."""
        query_normalized = self.normalize_query(query)
        
        # Check if formula exists in memory
        if query_normalized in self.formulas_db:
            logging.info(f"Formula for '{query_normalized}' found in database.")
            return f"📘 From Database:\n{self.formulas_db[query_normalized]}"
        
        logging.info(f"Formula for '{query_normalized}' not found. Generating...")
        structured_prompt = f"""
Provide a structured response for {query} using this exact format:

### Formula
Plain text: [Write the formula using simple characters like ^, *, /, sqrt()]
LaTeX: [Write the formula using LaTeX notation between $$ symbols]

### Definition
[Brief definition in 1-2 sentences]

### Components
- [variable]: [meaning] [unit if applicable]

### Example
Given: [input values]
Step 1: [show substitution with plain text formula]
Step 2: [show calculation]
Result: [final answer with unit]

### Notes
- [key point 1]
- [key point 2]
"""
        response = self.safe_generate(structured_prompt)
        self.formulas_db[query_normalized] = response
        try:
            with open(self.formulas_file, 'w', encoding='utf-8') as f:
                json.dump({"formulas": self.formulas_db}, f, indent=2)
            logging.info(f"Formula for '{query_normalized}' saved to formulas.json.")
        except Exception as e:
            logging.error(f"Failed to save formula to file: {str(e)}")
        
        return f"🧮 **Formula Result:**\n{response}"

    def analyze_image(self, image, page_num):
        """Optimized image analysis."""
        try:
            # Resize large images for faster processing
            height, width = image.shape[:2]
            if width > 1500 or height > 1500:
                scale = min(1500/width, 1500/height)
                image = cv2.resize(image, None, fx=scale, fy=scale)
            
            # Convert to grayscale and improve contrast
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            # Configure Tesseract for faster processing
            custom_config = r'--oem 3 --psm 1 -l eng'  # Fast mode
            text = pytesseract.image_to_string(gray, config=custom_config)
            
            return {
                'type': "Image",  # Simplified classification
                'text': text.strip(),
                'page': page_num
            }
        except Exception as e:
            logging.error(f"Image analysis failed: {str(e)}")
            return None

    def extract_text_from_image_page(self, page, save_path="images/page.png"):
        """Extract text from a PDF page containing images."""
        try:
            pix = page.get_pixmap()
            pix.save(save_path)
            img = cv2.imread(save_path)
            
            # Use the new analyze_image method
            analysis = self.analyze_image(img, page.number + 1)
            if analysis and analysis['text']:
                return f"""
Page {analysis['page']} Image Analysis:
Type: {analysis['type']}
Text Content: {analysis['text']}
"""
            return ""
            
        except Exception as e:
            logging.error(f"Error processing image page: {str(e)}")
            return ""

    def extract_full_pdf_content(self, file_path):
        """Optimized text extraction from PDF."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        try:
            full_text = []  # Use list instead of string concatenation
            with fitz.open(file_path) as doc:
                total_pages = len(doc)
                if total_pages == 0:
                    raise ValueError("PDF file is empty")
                    
                logging.info(f"Processing {total_pages} pages...")
                for i, page in enumerate(doc):
                    if self.progress_callback:
                        progress = (i + 1) / total_pages * 100
                        self.progress_callback(progress)
                        
                    # Extract text with optimized parameters
                    page_text = page.get_text(
                        "text",  # Extract plain text only
                        flags=fitz.TEXT_PRESERVE_WHITESPACE,  # Preserve formatting
                        sort=True  # Sort text blocks
                    ).strip()
                    
                    if page_text:
                        full_text.append(f"\n=== Page {i+1} ===\n{page_text}")
                    else:
                        # Only process images if no text is found
                        image_text = self.extract_text_from_image_page(page)
                        if image_text:
                            full_text.append(f"\n=== Page {i+1} ===\n{image_text}")
                        
            return "\n".join(full_text)  # Join at the end
                
        except Exception as e:
            logging.error(f"PDF processing failed: {str(e)}")
            raise

    def split_text_into_chunks(self, text, chunk_size=800, overlap=50):
        """Faster text chunking."""
        chunks = []
        sentences = text.split('.')
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def index_chapter(self, file_path):
        try:
            # Add memory management call
            self.manage_memory()
            
            # Ensure file path is in documents directory
            if not os.path.dirname(file_path):
                file_path = os.path.join("documents", file_path)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}. Please place PDF files in the 'documents' folder.")

            cache_key = os.path.basename(file_path)
            self.current_pdf = cache_key

            # Check cache first
            if cache_key in self.pdf_cache:
                logging.info("Using cached PDF content")
                self.indexed_db = self.pdf_cache[cache_key]['chunks']
                self.chunk_index = self.pdf_cache[cache_key]['index']
                return "✅ Using cached content - Ready for queries"

            logging.info(f"Processing PDF: {file_path}")
            raw_text = self.extract_full_pdf_content(file_path)
            
            # Create smaller, more focused chunks
            chunks = self.split_text_into_chunks(raw_text, chunk_size=500)
            
            # Create indexed database with semantic mapping
            self.indexed_db = {f"chunk_{i}": chunk for i, chunk in enumerate(chunks)}
            
            # Build semantic index
            self.chunk_index = {}
            for chunk_id, content in self.indexed_db.items():
                words = set(content.lower().split())
                for word in words:
                    if word not in self.chunk_index:
                        self.chunk_index[word] = []
                    self.chunk_index[word].append(chunk_id)
            
            # Cache the results
            self.pdf_cache[cache_key] = {
                'chunks': self.indexed_db,
                'index': self.chunk_index
            }
            
            for i, chunk in enumerate(chunks):
                self.update_progress(i + 1, len(chunks), "Indexing chunks")
                # Trim chunk if too large
                if len(chunk) > 8000:
                    chunk = chunk[:8000] + "..."
                
                # Add to indexed DB
                self.indexed_db[f"chunk_{i}"] = chunk
            
            return f"""✅ In-depth Study completed:
- PDF processed and indexed
- Created {len(chunks)} searchable chunks
- Built semantic index for faster queries"""
            
        except Exception as e:
            logging.error(f"Failed to index chapter: {str(e)}")
            raise

    def answer_from_chapter(self, question):
        matches = []
        for chunk in self.indexed_db.values():
            if any(word in chunk.lower() for word in question.lower().split()):
                matches.append(chunk)
        context = "\n\n".join(matches[:3])
        prompt = f"""
Answer the question using the following textbook content:

{context}

Question: {question}
Answer:
"""
        return self.model.generate(prompt)

    def summarize_chapter(self):
        all_text = "\n\n".join(self.indexed_db.values())
        prompt = f"""
Summarize this chapter in bullet points. Include:
- Key laws and concepts
- Important formulas (in simple plain text)
- Key figures or diagrams if mentioned
- Applications or implications if explained

{all_text[:8000]}
"""
        return self.model.generate(prompt)

    def query_pdf(self, filename, question):
        try:
            self.update_status(f"Processing PDF: {filename}")
            text = ""
            filepath = os.path.join("documents", filename)
            if not os.path.exists(filepath):
                logging.error(f"File '{filename}' not found.")
                return f"❌ Error: File '{filename}' not found."
            with fitz.open(filepath) as doc:
                for i, page in enumerate(doc):
                    if i >= 10:
                        break
                    text += page.get_text()

            if not text.strip():
                logging.warning("The document is empty or unreadable.")
                return "❌ Error: The document is empty or unreadable."

            self.update_status("Generating response...")
            response = self.model.generate(
                f"Answer based on this document:\n{text[:10000]}\n\nQuestion: {question}\nAnswer:"
            )
            self.update_status("Done")
            return f"📄 PDF Answer:\n{response}"
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            logging.error(f"Error processing PDF query: {str(e)}")
            return f"❌ Error: {str(e)}"

    def list_files(self):
        try:
            files = [f for f in os.listdir("documents") if f.endswith('.pdf')]
            return files
        except Exception as e:
            logging.error(f"Error listing files: {str(e)}")
            return []

    def indepth_query(self, question):
        """Get detailed answers using optimized search"""
        if not self.indexed_db:
            raise ValueError("Please complete In-depth Study first")
        
        try:
            # Use semantic index for faster searching
            question_words = set(question.lower().split())
            relevant_chunks = set()
            
            # Find chunks containing question words
            for word in question_words:
                if word in self.chunk_index:
                    relevant_chunks.update(self.chunk_index[word])
            
            # Get the most relevant chunks
            matches = [self.indexed_db[chunk_id] for chunk_id in list(relevant_chunks)[:3]]
            
            if not matches:
                # Fallback to basic search if no matches found
                matches = [chunk for chunk in self.indexed_db.values() 
                          if any(word in chunk.lower() for word in question_words)][:3]
            
            context = "\n\n".join(matches)
            
            # Use cached response if available
            cache_key = f"{self.current_pdf}:{question}"
            if cache_key in self.cache:
                return f"📚 From Cache:\n{self.cache[cache_key]}"
            
            prompt = f"""
Provide a concise answer using this content:

{context}

Question: {question}
Answer (be specific and brief):"""

            response = self.safe_generate(prompt, max_length=1024)
            self.cache[cache_key] = response
            return f"📚 Answer:\n{response}"
            
        except Exception as e:
            logging.error(f"Error in detailed query: {str(e)}")
            raise

    def search_query(self, question):
        """Hybrid search: Try online, fallback to offline model."""
        try:
            # 1. Try online search (Wikipedia API example)
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{question.replace(' ', '_')}"
            try:
                resp = requests.get(wiki_url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    extract = data.get('extract')
                    if extract:
                        return f"🌐 Wikipedia:\n{extract}"
            except Exception as e:
                logging.warning(f"Online search failed: {str(e)}")

            # 2. Fallback to offline model
            cache_key = f"search:{question.lower()}"
            if cache_key in self.cache:
                logging.info("Using cached search result")
                return f"🔍 From Cache:\n{self.cache[cache_key]}"
            prompt = f"""Provide a concise answer to: {question}
Format your response as:
1. Brief Answer (2-3 sentences)
2. Key Points (bullet points)
3. Example (if applicable)
Keep the response focused and direct.
"""
            response = self.model.generate(
                prompt,
                max_tokens=256,
                temp=0.7,
                top_k=40,
                top_p=0.4,
                repeat_penalty=1.18
            )
            self.cache[cache_key] = response
            return f"🤖 Offline Model:\n{response}"
        except Exception as e:
            logging.error(f"Search query failed: {str(e)}")
            raise ValueError(f"Search failed: {str(e)}")

    def clear_cache(self):
        """Clear the formula cache."""
        self.get_formula.cache_clear()
        self.cache.clear()
        logging.info("Cache cleared")
        if self.status_callback:
            self.status_callback("Cache cleared")

    def manage_cache(self):
        """Manage cache size to prevent memory issues."""
        max_cache_entries = 100
        max_pdf_cache = 5
        
        # Trim main cache if too large
        if len(self.cache) > max_cache_entries:
            items = sorted(self.cache.items(), key=lambda x: x[0])
            self.cache = dict(items[-max_cache_entries:])
            
        # Trim PDF cache if too large
        if len(self.pdf_cache) > max_pdf_cache:
            items = sorted(self.pdf_cache.items(), key=lambda x: x[0])
            self.pdf_cache = dict(items[-max_pdf_cache:])

    def manage_memory(self):
        """Add active memory management."""
        if len(self.pdf_cache) > 5:  # Keep only 5 most recent PDFs
            oldest_key = list(self.pdf_cache.keys())[0]
            del self.pdf_cache[oldest_key]
            logging.info(f"Removed {oldest_key} from PDF cache")
    
        # Clean up main cache if too large
        if len(self.cache) > 100:
            self.cache.clear()
            logging.info("Main cache cleared due to size limit")

    def safe_generate(self, prompt, max_retries=3, max_length=512):
        """Safe model generation with retries and error handling."""
        last_error = None
        for attempt in range(max_retries):
            try:
                if len(prompt) > 4096:
                    prompt = prompt[:4096] + "..."
                response = self.model.generate(
                    prompt,
                    max_tokens=max_length,
                    temp=0.7,
                    top_k=40,
                    top_p=0.4,
                    repeat_penalty=1.18
                )
                if not response or len(response.strip()) < 10:
                    raise ValueError("Generated response too short or empty")
                return response
            except Exception as e:
                last_error = e
                logging.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                time.sleep(1)
        raise RuntimeError(f"Model generation failed after {max_retries} attempts: {str(last_error)}")

    def detect_hardware(self):
        """Detect available hardware for model execution."""
        try:
            # Try to detect NVIDIA GPU first
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    logging.info(f"NVIDIA GPU detected: {gpu_name}")
                    return 'gpu'  # Changed from 'cuda' to 'gpu'
            except ImportError:
                logging.debug("PyTorch not found, skipping NVIDIA GPU detection")
            except Exception as e:
                logging.debug(f"Failed to detect NVIDIA GPU: {str(e)}")

            # Try to detect AMD GPU
            try:
                if os.name == 'nt':  # Windows
                    import ctypes
                    ctypes.CDLL('amddxx64.dll')
                    logging.info("AMD GPU detected")
                    return 'gpu'  # Changed from 'rocm' to 'gpu'
            except Exception as e:
                logging.debug(f"Failed to detect AMD GPU: {str(e)}")

            # CPU fallback with optimization
            cpu_count = os.cpu_count() or 1
            threads = max(1, min(cpu_count - 1, 8))  # Leave one core free, max 8 threads
            logging.info(f"Using CPU with {threads} threads")
            return 'cpu'  # Simplified to just 'cpu'

        except Exception as e:
            logging.warning(f"Hardware detection failed: {str(e)}")
            return 'cpu'  # Safe fallback

    def cli_progress_bar(self, current, total, width=50):
        progress = int(width * current / total)
        return f"[{'=' * progress}{' ' * (width-progress)}] {current}/{total}"

def main():
    bot = StudyBot()
    print("🔍 AI Study Bot - Type 'help' for commands")
    
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            print("❌ Please enter a command. Type 'help' for available commands.")
            continue

        if user_input.lower() in ['exit', 'quit']:
            break
        elif user_input.lower() == 'help':
            print("\nCommands:")
            print("ask <question> - Get a formula")
            print("search <question> - General knowledge search")  # Add this
            print("pdf <filename> <question> - Query a PDF")
            print("list - Show available PDFs")
            print("index <filename> - Study PDF in-depth")
            print("query <question> - Ask detailed question after indexing")
            print("summary - Summarize the chapter")
            print("exit - Quit")
        elif user_input.lower() == 'list':
            files = bot.list_files()
            print("\nAvailable PDFs:" if files else "\nNo PDFs found")
            for f in files:
                print(f"- {f}")
        elif user_input.lower().startswith('pdf '):
            parts = user_input.split(maxsplit=2)
            if len(parts) < 3:
                print("Usage: pdf filename.pdf 'your question'")
            else:
                print(bot.query_pdf(parts[1], parts[2]))
        elif user_input.lower().startswith('index '):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: index filename.pdf")
            else:
                print(bot.index_chapter(parts[1]))
        elif user_input.lower() == 'summary':
            print(bot.summarize_chapter())
        elif user_input.lower().startswith('ask '):
            print(bot.get_formula(user_input[4:]))
        elif user_input.lower().startswith('query '):
            if not bot.indexed_db:
                print("❌ Please index a PDF chapter first using 'index filename.pdf'")
            else:
                question = user_input[6:].strip()
                print(bot.indepth_query(question))
        elif user_input.lower().startswith('search '):
            question = user_input[7:].strip()
            print(bot.search_query(question))
        else:
            print("❌ Invalid command. Type 'help' for available commands.")

if __name__ == "__main__":
    start = time.time()
    main()
    print("Step took", time.time() - start, "seconds")
