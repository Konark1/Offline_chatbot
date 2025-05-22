import os
import json
import fitz  # PyMuPDF
from gpt4all import GPT4All
from functools import lru_cache
import time

def detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            print("GPU detected. Using GPU for inference.")
            return 'gpu'
    except ImportError:
        pass
    print("GPU not detected. Using CPU for inference.")
    return 'cpu'

class StudyBot:
    def __init__(self):
        for directory in ['models', 'documents']:
            os.makedirs(directory, exist_ok=True)
        model_path = os.path.join("models", "mistral-7b-instruct-v0.1.Q4_0.gguf")
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
            exit(1)
        device = detect_device()
        self.model = GPT4All(
            "mistral-7b-instruct-v0.1.Q4_0.gguf",
            model_path="models/",
            device=device,
            allow_download=False
        )
        self.formulas_file = "formulas.json"
        self.formulas_db = {}
        self.init_formulas_db()
        self.cache = {}
        self.pdf_cache = {}
        self.chunk_index = {}
        self.current_pdf = None

    def init_formulas_db(self):
        try:
            with open(self.formulas_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.formulas_db = data.get("formulas", {})
        except Exception:
            self.formulas_db = {}
            with open(self.formulas_file, 'w', encoding='utf-8') as f:
                json.dump({"formulas": {}}, f, indent=2)

    def normalize_query(self, query):
        filler_words = ['the', 'a', 'an', 'of', 'for', 'to', 'in', 'formula', 'equation']
        words = query.lower().split()
        return ' '.join(word for word in words if word not in filler_words).strip()

    @lru_cache(maxsize=100)
    def get_formula(self, query):
        query_normalized = self.normalize_query(query)
        if query_normalized in self.formulas_db:
            return f"📘 From Database:\n{self.formulas_db[query_normalized]}"
        prompt = f"""
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
        response = self.safe_generate(prompt)
        self.formulas_db[query_normalized] = response
        with open(self.formulas_file, 'w', encoding='utf-8') as f:
            json.dump({"formulas": self.formulas_db}, f, indent=2)
        return f"🧮 **Formula Result:**\n{response}"

    def extract_full_pdf_content(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        full_text = []
        with fitz.open(file_path) as doc:
            for i, page in enumerate(doc):
                page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE, sort=True).strip()
                if page_text:
                    full_text.append(page_text)
        return "\n".join(full_text)

    def split_text_into_chunks(self, text, chunk_size=400):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunks.append(' '.join(words[i:i+chunk_size]))
        return chunks

    def index_chapter(self, file_path):
        if not os.path.dirname(file_path):
            file_path = os.path.join("documents", file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}. Please place PDF files in the 'documents' folder.")
        cache_key = os.path.basename(file_path)
        self.current_pdf = cache_key
        if cache_key in self.pdf_cache:
            self.indexed_db = self.pdf_cache[cache_key]['chunks']
            self.chunk_index = self.pdf_cache[cache_key]['index']
            return "✅ Using cached content - Ready for queries"
        raw_text = self.extract_full_pdf_content(file_path)
        chunks = self.split_text_into_chunks(raw_text, chunk_size=500)
        self.indexed_db = {f"chunk_{i}": chunk for i, chunk in enumerate(chunks)}
        self.chunk_index = {}
        for chunk_id, content in self.indexed_db.items():
            words = set(content.lower().split())
            for word in words:
                if word not in self.chunk_index:
                    self.chunk_index[word] = []
                self.chunk_index[word].append(chunk_id)
        self.pdf_cache[cache_key] = {'chunks': self.indexed_db, 'index': self.chunk_index}
        return f"✅ In-depth Study completed: {len(chunks)} chunks indexed."

    def indepth_query(self, question):
        if not hasattr(self, 'indexed_db') or not self.indexed_db:
            raise ValueError("Please complete In-depth Study first")
        question_words = set(question.lower().split())
        relevant_chunks = set()
        for word in question_words:
            if word in self.chunk_index:
                relevant_chunks.update(self.chunk_index[word])
        # Use the top 2 most relevant chunks for context
        matches = [self.indexed_db[chunk_id] for chunk_id in list(relevant_chunks)[:2]]
        if not matches:
            matches = [chunk for chunk in self.indexed_db.values() if any(word in chunk.lower() for word in question_words)][:2]
        context = "\n\n".join(matches)
        cache_key = f"{self.current_pdf}:{question}"
        if cache_key in self.cache:
            return f"📚 From Cache:\n{self.cache[cache_key]}"
        prompt = f"""
Use the following context to answer the question as specifically as possible.

Context:
{context}

Question: {question}
Answer:"""
        response = self.safe_generate(prompt, max_length=192)
        self.cache[cache_key] = response
        return f"📚 Answer:\n{response}"

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

    def list_files(self):
        return [f for f in os.listdir("documents") if f.endswith('.pdf')]

    def safe_generate(self, prompt, max_retries=3, max_length=512):
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
                time.sleep(1)
        raise RuntimeError(f"Model generation failed after {max_retries} attempts: {str(last_error)}")

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
            if not hasattr(bot, 'indexed_db') or not bot.indexed_db:
                print("❌ Please index a PDF chapter first using 'index filename.pdf'")
            else:
                question = user_input[6:].strip()
                print(bot.indepth_query(question))
        else:
            print("❌ Invalid command. Type 'help' for available commands.")

if __name__ == "__main__":
    start = time.time()
    main()
    print("Step took", time.time() - start, "seconds")
