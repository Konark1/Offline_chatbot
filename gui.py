import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from main import StudyBot
import threading
from queue import Queue
import os
import logging
import fitz  # PyMuPDF

class StudyBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("📘 StudyBot AI")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')

        # Initialize variables
        self.count_var = tk.StringVar(value="Words: 0 | Chars: 0")
        self.status_var = tk.StringVar(value="Ready")
        self.mode_var = None
        self.selected_file = None
        self.indepth_completed = False

        # Initialize bot
        self.bot = StudyBot()
        self.bot.set_callbacks(
            status_cb=self.update_status,
            progress_cb=self.update_progress
        )
        
        # Queue and threading for processing
        self.processing_queue = Queue()
        self.processing_thread = None
        
        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        # Configure styles for widgets
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TButton', padding=5, font=('Segoe UI', 10))
        style.configure('TLabel', background='#f0f0f0', font=('Segoe UI', 10))
        style.configure('TCombobox', padding=5, font=('Segoe UI', 10))

    def create_widgets(self):
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Top frame for controls
        top_frame = ttk.Frame(main_container)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        # Mode selection frame
        mode_frame = ttk.LabelFrame(top_frame, text="Mode", padding="5")
        mode_frame.pack(side=tk.LEFT, padx=(0, 10))

        # Update mode selection with all options
        self.mode_var = tk.StringVar(value="ask")
        modes = [
            ("Ask Formula", "ask"),
            ("Search", "search"),  # Add search mode
            ("PDF Query", "pdf"),
            ("In-depth Study", "index"),
            ("Summarize", "summary")
        ]
        for text, value in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, 
                          value=value, command=self.update_interface).pack(side=tk.LEFT, padx=5)
        
        # PDF controls frame
        pdf_frame = ttk.Frame(top_frame)
        pdf_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.file_button = ttk.Button(pdf_frame, text="📂 Select PDF", command=self.select_pdf)
        self.file_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.list_btn = ttk.Button(pdf_frame, text="📄 List PDFs", command=self.list_pdfs)
        self.list_btn.pack(side=tk.LEFT)

        self.file_label = ttk.Label(pdf_frame, text="No PDF selected")
        self.file_label.pack(side=tk.LEFT, padx=10)

        # Query frame
        query_frame = ttk.LabelFrame(main_container, text="Query", padding="5")
        query_frame.pack(fill=tk.X, pady=(0, 10))

        self.query_entry = ttk.Entry(query_frame, font=("Segoe UI", 12))
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))
        self.query_entry.bind("<Return>", lambda event: self.handle_query())

        # Add word count display
        self.count_var = tk.StringVar(value="Words: 0 | Chars: 0")
        count_label = ttk.Label(
            query_frame,
            textvariable=self.count_var,
            style='Info.TLabel'
        )
        count_label.pack(side=tk.RIGHT, padx=5)
        
        # Bind update to text changes
        self.query_entry.bind('<KeyRelease>', self.update_counts)

        self.submit_btn = tk.Button(query_frame, text="🔍 Submit", command=self.handle_query,
                                  bg="#2196F3", fg="white", font=("Segoe UI", 10, "bold"),
                                  relief=tk.FLAT, padx=20)
        self.submit_btn.pack(side=tk.LEFT)

        # Add after query_entry
        self.history = []
        self.history_var = tk.StringVar()
        self.history_dropdown = ttk.Combobox(
            query_frame,
            textvariable=self.history_var,
            values=self.history,
            width=30
        )
        self.history_dropdown.pack(side=tk.LEFT, padx=5)
        self.history_dropdown.bind('<<ComboboxSelected>>', self.load_history)

        # Add after submit button
        shortcuts_label = ttk.Label(
            query_frame,
            text="Submit (Enter) | Clear (Esc)",
            style='Info.TLabel'
        )
        shortcuts_label.pack(side=tk.RIGHT, padx=10)

        # Output area
        output_frame = ttk.LabelFrame(main_container, text="Response", padding="5")
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.output_box = scrolledtext.ScrolledText(
            output_frame, 
            wrap=tk.WORD, 
            font=("Consolas", 11),
            bg='#ffffff',
            border=0
        )
        self.output_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add Export button frame and button
        export_frame = ttk.Frame(main_container)
        export_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.export_btn = ttk.Button(
            export_frame,
            text="📥 Export Results",
            command=self.export_results,
            style='Primary.TButton'
        )
        self.export_btn.pack(side=tk.RIGHT)

        # Add near export button
        clear_btn = ttk.Button(
            export_frame,
            text="🗑️ Clear Output",
            command=self.clear_output
        )
        clear_btn.pack(side=tk.RIGHT, padx=5)

        # Add progress bar
        self.progress_frame = ttk.Frame(main_container)
        self.progress_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X)
        self.progress_frame.pack_forget()  # Hide initially

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var,
            relief=tk.SUNKEN, 
            anchor=tk.W,
            padding=(5, 2)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Add loading indicator
        self.loading_label = ttk.Label(
            main_container,
            text="🔄 Processing...",
            style='Loading.TLabel'
        )
        # Will be shown/hidden as needed
    
    def show_loading(self):
        self.loading_label.pack(before=self.output_box)
        self.root.update_idletasks()
    
    def hide_loading(self):
        self.loading_label.pack_forget()

    def select_pdf(self):
        """Handle PDF file selection."""
        try:
            # Ensure documents directory exists
            documents_dir = os.path.abspath("documents")
            if not os.path.exists(documents_dir):
                os.makedirs(documents_dir)
                
            file_path = filedialog.askopenfilename(
                initialdir=documents_dir,
                title="Select PDF",
                filetypes=(("PDF Files", "*.pdf"),)
            )
            
            if file_path:
                # Normalize paths for comparison
                file_path = os.path.normpath(file_path)
                filename = os.path.basename(file_path)
                destination = os.path.normpath(os.path.join(documents_dir, filename))
                
                # Check if file needs to be copied
                if os.path.normcase(file_path) != os.path.normcase(destination):
                    import shutil
                    shutil.copy2(file_path, destination)
                    logging.info(f"Copied PDF to documents folder: {destination}")
                
                # Store the filename only
                self.selected_file = filename
                
                # Update GUI elements
                self.file_label.config(text=f"Selected: {filename}")
                
                # Verify file is readable
                try:
                    with fitz.open(destination) as doc:
                        page_count = len(doc)
                        self.status_var.set(f"PDF loaded: {filename} ({page_count} pages)")
                except Exception as e:
                    raise ValueError(f"Invalid or corrupted PDF file: {str(e)}")
                    
                # Enable/update relevant controls
                self.update_interface()
                logging.info(f"Selected PDF: {filename}")
                    
            else:
                self.status_var.set("PDF selection cancelled")
                
        except Exception as e:
            logging.error(f"Error selecting PDF: {str(e)}")
            messagebox.showerror("Error", str(e))
            self.file_label.config(text="No PDF selected")
            self.selected_file = None
            self.update_interface()

    def update_interface(self):
        mode = self.mode_var.get()
        self.query_entry.delete(0, tk.END)
        
        if mode == "ask":
            self.query_entry.config(state="normal")
            self.file_button.config(state="disabled")
            self.query_entry.insert(0, "Enter formula query...")
        elif mode == "search":  # Add search mode
            self.query_entry.config(state="normal")
            self.file_button.config(state="disabled")
            self.query_entry.insert(0, "Enter any question...")
        elif mode == "pdf":
            self.query_entry.config(state="normal")
            self.file_button.config(state="normal")
            self.query_entry.insert(0, "Enter question about the PDF...")
        elif mode == "index":
            self.file_button.config(state="normal")
            if self.indepth_completed:
                self.query_entry.config(state="normal")
                self.query_entry.insert(0, "Enter question for detailed study...")
                self.file_button.config(state="disabled")  # Disable file selection after indexing
            else:
                self.query_entry.config(state="disabled")
                self.status_var.set("Select a PDF file to begin in-depth study")
        elif mode == "summary":
            if not self.indepth_completed:
                messagebox.showwarning("Warning", "Please perform In-depth Study first!")
                self.mode_var.set("index")
                self.update_interface()
                return
            self.query_entry.config(state="disabled")
            self.file_button.config(state="disabled")

    def run_in_thread(self, target, args=(), callback=None):
        """Run a function in a separate thread."""
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Warning", "Please wait for current operation to complete.")
            return
            
        def thread_target():
            try:
                result = target(*args)
                if callback:
                    self.root.after(0, callback, result)
            except Exception as e:
                self.root.after(0, self.handle_error, str(e))
            finally:
                self.root.after(0, self.enable_interface)
                
        self.disable_interface()
        self.processing_thread = threading.Thread(target=thread_target)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def disable_interface(self):
        """Disable interface during processing."""
        self.query_entry.config(state='disabled')
        self.submit_btn.config(state='disabled')
        self.file_button.config(state='disabled')
        self.progress_frame.pack(fill=tk.X, pady=(0, 5))
        
    def enable_interface(self):
        """Re-enable interface after processing."""
        self.update_interface()
        self.progress_frame.pack_forget()

    def handle_query(self):
        query = self.query_entry.get().strip()
        mode = self.mode_var.get()
        
        self.output_box.delete(1.0, tk.END)
        self.status_var.set("Processing...")
        
        try:
            if mode == "search":
                if not query or query == "Enter any question...":
                    raise ValueError("Please enter a question")
                    
                def search_callback(result):
                    self.output_box.insert(tk.END, result)
                    self.status_var.set("Search completed")
                    # Add to history
                    self.add_to_history(query)
                
                self.run_in_thread(
                    target=self.bot.search_query,
                    args=(query,),
                    callback=search_callback
                )
                return
                
            elif mode == "index":
                if not self.selected_file:
                    raise ValueError("Please select a PDF file first")
                pdf_path = os.path.join(os.path.abspath("documents"), self.selected_file)
                
                def index_callback(result):
                    self.output_box.insert(tk.END, result)
                    self.indepth_completed = True
                    self.status_var.set("In-depth study completed")
                    self.update_interface()
                
                self.run_in_thread(
                    target=self.bot.index_chapter,
                    args=(pdf_path,),
                    callback=index_callback
                )
                return

            elif mode == "summary":
                if not self.indepth_completed:
                    raise ValueError("Please complete In-depth Study first")
                    
                def summary_callback(result):
                    self.output_box.insert(tk.END, result)
                    self.status_var.set("Summary generated")
                    self.add_to_history("Summary request")
                
                self.run_in_thread(
                    target=self.bot.summarize_chapter,
                    args=(),
                    callback=summary_callback
                )
                return
                
            elif mode == "ask":
                if not query or query == "Enter formula query...":
                    raise ValueError("Please enter a formula query")
                    
                def formula_callback(result):
                    self.output_box.insert(tk.END, result)
                    self.status_var.set("Formula retrieved")
                    self.add_to_history(query)
                
                self.run_in_thread(
                    target=self.bot.get_formula,
                    args=(query,),
                    callback=formula_callback
                )
                return
                
            elif mode == "pdf":
                if not self.selected_file:
                    raise ValueError("Please select a PDF file first")
                if not query or query == "Enter question about the PDF...":
                    raise ValueError("Please enter a question about the PDF")
                    
                pdf_path = os.path.join(os.path.abspath("documents"), self.selected_file)
                
                def pdf_callback(result):
                    self.output_box.insert(tk.END, result)
                    self.status_var.set("PDF query completed")
                    self.add_to_history(query)
                
                self.run_in_thread(
                    target=self.bot.query_pdf,
                    args=(pdf_path, query),
                    callback=pdf_callback
                )
                return
                
        except Exception as e:
            self.handle_error(str(e))

    def add_to_history(self, query):
        """Add query to history dropdown."""
        if query not in self.history:
            self.history.insert(0, query)
            if len(self.history) > 10:  # Keep last 10 queries
                self.history.pop()
            self.history_dropdown['values'] = self.history

    def load_history(self, event):
        """Load selected history item into query entry."""
        selected = self.history_var.get()
        if selected:
            self.query_entry.delete(0, tk.END)
            self.query_entry.insert(0, selected)

    def handle_error(self, error_message):
        """Handle errors from threaded operations."""
        messagebox.showerror("Error", error_message)
        self.status_var.set("Error occurred")
        logging.error(f"Error in operation: {error_message}")
        self.enable_interface()

    def export_results(self):
        """Export results to a text file."""
        try:
            content = self.output_box.get(1.0, tk.END).strip()
            if not content:
                raise ValueError("No content to export")
                
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
                initialdir="documents",
                title="Export Results"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.status_var.set(f"Results exported to {os.path.basename(file_path)}")
                
        except Exception as e:
            self.handle_error(f"Failed to export results: {str(e)}")
