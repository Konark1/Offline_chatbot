import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from main import StudyBot
import threading
import os

class StudyBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("📘 StudyBot AI")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')

        self.status_var = tk.StringVar(value="Ready")
        self.mode_var = None
        self.selected_file = None
        self.indepth_completed = False

        self.bot = StudyBot()
        self.processing_thread = None

        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TButton', padding=5, font=('Segoe UI', 10))
        style.configure('TLabel', background='#f0f0f0', font=('Segoe UI', 10))

    def create_widgets(self):
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.Frame(main_container)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        mode_frame = ttk.LabelFrame(top_frame, text="Mode", padding="5")
        mode_frame.pack(side=tk.LEFT, padx=(0, 10))

        self.mode_var = tk.StringVar(value="ask")
        modes = [
            ("Ask Formula", "ask"),
            ("Search", "search"),
            ("In-depth Study", "index"),
            ("Summarize", "summary")
        ]
        for text, value in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, 
                          value=value, command=self.update_interface).pack(side=tk.LEFT, padx=5)

        pdf_frame = ttk.Frame(top_frame)
        pdf_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.file_button = ttk.Button(pdf_frame, text="📂 Select PDF", command=self.select_pdf)
        self.file_button.pack(side=tk.LEFT, padx=(0, 5))

        self.file_label = ttk.Label(pdf_frame, text="No PDF selected")
        self.file_label.pack(side=tk.LEFT, padx=10)

        query_frame = ttk.LabelFrame(main_container, text="Query", padding="5")
        query_frame.pack(fill=tk.X, pady=(0, 10))

        self.query_entry = ttk.Entry(query_frame, font=("Segoe UI", 12))
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 10))
        self.query_entry.bind("<Return>", lambda event: self.handle_query())

        self.submit_btn = tk.Button(query_frame, text="🔍 Submit", command=self.handle_query,
                                  bg="#2196F3", fg="white", font=("Segoe UI", 10, "bold"),
                                  relief=tk.FLAT, padx=20)
        self.submit_btn.pack(side=tk.LEFT)

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

        self.status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var,
            relief=tk.SUNKEN, 
            anchor=tk.W,
            padding=(5, 2)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def select_pdf(self):
        try:
            documents_dir = os.path.abspath("documents")
            if not os.path.exists(documents_dir):
                os.makedirs(documents_dir)
            file_path = filedialog.askopenfilename(
                initialdir=documents_dir,
                title="Select PDF",
                filetypes=(("PDF Files", "*.pdf"),)
            )
            if file_path:
                filename = os.path.basename(file_path)
                destination = os.path.join(documents_dir, filename)
                if os.path.normcase(file_path) != os.path.normcase(destination):
                    import shutil
                    shutil.copy2(file_path, destination)
                self.selected_file = filename
                self.file_label.config(text=f"Selected: {filename}")
                self.status_var.set(f"PDF loaded: {filename}")
                self.update_interface()
            else:
                self.status_var.set("PDF selection cancelled")
        except Exception as e:
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
        elif mode == "search":
            self.query_entry.config(state="normal")
            self.file_button.config(state="disabled")
            self.query_entry.insert(0, "Enter any question...")
        elif mode == "index":
            self.file_button.config(state="normal")
            if self.indepth_completed:
                self.query_entry.config(state="normal")
                self.query_entry.insert(0, "Enter question for detailed study...")
                self.file_button.config(state="disabled")
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
        self.query_entry.config(state='disabled')
        self.submit_btn.config(state='disabled')
        self.file_button.config(state='disabled')

    def enable_interface(self):
        self.update_interface()

    def handle_query(self):
        query = self.query_entry.get().strip()
        mode = self.mode_var.get()
        self.output_box.delete(1.0, tk.END)
        self.status_var.set("Processing...")  # <-- Show processing status
        self.submit_btn.config(state='disabled')  # <-- Disable submit button
        try:
            if mode == "search":
                if not query or query == "Enter any question...":
                    raise ValueError("Please enter a question")
                def search_callback(result):
                    self.output_box.insert(tk.END, result)
                    self.status_var.set("Search completed")
                    self.submit_btn.config(state='normal')  # <-- Re-enable submit
                self.run_in_thread(
                    target=self.bot.search_query,
                    args=(query,),
                    callback=search_callback
                )
                return
            elif mode == "index":
                if not self.selected_file:
                    raise ValueError("Please select a PDF file first")
                # If in-depth study is not done, run indexing
                if not self.indepth_completed:
                    pdf_path = os.path.join(os.path.abspath("documents"), self.selected_file)
                    def index_callback(result):
                        self.output_box.insert(tk.END, result)
                        self.indepth_completed = True
                        self.status_var.set("In-depth study completed")
                        self.update_interface()
                        self.submit_btn.config(state='normal')
                    self.run_in_thread(
                        target=self.bot.index_chapter,
                        args=(pdf_path,),
                        callback=index_callback
                    )
                    return
                else:
                    # If already indexed, treat as in-depth query
                    if not query or query == "Enter question for detailed study...":
                        raise ValueError("Please enter a question for detailed study")
                    def indepth_callback(result):
                        self.output_box.insert(tk.END, result)
                        self.status_var.set("Query completed")
                        self.submit_btn.config(state='normal')
                    self.run_in_thread(
                        target=self.bot.indepth_query,
                        args=(query,),
                        callback=indepth_callback
                    )
                return
            elif mode == "summary":
                if not self.indepth_completed:
                    raise ValueError("Please complete In-depth Study first")
                def summary_callback(result):
                    self.output_box.insert(tk.END, result)
                    self.status_var.set("Summary generated")
                    self.submit_btn.config(state='normal')  # <-- Re-enable submit
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
                    self.submit_btn.config(state='normal')  # <-- Re-enable submit
                self.run_in_thread(
                    target=self.bot.get_formula,
                    args=(query,),
                    callback=formula_callback
                )
                return
        except Exception as e:
            self.handle_error(str(e))

    def handle_error(self, error_message):
        messagebox.showerror("Error", error_message)
        self.status_var.set("Error occurred")
        self.enable_interface()

if __name__ == "__main__":
    root = tk.Tk()
    app = StudyBotGUI(root)
    root.mainloop()
