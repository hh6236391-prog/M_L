import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.photo_analyzer import PhotoAnalyzer
    from src.utils import load_config, get_image_files, save_best_photos, count_images_in_folder
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required dependencies are installed.")
    sys.exit(1)

class PhotoSelectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Best Photo Selector")
        self.root.geometry("1000x700")
        
        self.analyzer = None
        self.results = []
        self.current_folder = ""
        
        self.setup_ui()
        self.load_config()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Best Photo Selector", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Folder selection frame
        folder_frame = ttk.LabelFrame(main_frame, text="Folder Selection", padding="5")
        folder_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Input folder selection
        ttk.Label(folder_frame, text="Input Folder:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.folder_var = tk.StringVar()
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var, width=60)
        folder_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        ttk.Button(folder_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2, padx=5, pady=2)
        
        # Scan button
        ttk.Button(folder_frame, text="Scan Folder", command=self.scan_folder).grid(row=0, column=3, padx=5, pady=2)
        
        # Folder info
        self.folder_info_var = tk.StringVar(value="No folder selected")
        folder_info_label = ttk.Label(folder_frame, textvariable=self.folder_info_var, foreground="blue")
        folder_info_label.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=2)
        
        # Output folder selection
        ttk.Label(folder_frame, text="Output Folder:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.output_var = tk.StringVar(value="results/best_photos")
        output_entry = ttk.Entry(folder_frame, textvariable=self.output_var, width=60)
        output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        ttk.Button(folder_frame, text="Browse", command=self.browse_output).grid(row=2, column=2, padx=5, pady=2)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Analysis Parameters", padding="5")
        params_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(params_frame, text="Number of best photos:").grid(row=0, column=0, sticky=tk.W)
        self.num_photos_var = tk.StringVar(value="1")
        num_spinbox = ttk.Spinbox(params_frame, from_=1, to=5, textvariable=self.num_photos_var, width=10)
        num_spinbox.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Minimum score:").grid(row=0, column=2, sticky=tk.W, padx=10)
        self.min_score_var = tk.StringVar(value="0.2")
        min_score_spinbox = ttk.Spinbox(params_frame, from_=0.0, to=1.0, increment=0.05, 
                                       textvariable=self.min_score_var, width=10)
        min_score_spinbox.grid(row=0, column=3, padx=5, sticky=tk.W)
        
        # Search options
        self.recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Search subfolders", variable=self.recursive_var).grid(row=1, column=0, sticky=tk.W, pady=5)
        
        # Analyze button
        self.analyze_btn = ttk.Button(main_frame, text="Analyze Photos", 
                                     command=self.analyze_photos, state=tk.DISABLED)
        self.analyze_btn.grid(row=3, column=0, columnspan=3, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Progress label
        self.progress_var = tk.StringVar(value="Ready")
        progress_label = ttk.Label(main_frame, textvariable=self.progress_var)
        progress_label.grid(row=5, column=0, columnspan=3, pady=2)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        self.results_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Results treeview
        columns = ('rank', 'filename', 'score', 'faces', 'sharpness', 'brightness', 'status')
        self.results_tree = ttk.Treeview(self.results_frame, columns=columns, show='headings', height=12)
        
        # Define headings
        self.results_tree.heading('rank', text='Rank')
        self.results_tree.heading('filename', text='Filename')
        self.results_tree.heading('score', text='Score')
        self.results_tree.heading('faces', text='Faces')
        self.results_tree.heading('sharpness', text='Sharpness')
        self.results_tree.heading('brightness', text='Brightness')
        self.results_tree.heading('status', text='Status')
        
        # Configure columns
        self.results_tree.column('rank', width=50)
        self.results_tree.column('filename', width=250)
        self.results_tree.column('score', width=80)
        self.results_tree.column('faces', width=60)
        self.results_tree.column('sharpness', width=80)
        self.results_tree.column('brightness', width=80)
        self.results_tree.column('status', width=200)
        
        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Save button
        self.save_btn = ttk.Button(main_frame, text="Save Best Photos", 
                                  command=self.save_results, state=tk.DISABLED)
        self.save_btn.grid(row=7, column=0, columnspan=3, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to select folder")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)
        folder_frame.columnconfigure(1, weight=1)
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(0, weight=1)
    
    def load_config(self):
        """Load configuration"""
        try:
            config_path = "config/default_config.yaml"
            if os.path.exists(config_path):
                config = load_config(config_path)
            else:
                from src.utils import get_default_config
                config = get_default_config()
            self.analyzer = PhotoAnalyzer(config)
            self.status_var.set("Configuration loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load configuration: {e}")
            self.status_var.set("Configuration error")
    
    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select Folder Containing Photos")
        if folder:
            self.folder_var.set(folder)
            self.current_folder = folder
            self.scan_folder()
    
    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_var.set(folder)
    
    def scan_folder(self):
        """Scan selected folder for images"""
        folder_path = self.folder_var.get()
        if not folder_path or not os.path.exists(folder_path):
            messagebox.showwarning("Warning", "Please select a valid folder first")
            return
        
        try:
            self.status_var.set("Scanning folder for images...")
            self.root.update()
            
            # Count images
            recursive = self.recursive_var.get()
            image_count = count_images_in_folder(folder_path, recursive=recursive)
            
            if image_count == 0:
                self.folder_info_var.set(f"No images found in: {os.path.basename(folder_path)}")
                self.analyze_btn.config(state=tk.DISABLED)
                messagebox.showwarning(
                    "No Images Found", 
                    f"No supported images found in:\n{folder_path}\n\n"
                    f"Supported formats: JPG, JPEG, PNG, BMP, TIFF, WEBP\n"
                    f"Search subfolders: {'Yes' if recursive else 'No'}"
                )
                self.status_var.set("No images found in selected folder")
            else:
                search_type = "including subfolders" if recursive else "in main folder only"
                self.folder_info_var.set(f"Found {image_count} images {search_type}")
                self.analyze_btn.config(state=tk.NORMAL)
                self.status_var.set(f"Ready to analyze {image_count} images")
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not scan folder: {e}")
            self.status_var.set("Error scanning folder")
    
    def analyze_photos(self):
        if not self.folder_var.get():
            messagebox.showerror("Error", "Please select an input folder")
            return
        
        try:
            self.analyze_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)
            self.progress['value'] = 0
            self.progress.start()
            
            # Get parameters
            num_photos = int(self.num_photos_var.get())
            min_score = float(self.min_score_var.get())
            recursive = self.recursive_var.get()
            
            # Analyze in separate steps to avoid GUI freeze
            self.root.after(100, lambda: self._perform_analysis(num_photos, min_score, recursive))
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            self.analyze_btn.config(state=tk.NORMAL)
            self.progress.stop()
            self.progress['value'] = 0
    
    def _perform_analysis(self, num_photos: int, min_score: float, recursive: bool):
        try:
            # Get image files
            image_files = get_image_files(self.folder_var.get(), recursive=recursive)
            
            if not image_files:
                messagebox.showerror("Error", "No images found in the selected folder")
                self.analyze_btn.config(state=tk.NORMAL)
                return
            
            # Analyze images
            self.results = []
            total_images = len(image_files)
            
            for i, image_path in enumerate(image_files):
                self.progress_var.set(f"Analyzing {i+1}/{total_images}: {image_path.name}")
                self.progress['value'] = (i / total_images) * 100
                self.root.update()
                
                analysis = self.analyzer.analyze_image(image_path)
                self.results.append(analysis)
            
            # Filter and sort results
            valid_results = [r for r in self.results if r.get('final_score', 0) >= min_score]
            valid_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            best_photos = valid_results[:num_photos]
            
            # Display results
            self._display_results(best_photos)
            
            self.progress['value'] = 100
            self.progress_var.set(f"Analysis complete! Found {len(best_photos)} best photos")
            self.status_var.set(f"Analysis complete - {len(best_photos)} best photos found")
            
            self.save_btn.config(state=tk.NORMAL)
            
            if len(best_photos) > 0:
                messagebox.showinfo(
                    "Analysis Complete", 
                    f"Found {len(best_photos)} best photos!\n\n"
                    f"Top photo: {best_photos[0]['file_name']}\n"
                    f"Score: {best_photos[0]['final_score']:.3f}"
                )
            else:
                messagebox.showwarning(
                    "No Suitable Photos", 
                    "No photos met the minimum quality criteria.\n"
                    "Try lowering the minimum score threshold."
                )
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            self.status_var.set("Analysis failed")
        finally:
            self.analyze_btn.config(state=tk.NORMAL)
            self.progress.stop()
            self.progress['value'] = 0
    
    def _display_results(self, best_photos: list):
        """Display results in treeview"""
        # Clear existing results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Add new results
        for i, photo in enumerate(best_photos):
            status = "✓ Good" if not photo['rejection_reasons'] else "⚠ " + ", ".join(photo['rejection_reasons'][:1])
            
            self.results_tree.insert('', 'end', values=(
                i + 1,
                photo['file_name'],
                f"{photo.get('final_score', 0):.3f}",
                photo.get('num_faces', 0),
                f"{photo.get('sharpness', 0):.1f}",
                f"{photo.get('brightness', 0):.1f}",
                status
            ))
    
    def save_results(self):
        """Save best photos to output directory"""
        try:
            valid_results = [r for r in self.results if r.get('final_score', 0) >= float(self.min_score_var.get())]
            valid_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
            num_photos = int(self.num_photos_var.get())
            best_photos = valid_results[:num_photos]
            
            if not best_photos:
                messagebox.showwarning("Warning", "No photos to save")
                return
            
            output_dir = self.output_var.get()
            save_best_photos(best_photos, output_dir)
            
            # Count actual saved files
            output_path = Path(output_dir)
            saved_images = list(output_path.glob("best_*.jpg")) + list(output_path.glob("best_*.png")) + list(output_path.glob("best_*.jpeg"))
            
            messagebox.showinfo(
                "Success", 
                f"Saved {len(saved_images)} best photos to:\n{output_dir}\n\n"
                f"Including:\n"
                f"• Best photos (copied)\n"
                f"• Analysis reports\n"
                f"• Ranking information"
            )
            
            self.status_var.set(f"Saved {len(saved_images)} photos to {output_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not save results: {e}")
            self.status_var.set("Save failed")

def main():
    root = tk.Tk()
    app = PhotoSelectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()