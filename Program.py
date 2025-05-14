import os
import cv2 
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Button, Label, Frame, messagebox
from PIL import Image, ImageTk
from collections import defaultdict
import math

class SignatureRecognizer:
    def __init__(self):
        self.database_path = "Signature"
        self.signature_groups = defaultdict(list)
        self.load_database()
        
    def load_database(self):
        """Load signature database and group by name"""
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
            messagebox.showinfo("Info", f"Created database directory at {self.database_path}")
            return
            
        for filename in os.listdir(self.database_path):
            # Check if file is PNG (by content, not extension)
            file_path = os.path.join(self.database_path, filename)
            try:
                # Try to open as image to verify it's a PNG
                with Image.open(file_path) as img:
                    if img.format != 'PNG':
                        continue
            except:
                continue
                
            # Parse filename (name)(number)
            name_part = os.path.splitext(filename)[0]  # Remove any extension
            name = ''.join([c for c in name_part if not c.isdigit()])
            number = ''.join([c for c in name_part if c.isdigit()])
            
            # Load and process image
            processed_img, minutiae = self.process_image(file_path)
            
            # Add to group
            self.signature_groups[name].append({
                'path': file_path,
                'processed_img': processed_img,
                'minutiae': minutiae,
                'number': number
            })
    
    def process_image(self, img_path):
        """Process image: Otsu binarization, skeletonization, minutiae extraction"""
        # Load grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image file {img_path}")
        
        # Otsu binarization
        _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Skeletonization
        skeleton = skeletonize(binary_img // 255)
        skeleton_img = img_as_ubyte(skeleton)
        
        # Minutiae extraction
        minutiae = self.extract_minutiae(skeleton_img)
        
        return skeleton_img, minutiae
    
    def extract_minutiae(self, skeleton_img):
        """Extract minutiae points from skeletonized image"""
        minutiae = []
        
        # Find endpoints and bifurcations
        for i in range(1, skeleton_img.shape[0]-1):
            for j in range(1, skeleton_img.shape[1]-1):
                if skeleton_img[i,j] == 255:
                    # Count neighbors
                    neighbors = skeleton_img[i-1:i+2, j-1:j+2]
                    num_neighbors = np.sum(neighbors) // 255 - 1
                    
                    # Endpoint (1 neighbor) or bifurcation (3+ neighbors)
                    if num_neighbors == 1 or num_neighbors >= 3:
                        minutiae.append((j, i))  # (x, y)
        
        return minutiae
    
    def compare_minutiae(self, minutiae1, minutiae2, threshold=10):
        """Compare two minutiae lists and return number of matches"""
        matches = 0
        used = set()
        
        for m1 in minutiae1:
            for idx, m2 in enumerate(minutiae2):
                if idx not in used:
                    distance = math.sqrt((m1[0]-m2[0])**2 + (m1[1]-m2[1])**2)
                    if distance < threshold:
                        matches += 1
                        used.add(idx)
                        break
        
        return matches
    
    def compare_with_database(self, query_minutiae):
        """Compare signature with database and return similarity results"""
        results = {}
        
        for name, signatures in self.signature_groups.items():
            total_matches = 0
            for sig in signatures:
                matches = self.compare_minutiae(query_minutiae, sig['minutiae'])
                total_matches += matches
            
            avg_matches = total_matches / len(signatures) if signatures else 0
            results[name] = avg_matches
        
        return results

class SignatureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Signature Recognition")
        self.recognizer = SignatureRecognizer()
        
        # GUI setup
        self.setup_ui()
    
    def setup_ui(self):
        """Configure user interface"""
        # Button frame
        button_frame = Frame(self.root)
        button_frame.pack(pady=10)
        
        # Image selection button
        self.select_btn = Button(button_frame, text="Select Signature", command=self.load_signature)
        self.select_btn.pack(side="left", padx=5)
        
        # Compare button
        self.compare_btn = Button(button_frame, text="Compare", command=self.compare_signature, state="disabled")
        self.compare_btn.pack(side="left", padx=5)
        
        # Image label
        self.image_label = Label(self.root)
        self.image_label.pack(pady=10)
        
        # Result label
        self.result_label = Label(self.root, text="Select signature to compare")
        self.result_label.pack(pady=10)
    
    def load_signature(self):
        """Load signature for comparison"""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.*")])
        if file_path:
            try:
                # Verify it's a PNG file
                with Image.open(file_path) as img:
                    if img.format != 'PNG':
                        messagebox.showerror("Error", "Please select a PNG format image")
                        return
                
                # Process image
                self.processed_img, self.query_minutiae = self.recognizer.process_image(file_path)
                
                # Display processed image
                img = Image.fromarray(self.processed_img)
                img.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(img)
                
                self.image_label.config(image=photo)
                self.image_label.image = photo
                
                self.compare_btn.config(state="normal")
                self.result_label.config(text="Signature loaded. Click 'Compare' to analyze.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def compare_signature(self):
        """Compare signature with database"""
        if hasattr(self, 'query_minutiae'):
            if not self.recognizer.signature_groups:
                messagebox.showwarning("Warning", "Database is empty. Add signatures to the Signature folder.")
                return
                
            results = self.recognizer.compare_with_database(self.query_minutiae)
            
            # Plot results
            self.plot_results(results)
            
            # Find best match
            best_match = max(results.items(), key=lambda x: x[1]) if results else (None, 0)
            self.result_label.config(text=f"Best match: {best_match[0]} (score: {best_match[1]:.2f})")
    
    def plot_results(self, results):
        """Plot comparison results as bar chart"""
        if not results:
            messagebox.showinfo("Info", "No comparison results available")
            return
            
        plt.figure(figsize=(10, 5))
        names = list(results.keys())
        values = list(results.values())
        
        plt.bar(names, values)
        plt.xlabel('Signature Groups')
        plt.ylabel('Similarity Score')
        plt.title('Signature Comparison Results')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = Tk()
    app = SignatureGUI(root)
    root.mainloop()
