from summarizer import Summarizer
from PyPDF2 import PdfReader
from tqdm import tqdm
import textwrap
import tkinter as tk
from tkinter import filedialog
import shutil

# GUIウィンドウを作成
root = tk.Tk()
root.withdraw()  # メインウィンドウを表示しない

# ファイル選択ダイアログを表示
PDF = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])

with open(PDF, 'rb') as file:
    reader = PdfReader(file)
    model = Summarizer()
    summary_file_path = PDF + '_Summary.txt'
    with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
        for page_num in tqdm(range(len(reader.pages)), desc="Reading pages"):
            text = reader.pages[page_num].extract_text()
            print(f"Summarizing page {page_num}...")
            chunks = textwrap.wrap(text, 512)
            for chunk in chunks:
                result = model(chunk, min_length=200)
                summary = "".join(result)
                # Remove NUL characters
                summary = summary.replace('\0', '')
                print(f"Writing summary of page {page_num} to file...")
                summary_file.write(f"Summary of page {page_num}:\n{summary}\n\n")
# 移動先のディレクトリ
destination_directory = 'Summary/'

# ファイルを移動
shutil.move(summary_file_path, destination_directory)