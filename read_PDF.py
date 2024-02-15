from PyPDF2 import PdfReader
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
import shutil
from chardet import UniversalDetector
import pytesseract
from PIL import Image
import pdf2image

def detect_encoding(file_path):
    detector = UniversalDetector()
    with open(file_path, 'rb') as f:
        for line in f:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
    return detector.result['encoding']

def pdf_to_text_tesseract(pdf_path):
    # PDFを画像に変換
    images = pdf2image.convert_from_path(pdf_path)
    
    text = ""
    for image in images:
        # Tesseractを使用して画像からテキストを抽出
        text += pytesseract.image_to_string(image, lang='jpn')
    
    return text

# GUIウィンドウを作成
root = tk.Tk()
root.withdraw()  # メインウィンドウを表示しない

# ファイル選択ダイアログを表示
PDF = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])

# ファイルのエンコーディングを検出
encoding = detect_encoding(PDF)

with open(PDF, 'rb') as file:
    reader = PdfReader(file)
    text_file_path = PDF + '_Text.txt'
    with open(text_file_path, 'w', encoding='utf-8') as text_file:
        for page_num in tqdm(range(len(reader.pages)), desc="Reading pages"):
            text = reader.pages[page_num].extract_text()
            # ヌル文字を削除
            text = text.replace('\x00', '')
            print(f"Writing text of page {page_num} to file...")
            text_file.write(f"Text of page {page_num}:\n\n{text}\n\n")

# PDFからテキストへの変換
print("変換を開始します...")
text = pdf_to_text_tesseract(PDF)
with open(text_file_path, 'w', encoding='utf-8') as file:
    file.write(text)
print("変換が完了しました。")

# 移動先のディレクトリ
destination_directory = 'Text/'

# ファイルを移動
shutil.move(text_file_path, destination_directory)