from PyPDF2 import PdfReader
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
import shutil
from chardet import UniversalDetector
import pytesseract
from PIL import Image
import pdf2image
from typing import List
import os

def detect_encoding(file_path):
    detector = UniversalDetector()
    with open(file_path, 'rb') as f:
        for line in f:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
    return detector.result['encoding']

def convert_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """
    PDFファイルを画像のリストに変換します。
    """
    return pdf2image.convert_from_path(pdf_path)

def extract_text_from_image(image: Image.Image) -> str:
    """
    画像からテキストを抽出します。
    """
    return pytesseract.image_to_string(image, lang='jpn')

def pdf_to_text_tesseract(pdf_path: str) -> str:
    """
    PDFファイルからテキストを抽出します。
    """
    images = convert_pdf_to_images(pdf_path)
    text = ""
    for page_num, image in enumerate(tqdm(images, desc="Processing pages"), start=1):
        print(f"Page {page_num}/{len(images)}: Text extraction in progress...")
        text += extract_text_from_image(image)
    return text

# GUIウィンドウを作成
root = tk.Tk()
root.withdraw()  # メインウィンドウを表示しない

# 複数のPDFファイル選択ダイアログを表示
pdf_files = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])

# 移動先のディレクトリ
destination_directory = 'Text/'

# destination_directoryが存在しない場合にディレクトリを作成
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory, exist_ok=True)

# 選択されたPDFファイルごとに処理
for PDF in pdf_files:
    print(f"処理中: {PDF}")
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

    # ファイルを移動
    shutil.move(text_file_path, destination_directory)
    print(f"{PDF} の処理が完了しました。")
