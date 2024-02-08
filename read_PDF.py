from PyPDF2 import PdfReader
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog
import shutil
from chardet import UniversalDetector

def detect_encoding(file_path):
    detector = UniversalDetector()
    with open(file_path, 'rb') as f:
        for line in f:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
    return detector.result['encoding']

def convert_vertical_to_horizontal(file_path, encoding):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.read().split('\n')
    paragraphs = ' '.join(line if line else '\n' for line in lines)
    return paragraphs

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

# 縦書きのテキストを横書きに変換
print("変換を開始します...")
encoding = detect_encoding(text_file_path)
with open(text_file_path, 'r', encoding='utf-8', errors='ignore') as file:
    horizontal_text = convert_vertical_to_horizontal(text_file_path, encoding)
with open(text_file_path, 'w', encoding=encoding) as file:
    file.write(horizontal_text)
print("変換が完了しました。")

# 移動先のディレクトリ
destination_directory = 'Text/'

# ファイルを移動
shutil.move(text_file_path, destination_directory)