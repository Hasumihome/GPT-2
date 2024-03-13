import MeCab
import re
import os
import psutil
from sklearn.model_selection import train_test_split
from transformers import MarianMTModel, MarianTokenizer, GPT2Tokenizer
import torch

# MeCabの初期化
mecab = MeCab.Tagger("-Owakati")

# ストップワードの定義
stop_words = ["と", "から"]

# テキストのクリーニング
def clean_text(text):
    text = re.sub(r'\n', ' ', text)  # 改行を空白に置換
    text = re.sub(r'\s+', ' ', text)  # 連続する空白を一つの空白に置換
    text = text.strip()  # 文字列の先頭と末尾の空白を削除
    return text

# 形態素解析
def tokenize(text):
    text = mecab.parse(text)
    words = text.split(' ')
    return words

# ストップワードの置換
def replace_stopwords(words):
    words = [' ' if word in stop_words else word for word in words]
    return words

# テキストの前処理
def preprocess_text(text):
    text = clean_text(text)
    words = tokenize(text)
    words = replace_stopwords(words)
    return words

import torch
from transformers import MarianMTModel, MarianTokenizer

def adjust_parameters_based_on_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        available_memory = total_memory - reserved_memory
        
        if available_memory > 8e9:  # 8GB以上の空きがある場合
            batch_size = 16
            max_length = 512
        elif available_memory > 4e9:  # 4GB以上の空きがある場合
            batch_size = 8
            max_length = 256
        else:  # それ以下の場合
            batch_size = 4
            max_length = 128
    else:
        batch_size = 2
        max_length = 128
    
    return batch_size, max_length

def translate_text(text):
    model_name = 'Helsinki-NLP/opus-mt-ja-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    sentences = split_text_custom(text)
    translated_text = ''

    # GPUのメモリに基づいてバッチサイズと最大長を調整
    batch_size, max_length = adjust_parameters_based_on_gpu_memory()
    print(f"Batch size: {batch_size}, Max length: {max_length}")

    sentence_batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]

    for batch in sentence_batches:
        batch_input_ids = []
        for sentence in batch:
            inputs = tokenizer.encode(sentence, return_tensors="pt", max_length=max_length, truncation=True).to(device)
            batch_input_ids.append(inputs)

        with torch.no_grad():
            outputs = [model.generate(input_ids, max_length=max_length, num_beams=4, early_stopping=True) for input_ids in batch_input_ids]

        for sentence, output in zip(batch, outputs):
            translated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
            print(sentence + '\n' + translated_sentence)
            translated_text += translated_sentence + '.\n'

        torch.cuda.empty_cache()

    return translated_text.strip()

def split_text_custom(text):
    # 空行でテキストを分割する
    sections = text.split('\n\n')
    
    # 各セクションをさらに処理する
    sentences = []
    for section in sections:
        # ここで、セクション内の文をさらに細かく分割するカスタムロジックを適用できます
        sentences.extend(section.split('。'))
    
    # 空の要素を削除
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

# ファイルの前処理
def preprocess_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        translated_text = translate_text(text)
    new_file_path = file_path.replace('StudyData', 'Dataset/English_data')  # パスを変更
    with open(new_file_path, 'w', encoding='utf-8') as f:
        f.write(translated_text)

# Ensure Dataset/English_data directory exists
if not os.path.exists('./Dataset/English_data'):
    os.makedirs('./Dataset/English_data')

# StudyDataフォルダ内のすべてのテキストファイルの数を取得
folder_path = './StudyData'
total_files = len([name for name in os.listdir(folder_path) if name.endswith('.txt')])

# Ensure Dataset directory exists
if not os.path.exists('./Dataset'):
    os.makedirs('./Dataset')

processed_files = 0

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        preprocess_file(file_path)
        processed_files += 1
        print(f'進行中: {processed_files / total_files * 100:.2f}% 完了')

# テキストデータのロード
texts = []
folder_path = './Dataset/English_data'  # 変更されたフォルダパス
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            texts.append(f.read())

# データを訓練データとテストデータに分割
if len(texts) < 2:
    print("Error: Not enough samples to split into training and test sets.")
else:
    train_texts, test_texts = train_test_split(texts, test_size=0.2)

# トークナイザーの初期化とパディングトークンの設定
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<pad>')

# テキストをトークン化
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# データセットの作成
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids']
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = TextDataset(train_encodings)
test_dataset = TextDataset(test_encodings)

# Ensure TextData directory exists
if not os.path.exists('./Dataset/pt'):
    os.makedirs('./Dataset/pt')

# データセットの保存
torch.save(train_dataset, './Dataset/pt/train_dataset.pt')
torch.save(test_dataset, './Dataset/pt/test_dataset.pt')