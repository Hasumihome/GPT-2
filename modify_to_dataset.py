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

# テキストの翻訳
def translate_text(text):
    model_name = 'Helsinki-NLP/opus-mt-ja-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # GPUが利用可能な場合はGPUを使用し、そうでない場合はCPUを使用します
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    sentences = text.split('。')
    translated_text = ''

    for sentence in sentences:
        # テキストをトークン化してモデルに入力できる形式に変換
        inputs = tokenizer.encode(sentence, return_tensors="pt", max_length=512, truncation=True).to(device)

        # # GPUメモリ使用量を表示
        # if torch.cuda.is_available():
        #     print(f"Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        #     print(f"Reserved: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")

        # モデルで翻訳を実行
        outputs = model.generate(inputs, max_length=400, num_beams=4, early_stopping=True)

        # 翻訳結果をデコード
        translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(sentence + '\n' + translated_sentence)
        translated_text += translated_sentence + '.\n'  # 各文を新しい行に配置

    return translated_text.strip()

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