import os
import torch
import tkinter as tk
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from modify_to_dataset import preprocess_file, preprocess_text, TextDataset

# CUDA（GPUサポート）が利用可能かどうかを確認
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Training on GPU cannot be performed.")

# デバイスをGPU（CUDA）に設定
device = torch.device("cuda")

# トークナイザとモデルの初期化
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

# # フォルダ内のすべてのテキストファイルをリストアップ
# folder_path = "StudyData"
# file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]

# # 前回の訓練時に使用したファイルのリストをロード
# old_file_paths = []
# if os.path.exists('file_list.txt'):
#     with open('file_list.txt', 'r') as f:
#         old_file_paths = f.read().splitlines()

# # 新しく追加されたファイルを特定
# new_file_paths = list(set(file_paths) - set(old_file_paths))

# # 新しく追加されたファイルからデータセットを作成
# texts = []
# for file_path in new_file_paths:
#     with open(file_path, 'r', encoding='utf-8') as f:
#         text = f.read()
#         words = preprocess_text(text)  # テキストの前処理
#         texts.append(' '.join(words))  # 単語をスペースで結合

# # データを訓練データとテストデータに分割
# train_texts, test_texts = train_test_split(texts, test_size=0.2)

# # テキストをトークン化
# train_encodings = tokenizer(train_texts, truncation=True, padding=True)
# test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# # データセットの作成
# train_dataset = TextDataset(train_encodings)
# test_dataset = TextDataset(test_encodings)

# データセットのロード
train_dataset = torch.load('./Dataset/pt/train_dataset.pt')
test_dataset = torch.load('./Dataset/pt/test_dataset.pt')

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# トレーニングの設定
training_args = TrainingArguments(
    output_dir="./gpt2_finetuned", # モデルと訓練結果を保存するディレクトリ
    overwrite_output_dir=True,
    num_train_epochs=1, # 訓練エポック数
    per_device_train_batch_size=1, # バッチサイズ
    save_steps=500, # 500ステップごとにモデルを保存
    save_total_limit=2,
    save_strategy='epoch', # 各エポックの終わりにモデルを保存
)

# トレーナーの初期化と訓練の開始
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,  # 'train'スプリットを指定
)

# 訓練の開始
trainer.train()

# モデルの保存
trainer.save_model("./gpt2_finetuned")

# 現在のファイルリストを保存
# with open('file_list.txt', 'w') as f:
#     for file_path in file_paths:
#         f.write(f"{file_path}\n")
