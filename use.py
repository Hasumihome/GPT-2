import tkinter as tk
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from googletrans import Translator

# モデルとトークナイザーのロード
model = GPT2LMHeadModel.from_pretrained("./gpt2_finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_text():
    # 入力テキストの取得
    input_text = text_entry.get()
    print(f"入力テキスト: {input_text}")  # 入力テキストを表示

    # 入力テキストを英語に翻訳
    translator = Translator()
    input_text_en = translator.translate(input_text, dest='en').text
    print(f"翻訳後のテキスト (英語): {input_text_en}")  # 翻訳後のテキストを表示

    # テキストの生成
    input_ids = tokenizer.encode(input_text_en, return_tensors='pt')
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"生成されたテキスト (英語): {output_text}")  # 生成されたテキストを表示

    # 生成したテキストを日本語に翻訳
    output_text_ja = translator.translate(output_text, dest='ja').text
    print(f"生成されたテキスト (日本語): {output_text_ja}")  # 日本語に翻訳されたテキストを表示
    # 生成したテキストの表示
    result_label['text'] = output_text_ja

# GUIの作成
root = tk.Tk()
root.title("GPT-2")

# 入力フィールドの作成
text_entry = tk.Entry(root, width=50)
text_entry.pack()

# ボタンの作成
generate_button = tk.Button(root, text="Generate", command=generate_text)
generate_button.pack()

# 結果表示ラベルの作成、wraplengthを設定して自動改行を有効にする
result_label = tk.Label(root, text="", wraplength=400)  # wraplengthはピクセル単位
result_label.pack()

root.mainloop()
