import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# CUDA（GPUサポート）が利用可能かどうかを確認
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Training on GPU cannot be performed.")

# デバイスをGPU（CUDA）に設定
device = torch.device("cuda")

def adjust_batch_size():
    print("adjust_batch_size function is called")  # 関数が呼び出されたことを確認するためのデバッグメッセージ
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        available_memory = total_memory - reserved_memory
        print(f"Available GPU memory: {available_memory / 1e9} GB")  # 利用可能なメモリをGB単位で出力
        
        if available_memory > 10e9:  # 10GB以上の場合
            batch_size = 64
        elif available_memory > 5e9:  # 5GB以上の場合
            batch_size = 32
        else:
            batch_size = 16
    else:
        batch_size = 1  # GPUが利用不可能な場合
    
    print(f"Batch size set to: {batch_size}")  # 設定されたバッチサイズを出力
    return batch_size

# トークナイザとモデルの初期化
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

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
    per_device_train_batch_size=adjust_batch_size(), # バッチサイズを動的に調整
    save_steps=500, # 500ステップごとにモデルを保存
    save_total_limit=2,
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