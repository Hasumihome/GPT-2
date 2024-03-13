import torch
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Training on GPU cannot be performed.")
# x = torch.rand(5, 3)
# print(x)
print("torch_ver:" + torch.__version__)

import transformers
print("transformers_ver:" + transformers.__version__)

try:
    import MeCab
    print("MeCabは正しくインストールされています。")
except ImportError:
    print("MeCabがインストールされていません。")

m = MeCab.Tagger("-Owakati")
text = "これはテストです。"
print(m.parse(text))

import sys
print("sis_ver:" + sys.version)
print(sys.executable)


