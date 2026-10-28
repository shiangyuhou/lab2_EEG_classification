# Lab2 EEG Classification

使用 PyTorch 自行實作 EEGNet，對課程提供的 BCI Competition III、IIIb EEG 資料進行二分類，並比較不同訓練設定。

## 目前進度

已建立 Git 版本管理；保留現有程式。EEGNet 模型仍待實作，訓練、評估與繪圖也尚需修正及驗證。目前沒有已驗證的實驗結果。

## 專案結構

```text
lab2_EEG_classification/
├── main.py             # 訓練、評估與繪圖
├── dataloader.py       # 載入及處理 EEG 資料
├── models/
│   └── EEGNet.py       # EEGNet / DeepConvNet 模型空殼
└── data/               # 本機資料，不納入 Git
```

將課程提供的 `S4b_train.npz`、`X11b_train.npz`、`S4b_test.npz`、`X11b_test.npz` 放入 `data/`。資料集、模型權重及產生的結果透過 SFTP 或 rsync 另外傳輸。

## 環境與執行方式

目前程式匯入的第三方套件為 PyTorch、NumPy、pandas、tqdm、Matplotlib 與 torchsummary。尚未建立或驗證固定版本的環境；PyTorch 安裝需依遠端主機的 CPU / CUDA 環境選擇。

執行./run.sh 

```bash
for seed in 42 123 2026 1026 3600
do
    python main.py -mode D -type 0 -seed 123 -lr 0.01
    python main.py -mode E -type 0 -seed 42 -lr 0.005
    python main.py -mode E -type 25 -seed 123 -lr 0.005
    python main.py -mode E -type 30 -seed "$seed" -lr 0.005  # E25 with temporal kernel=51
    python main.py -mode E -type 31 -seed "$seed" -lr 0.005  # E25 with temporal kernel=63
    python main.py -mode E -type 29 -seed "$seed" -lr 0.005  # E25 with ReLU activation
    python main.py -mode E -type 27 -seed "$seed" -lr 0.005  # E25 + 4 Hz highpass FIR
    python main.py -mode E -type 28 -seed "$seed" -lr 0.005  # E25 + 30 Hz lowpass FIR
    python main.py -mode E -type 26 -seed "$seed" -lr 0.005  # case 20 architecture, cosine LR decay to 0.0001
done
```
