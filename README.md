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

目前 dataloader 使用相對於工作目錄的 `./lab2_EEG_classification/data/`。因此，需從 repository 的上一層執行：

```bash
python3 lab2_EEG_classification/main.py -num_epochs 150 -batch_size 64 -lr 0.01
```

以上為現有程式的命令格式；模型、輸出目錄與程式問題修正前，尚不能完成訓練。

## 日常 Git 操作

在此 repository 目錄執行：

```bash
git status
git diff
git add main.py dataloader.py models/EEGNet.py README.md
git commit -m "Describe the change"
```

設定 GitHub remote 後，可用 `git push -u origin main` 首次上傳，後續使用 `git push`。遠端主機首次使用 `git clone` 取得程式，後續用 `git pull --ff-only` 更新。

實驗時記錄 `git rev-parse HEAD` 的 commit ID 及完整超參數，方便追溯程式版本。兩台主機切換前，先提交並推送需要保留的修改。
