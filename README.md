# 2026 富邦 MA 簡報專題

## 🔹 Table of Contents

- [Problem](#-problem)
- [Installation](#-installation)
- [Usage](#-usage)

---

## 📈 Problem

1. 研究所有近月轉倉契約的歷史趨勢，分析身為期貨多單或空單持有者，何時轉倉較為有利，並以數據佐證。
2. 設計一個台指期日內交易策略，詳述交易邏輯，並完整呈現回測績效。

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/juju0621/2026-fubon-ma.git
cd 2026-fubon-ma
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🏃 Usage

### 0. Data preparation

將 **data.csv** 放入專案目錄下

### 1. Run Problem 1

```bash
chmod +x run_rollover.sh
./run_rollover.sh
```

### 2. Run Problem 2

```bash
chmod +x run_strategy.sh
./run_strategy.sh
```
