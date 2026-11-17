# Kaggle AI Agent Security — 銀牌（第 163／4,186 名，前 3.89%）

[AI Agent Security - Multi-Step Tool Attacks](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks) 是 Kaggle 的紅隊競賽：在主辦方提供的**離線沙箱**中，自動搜尋會使用工具的 AI agent 的多步驟安全失敗（例如洩漏機密、執行不可信來源的指令、未授權動作）。本 repo 是我參賽的核心程式碼，最終名次第 163 名，獲得競賽銀牌（2026-09-02 頒發）。

所有攻擊只針對競賽 SDK 提供的沙箱靶標與測試資料，不針對任何真實系統。

## 方法

- 把攻擊分成數個「家族」，各自產生候選，再依沙箱內的評分排序（`src/aiagentsecurity/`）。
- 自適應提交：執行時探測評分環境，再分派到不同的攻擊分支，以應對公開榜與私人榜規則不同的風險。
- 本地以 llama.cpp 跑開源模型，模擬目標 agent 做大量篩選。

## 內容

- `attack.py`：攻擊邏輯與所有提交策略。以檔內常數 `SUBMISSION_VARIANT` 選擇策略，預設為 `cd_bulk_v2`。最終選入私人榜的兩個提交是 `cd_bulk_v3`，以及自適應分派版本（內部標記 `adaptive_v2`，加入了真實機密值外送分支）。程式中這個自適應版本沿用策略名稱 `adaptive_v1` 來選取。
- `notebooks/submission.ipynb`：最終自適應提交所用的 notebook，內嵌一份 `attack.py`，策略設為 `adaptive_v1`（即上述自適應版本）。改 `attack.py` 不會自動同步到 notebook。
- `src/aiagentsecurity/`：攻擊家族、候選排序、評估、guardrail 模擬與 llama.cpp 後端。
- `tests/`：測試。

## 執行

- 需要 Python 3.12 以上，環境以 [uv](https://github.com/astral-sh/uv) 管理：`uv sync`。
- 需另行安裝競賽提供的 `aicomp_sdk`（未包含在本 repo，也未列在 `pyproject.toml`）。
- 本地模擬需自備 llama.cpp 與 GGUF 模型檔。

只包含程式碼，不含競賽 SDK、資料、模型與實驗紀錄。
