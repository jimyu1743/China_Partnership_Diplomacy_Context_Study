"""
夥伴關係外交話語分析系統 (學術防禦升級版)
研究脈絡：中國夥伴關係外交的話語結構變遷
比較 P1（胡溫時期 2002-2012）與 P2（習近平時期 2013-2023）聯合聲明
特色：消除「假性精確(浮點數)」與「跨期測量偏誤(Temporal Bias)」
"""

import os
import json
import time
import re
import pandas as pd
from dotenv import load_dotenv

# ==========================================
# 0. 選擇 LLM Provider（改此變數切換）
# ==========================================
# 可選值: "openai"（GPT-4o）或 "claude"（Claude Sonnet 3.5/4.6，備案）
LLM_PROVIDER = "openai"

# ==========================================
# 1. 環境變數與 API 用戶端初始化
# ==========================================
load_dotenv("context_study.env")

if LLM_PROVIDER == "openai":
    from openai import OpenAI, RateLimitError as OpenAIRateLimitError
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_KEY:
        raise ValueError("找不到 OPENAI_API_KEY！請確認 context_study.env。")
    client = OpenAI(api_key=OPENAI_KEY)
    MODEL_NAME = "gpt-4o"

elif LLM_PROVIDER == "claude":
    from anthropic import Anthropic
    ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
    if not ANTHROPIC_KEY:
        raise ValueError("找不到 ANTHROPIC_API_KEY！請確認 context_study.env。")
    client = Anthropic(api_key=ANTHROPIC_KEY)
    MODEL_NAME = "claude-3-5-sonnet-20241022" # 確保使用支援的 Claude 模型版本

else:
    raise ValueError(f"不支援的 LLM_PROVIDER: '{LLM_PROVIDER}'。請選擇 'openai' 或 'claude'。")

# ==========================================
# 2. Prompt 模板（學術防禦升級版：整數尺度與時間中立）
# ==========================================
SYSTEM_PROMPT = """
你是一位精通中國外交與國際關係（IR）批判話語分析（CDA）的頂尖學術專家。
你的任務是比較中國在 P1（2002-2012）與 P2（2013-2023）兩個時期的《聯合聲明》文本，測量三個核心 IR 變數。

【分析守則：過濾制式語言】
你必须忽略以下「外交樣板語言（Boilerplate Language）」：
- 一般性原則：「平等」「和平共處」「不干涉內政」「互利共贏」「睦鄰友好」
請「只針對為特定國家量身打造的客製化條款（Customized Clauses）」或「具有時代特徵的政治論述」進行分析。

【變數一：宣示性（Assertiveness）評分（整數 0, 1, 2）】
衡量文本中中國堅定表達核心利益與政治理念的強度。
- 2分（高度）：強力背書中國特定時期的全球戰略口號。
  * 若為 P2 時期：出現「一帶一路」、「人類命運共同體」、「三大倡議」等。
  * 若為 P1 時期：出現「和諧世界」、「和平發展道路」、「推動國際關係民主化」等。
  * 或明確宣示台海/涉疆/南海等核心主權絕對立場。
- 1分（中度）：提及雙邊或多邊的原則性合作，但未主動背書上述具體的中國專屬政治口號。
- 0分（低度）：文本完全以務實議題（貿易、投資、技術）為主，無政治規範性話語植入。

【變數二：強制性（Coerciveness）評分（整數 0, 1, 2）】
衡量文本中是否含有針對「第三方（如美國、西方聯盟）」的戰略反制信號或隱含威脅。
- 2分（高度）：明確出現針對性的外交黑話，如：「反對動輒使用單邊制裁」、「反對將國內法凌駕於國際法（暗指長臂管轄）」、「反對冷戰思維」、「強調不針對第三方（反向暗示）」、或出現聯合軍演/安全保障條款。
- 1分（中度）：隱晦表達對當前國際秩序的不滿，如：「主張推動國際體系變革」，但無具體反制裁或反霸權字眼。
- 0分（低度）：無第三方對象，無任何反制色彩，措辭僅限於雙邊經貿正向合作。

【變數三：話語妥協（Discursive Compliance）判定】
判斷夥伴國對中國規範性話語的接受程度，分為：
- "High"：夥伴方「主動且具體地」背書中國術語（如逐字重複中國口號並表示「堅決支持」、「高度讚賞」）。
- "Low"：夥伴方僅「客套知悉」（如「注意到」、「歡迎」）或完全迴避使用中國政治術語。

【Few-shot 範例】
輸入文本：
"雙方高度評價中方提出的構建人類命運共同體倡議。雙方堅決反對任何形式的霸權主義，反對單邊制裁與長臂管轄。雙方將進一步擴大農業貿易..."

期望輸出：
{
  "Assertiveness": 2,
  "Coerciveness": 2,
  "Compliance": "High",
  "Customized_Clauses": "反對單邊制裁與長臂管轄；高度評價人類命運共同體",
  "Reasoning": "文本強力宣示人類命運共同體（Assertiveness=2），且使用『反對單邊制裁與長臂管轄』作為對西方的強烈戰略反制信號（Coerciveness=2）。夥伴方主動高度評價中方倡議，屬高度話語妥協（High）。"
}

【強制輸出格式】
你「必須」以 JSON 格式輸出，且「只能」包含以下 5 個 Key：
{
  "Assertiveness": <0, 1, 2 其中的一個整數>,
  "Coerciveness": <0, 1, 2 其中的一個整數>,
  "Compliance": <"High" 或 "Low">,
  "Customized_Clauses": <字串，摘述你認定的客製化條款，若無則填 "無明顯客製化條款">,
  "Reasoning": <字串，簡要說明你的評分邏輯，50-150 字>
}
"""

# ==========================================
# 3. LLM API 呼叫模組（含指數退避重試）
# ==========================================
def call_llm(text: str, country: str, year, retries: int = 3) -> dict | None:
    user_message = (
        f"請分析以下【{country}】{year}年《聯合聲明》文本，"
        f"嚴格忽略制式外交語言，只針對客製化條款評分：\n\n{text}"
    )

    wait = 5
    for attempt in range(1, retries + 1):
        try:
            if LLM_PROVIDER == "openai":
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                raw = response.choices[0].message.content

            elif LLM_PROVIDER == "claude":
                response = client.messages.create(
                    model=MODEL_NAME,
                    max_tokens=1024,
                    temperature=0.0,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}]
                )
                raw = response.content[0].text

            # 嘗試解析 JSON（防呆：去除 markdown 程式碼區塊包裝）
            clean = re.sub(r"^
http://googleusercontent.com/immersive_entry_chip/0

### 💡 這次腳本的 3 個關鍵守護點：

1. **強制輸出 `0, 1, 2`**：您會發現現在 `Assertiveness` 和 `Coerciveness` 輸出的都是乾淨的整數。這樣在後續匯入 fsQCA 軟體時，您就能用理論為基礎，將 `2` 校準為 `1.0`（完全隸屬），將 `1` 校準為 `0.5`（交叉點），將 `0` 校準為 `0.0`（完全不隸屬）。**校準權回到了研究者手上。**
2. **消滅時間偏見 (Temporal Bias)**：AI 現在知道 P1 時代的「和諧世界」和 P2 時代的「人類命運共同體」是等價的高強度宣示，因此 2002-2012 年間的胡溫時代聲明，終於能獲得公平的 `2分` 評價，這讓您跨期的 $\Delta$ 變化量真正具備因果推論價值。
3. **優化了 DataFrame 欄位防呆**：我加了一小段代碼，無論您的 Excel 欄位叫作 `Text` 還是 `Declaration_Text`，程式都能自動辨識並執行，避免報錯。

您現在可以直接在終端機中輸入 `python 您命名的腳本名稱.py`，用您手邊的《聯合聲明》Excel 測試這套具備最高學術防禦力的測量系統了！