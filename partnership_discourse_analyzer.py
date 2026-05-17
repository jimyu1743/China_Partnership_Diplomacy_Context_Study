"""
夥伴關係外交話語分析系統
研究脈絡：中國夥伴關係外交的話語結構變遷
比較 P1（胡溫時期 2002-2012）與 P2（習近平時期 2013-2023）聯合聲明
樣本國家：俄羅斯、巴基斯坦、法國、越南、哈薩克、巴西
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
# 可選值: "openai"（GPT-4o）或 "claude"（Claude Sonnet 4.6，備案）
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
    MODEL_NAME = "claude-sonnet-4-6"

else:
    raise ValueError(f"不支援的 LLM_PROVIDER: '{LLM_PROVIDER}'。請選擇 'openai' 或 'claude'。")

# ==========================================
# 2. Prompt 模板（含 IR 理論框架與 Few-shot 範例）
# ==========================================
SYSTEM_PROMPT = """
你是一位精通中國外交與國際關係（IR）批判話語分析（CDA）的頂尖學術專家。
你的任務是分析中國《聯合聲明》文本，測量三個核心 IR 變數。

【分析守則：過濾制式語言】
你必須忽略以下「外交樣板語言（Boilerplate Language）」，這些詞彙在所有聲明中高頻出現，不具備鑑別效度：
- 一般性原則：「平等」「和平共處」「不干涉內政」「互利共贏」「睦鄰友好」
- 常見慣語：「雙方高度重視」「一致認為」「深化合作」「進一步加強」「戰略夥伴關係」
請「只針對為特定國家量身打造的客製化條款（Customized Clauses）」進行分析，
即那些反映特定雙邊關係動態、不可被套用於所有夥伴國的具體表述。

【變數一：宣示性（Assertiveness）評分（0.0 - 1.0）】
衡量文本中中國堅定表達核心利益的強度，但「未伴隨明確威脅」。
- 高分（0.7-1.0）：強力背書一帶一路、人類命運共同體、三大倡議；明確宣示台海/南海主權立場；夥伴方「主動使用」中國規範性術語。
- 低分（0.0-0.3）：文本以務實議題（貿易、投資、技術）為主；少有規範性話語植入。

【變數二：強制性（Coerciveness）評分（0.0 - 1.0）】
衡量文本中是否含有針對「第三方（如美國、西方聯盟）」的隱含威脅、戰略反制信號、或懲罰性後果。
- 高分（0.7-1.0）：「採取必要措施捍衛核心利益」；明確反對某一陣營或聯盟；強調不針對第三方（反向暗示）；出現武裝合作或安全保障條款。
- 低分（0.0-0.3）：無第三方對象；措辭僅限於雙邊正向合作。

【變數三：話語妥協（Discursive Compliance）判定】
判斷夥伴國對中國規範性話語的接受程度，分為：
- "High"：夥伴方「主動且具體地」背書中國術語（如逐字重複「人類命運共同體」並表示「堅決支持」）。
- "Low"：夥伴方僅「客套知悉」（如「注意到」「歡迎」而非主動擁護）或完全迴避使用中國術語。

【Few-shot 範例】
輸入文本：
"雙方高度評價一帶一路倡議，並堅決反對任何形式的霸權主義，將採取必要措施捍衛核心利益，雙方一致認為人類命運共同體是全球治理的正確方向..."

期望輸出：
{
  "Assertiveness": 0.9,
  "Coerciveness": 0.7,
  "Compliance": "High",
  "Customized_Clauses": "反霸權與採取必要措施捍衛核心利益；堅決支持人類命運共同體為治理正確方向",
  "Reasoning": "文本強力宣示一帶一路與人類命運共同體（高宣示性），且「採取必要措施捍衛核心利益」帶有隱含威脅信號，具有中等強制性。夥伴方逐字重複並主動背書中國規範性術語，屬高度話語妥協。"
}

【強制輸出格式】
你「必須」以 JSON 格式輸出，且「只能」包含以下 5 個 Key，不得新增其他欄位：
{
  "Assertiveness": <0.0 到 1.0 之間的浮點數>,
  "Coerciveness": <0.0 到 1.0 之間的浮點數>,
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
            clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
            result = json.loads(clean)

            # 驗證 5 個必要欄位
            required_keys = {"Assertiveness", "Coerciveness", "Compliance",
                             "Customized_Clauses", "Reasoning"}
            missing = required_keys - result.keys()
            if missing:
                print(f"   ⚠️  JSON 缺少欄位 {missing}，以 None 補齊。")
                for k in missing:
                    result[k] = None

            return result

        except Exception as e:
            err_str = str(e)
            # 偵測 Rate Limit 類型錯誤
            is_rate_limit = any(kw in err_str.lower() for kw in
                                ["rate limit", "ratelimit", "429", "quota"])
            if is_rate_limit and attempt < retries:
                print(f"   ⚠️  限速（Rate Limit），{wait}s 後重試"
                      f"（第 {attempt}/{retries} 次）...")
                time.sleep(wait)
                wait *= 2
            else:
                print(f"   ❌ 呼叫失敗（嘗試 {attempt}/{retries}）: {e}")
                if attempt == retries:
                    return None

    return None


# ==========================================
# 4. 主程式：批量處理資料管線
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("  中國夥伴關係外交 — 話語結構分析系統")
    print(f"  Provider: {LLM_PROVIDER.upper()}  |  模型: {MODEL_NAME}")
    print("=" * 60)

    # 讀取 Excel
    input_file = input(
        "\n👉 請輸入 Excel 檔名（預設: joint_declarations.xlsx）: "
    ).strip() or "joint_declarations.xlsx"

    if not os.path.exists(input_file):
        print(f"❌ 找不到檔案 '{input_file}'，請確認路徑。")
        exit()

    df = pd.read_excel(input_file)

    # 驗證必要欄位
    required_cols = {"Country", "Year", "Declaration_Text"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"❌ Excel 缺少必要欄位：{missing_cols}")
        print(f"   目前欄位：{list(df.columns)}")
        exit()

    print(f"\n✅ 載入 {len(df)} 筆文本，開始分析...\n")
    print(f"{'No.':<5} {'Country':<12} {'Year':<6} {'狀態'}")
    print("-" * 60)

    # 初始化結果容器
    results = {
        "Assertiveness": [],
        "Coerciveness": [],
        "Compliance": [],
        "Customized_Clauses": [],
        "Reasoning": []
    }

    for idx, row in df.iterrows():
        country = str(row["Country"])
        year = str(row["Year"])
        text = row["Declaration_Text"]
        display_no = f"[{idx+1}/{len(df)}]"

        # 空文本跳過
        if pd.isna(text) or str(text).strip() in ("", "nan"):
            print(f"{display_no:<5} {country:<12} {year:<6} ⚠️  文本為空，跳過")
            for key in results:
                results[key].append(None)
            continue

        print(f"{display_no:<5} {country:<12} {year:<6} 分析中...", end=" ", flush=True)
        result = call_llm(str(text), country, year)

        if result:
            results["Assertiveness"].append(result.get("Assertiveness"))
            results["Coerciveness"].append(result.get("Coerciveness"))
            results["Compliance"].append(result.get("Compliance"))
            results["Customized_Clauses"].append(result.get("Customized_Clauses"))
            results["Reasoning"].append(result.get("Reasoning"))
            a = result.get("Assertiveness", "?")
            c = result.get("Coerciveness", "?")
            comp = result.get("Compliance", "?")
            print(f"✅  Assert={a}  Coerce={c}  Compliance={comp}")
        else:
            for key in results:
                results[key].append("ERROR")
            print("❌  分析失敗，標記為 ERROR")

        # 避免觸發 Rate Limit
        time.sleep(1.5)

    # 寫回 DataFrame
    for key, values in results.items():
        df[key] = values

    # 輸出結果
    output_file = "analyzed_joint_declarations.xlsx"
    df.to_excel(output_file, index=False)

    # 統計摘要
    success_count = sum(1 for v in results["Assertiveness"] if v not in (None, "ERROR"))
    error_count = sum(1 for v in results["Assertiveness"] if v == "ERROR")
    skip_count = sum(1 for v in results["Assertiveness"] if v is None)

    print("\n" + "=" * 60)
    print(f"🎉 分析完成！輸出檔案：{output_file}")
    print(f"   ✅ 成功：{success_count} 筆  ❌ 失敗：{error_count} 筆  ⚠️ 跳過：{skip_count} 筆")
    if error_count > 0:
        print("   → 請篩選 Assertiveness == 'ERROR' 的列進行人工複核。")
    print("=" * 60)
