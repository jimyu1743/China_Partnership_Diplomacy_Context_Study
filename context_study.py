import os
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# 1. 載入環境變數與 API 設定
# ==========================================
load_dotenv("context_study.env")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    raise ValueError("⚠️ 找不到 OPENAI_API_KEY！請確認 context_study.env 檔案已正確設定。")

client = OpenAI(api_key=OPENAI_KEY)

# ==========================================
# 2. 定義「外交語境知識庫 (Codebook)」 
# ==========================================
SYSTEM_PROMPT = """
<role>
你是一位頂尖的國際關係學者與中國外交政策專家，擅長批判性話語分析與編碼。
</role>

<task>
你的任務是閱讀被 <document> 標籤包裝的《聯合聲明》文本，並根據 <codebook> 中的四個變數定義進行客觀評分。
請嚴格遵循「先尋找證據 (quotes)，再進行推理 (reasoning)，最後才給予評分 (score)」的邏輯。
</task>

<codebook>
  <variable name="FZ_FACE" desc="面子與政治忠誠">
    <score_1>明確提及「台灣是中國不可分割的一部分」、「反對台獨」，或明確表態高度讚賞/支持「人類命運共同體」、「三大倡議(GDI/GSI/GCI)」。</score_1>
    <score_0>僅使用模糊的傳統外交辭令（如「尊重彼此主權和領土完整」），完全避談台灣具體地位，或對三大倡議隻字未提。</score_0>
  </variable>

  <variable name="FZ_SANCT" desc="制裁與外部威懾">
    <score_1>明確出現「反對單邊制裁」、「反對長臂管轄」、「反對干涉內政」、「反對霸權主義」等強烈防禦性與抗議性詞彙。</score_1>
    <score_0>沒有提及反對制裁或干涉內政，僅談論一般經貿、文化或科技合作。</score_0>
  </variable>

  <variable name="FZ_RED_LINE" desc="紅線與限制性辭令">
    <score_1>文本中強調「基於規則的國際秩序 (rules-based order)」、「去風險 (de-risking)」、「航行自由 (南海語境)」，或對人權表達單方面關切。</score_1>
    <score_0>無上述防範性辭令，語氣完全順應中方和諧共贏的話語框架。</score_0>
  </variable>

  <variable name="FZ_GLOBAL_GOV" desc="全球治理與公共財提供">
    <score_1>文本中明確將中方或雙邊合作與新興全球治理議題連結，包含「氣候變化」、「生物多樣性」、「綠色轉型」、「減貧」或「可持續發展」，展現中方在全球議程上的引領角色。</score_1>
    <score_0>未提及上述新興全球治理議題，僅侷限於傳統的雙邊經貿、基礎建設或文化交流。</score_0>
  </variable>
</codebook>

<few_shot_examples>
  <example>
    <input_snippet>"雙方重申尊重彼此主權。將致力於加強雙邊貿易，並確保全球供應鏈的韌性與去風險化。雙方同意在聯合國框架下共同應對氣候變化與保護生物多樣性。"</input_snippet>
    <expected_output>
      "FZ_FACE": {"quotes": ["重申尊重彼此主權"], "reasoning": "僅使用模糊主權辭令，未明確提及台灣或三大倡議", "score": 0},
      "FZ_RED_LINE": {"quotes": ["確保全球供應鏈的韌性與去風險化"], "reasoning": "出現了明顯的防範性西方秩序辭令（去風險化）", "score": 1},
      "FZ_GLOBAL_GOV": {"quotes": ["共同應對氣候變化與保護生物多樣性"], "reasoning": "明確將雙邊合作延伸至氣候變化與生物多樣性等新興全球治理議題", "score": 1}
    </expected_output>
  </example>
</few_shot_examples>

<output_format>
你必須以純 JSON 格式輸出，絕對不能包含任何其他文字或 Markdown 標記。JSON 結構必須嚴格如下：
{
  "FZ_FACE": {
    "quotes": ["引言1", "引言2"],  // 🌟 重要：若無引言，請務必填寫空列表 []，絕對不要填 null
    "reasoning": "...",
    "score": 1或0
  },
  "FZ_SANCT": {
    "quotes": ["..."],
    "reasoning": "...",
    "score": 1或0
  },
  "FZ_RED_LINE": {
    "quotes": ["..."],
    "reasoning": "...",
    "score": 1或0
  },
  "FZ_GLOBAL_GOV": {
    "quotes": ["..."],
    "reasoning": "...",
    "score": 1或0
  }
}
</output_format>
"""


# ==========================================
# 3. 雙層萃取引擎 
# ==========================================
def call_responses_api(text, model_name, effort_level):
    """使用 2026 最新 Responses API - 修正內容抓取與空值問題"""
    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"請分析以下文本：\n<document>\n{text}\n</document>"}
            ],
            reasoning={"effort": effort_level}, 
            # 確保輸出格式為 JSON，這在 GPT-5.4 中會直接影響 output_text 的純淨度
            text={"format": {"type": "json_object"}}
        )
        
        # 🌟 核心修正 1：在 2026 API 中，純文字內容位於 .output_text
        raw_output = getattr(response, 'output_text', "")
        
        # 如果 output_text 為空，嘗試備用方案
        if not raw_output and hasattr(response, 'text') and hasattr(response.text, 'value'):
            raw_output = response.text.value

        if not raw_output:
            print(f"   ⚠️ 模型回傳內容為空 (Status: {getattr(response, 'status', 'unknown')})")
            return None

        # 🌟 核心修正 2：處理可能的 Markdown 標籤包裹 (雖然使用了 json_object，但防禦性編程更穩健)
        clean_json = raw_output.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
        elif clean_json.startswith("```"):
            clean_json = clean_json.split("```")[1].split("```")[0].strip()

        return json.loads(clean_json)
        
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON 解析失敗: {e} | 原始內容: {raw_output[:100]}...")
        return None
    except Exception as e:
        print(f"   ❌ API 處理失敗 ({model_name}): {e}")
        return None

def extract_with_two_tier_logic(text):
    """實踐兩層架構：先用 5.4-mini 初篩，有疑慮再升級 5.4 旗艦模型"""
    
    # 【第一層：高性價比初篩】
    print("   ➔ 啟動第一層 (gpt-5.4-mini, medium) 進行廣泛掃描...")
    result = call_responses_api(text, model_name="gpt-5.4-mini", effort_level="medium")
    
    if not result:
        return None, "API_Failure"

    # 【觸發機制：檢查是否需要進入第二層複核】
    # 🌟 修正：移除「未明確」、「模糊」等陳述性詞彙，改為嚴格的「困惑指標詞」(涵蓋繁簡體)
    ambiguous_keywords = [
        "无法确定", "無法確定", 
        "难以判断", "難以判斷", 
        "资料不足", "資料不足", 
        "存在矛盾", 
        "无法归类", "無法歸類", 
        "不知如何", "不完全符合"
    ]
    needs_review = False
    
    # 🌟 修正：將新挖掘出的變數 FZ_GLOBAL_GOV 加入檢查名單
    for key in ["FZ_FACE", "FZ_SANCT", "FZ_RED_LINE", "FZ_GLOBAL_GOV"]:
        score = result.get(key, {}).get("score")
        reason_text = result.get(key, {}).get("reasoning", "") # 抓取 reasoning
        
        if score not in [0, 1] or any(word in reason_text for word in ambiguous_keywords):
            needs_review = True
            break
            
    # 【第二層：高疑難文本複核】
    if needs_review:
        print("   ⚠️ 偵測到語義模糊或保留字眼，升級第二層 (gpt-5.4, high) 深度推演...")
        deep_result = call_responses_api(text, model_name="gpt-5.4", effort_level="high")
        if deep_result:
            return deep_result, "Tier-2 (gpt-5.4 High)"
            
    return result, "Tier-1 (gpt-5.4-mini Medium)"


# ==========================================
# 4. 批量化主程式
# ==========================================
if __name__ == "__main__":
    print("="*65)
    print("🤖 啟動：中國外交聲明 雙層自動化編碼引擎 (GPT-5.4 世代 V2)")
    print("="*65)
    
    input_file = input("👉 請輸入包含聲明文本的 Excel 檔名 (預設為 joint_declarations.xlsx): ").strip()
    if not input_file:
        input_file = "joint_declarations.xlsx"
        
    if not os.path.exists(input_file):
        print(f"❌ 找不到檔案 {input_file}！")
        exit()

    df = pd.read_excel(input_file)
    text_col = 'Text' if 'Text' in df.columns else None
    if not text_col:
        print("❌ 錯誤：在 Excel 中找不到 'Text' 欄位！")
        exit()

    print(f"🚀 發現 {len(df)} 筆文本，準備開始雙層 AI 編碼...\n")

    # 準備用來存儲結果的 List
    face_scores, face_quotes, face_reasonings = [], [], []
    sanct_scores, sanct_quotes, sanct_reasonings = [], [], []
    redline_scores, redline_quotes, redline_reasonings = [], [], []
    engines_used = []

    for index, row in df.iterrows():
        case_name = row.get('Country', f"Case_{index}") 
        period_info = row.get('Period', '未知時期')
        text_content = str(row[text_col])
        
        print(f"\n[{index+1}/{len(df)}] 正在分析: {case_name} ({period_info})")
        
        if pd.isna(text_content) or text_content.strip() == "":
            print("   ⚠️ 文本為空，跳過。")
            for lst in [face_scores, sanct_scores, redline_scores]: lst.append(None)
            for lst in [face_quotes, sanct_quotes, redline_quotes]: lst.append(None)
            for lst in [face_reasonings, sanct_reasonings, redline_reasonings]: lst.append("無文本")
            engines_used.append("N/A")
            continue
            
        # 呼叫雙層邏輯引擎
        result, tier_used = extract_with_two_tier_logic(text_content)
        
        if result:
            # FZ_FACE
            face_scores.append(result.get("FZ_FACE", {}).get("score"))
            face_quotes.append(" | ".join(result.get("FZ_FACE", {}).get("quotes", [])))
            face_reasonings.append(result.get("FZ_FACE", {}).get("reasoning"))
            
            # FZ_SANCT
            sanct_scores.append(result.get("FZ_SANCT", {}).get("score"))
            sanct_quotes.append(" | ".join(result.get("FZ_SANCT", {}).get("quotes", [])))
            sanct_reasonings.append(result.get("FZ_SANCT", {}).get("reasoning"))
            
            # FZ_RED_LINE
            redline_scores.append(result.get("FZ_RED_LINE", {}).get("score"))
            redline_quotes.append(" | ".join(result.get("FZ_RED_LINE", {}).get("quotes", [])))
            redline_reasonings.append(result.get("FZ_RED_LINE", {}).get("reasoning"))
            
            engines_used.append(tier_used)
            
            print(f"   ✅ 完成！(使用引擎: {tier_used} | 面子: {face_scores[-1]} | 反制裁: {sanct_scores[-1]} | 紅線: {redline_scores[-1]})")
        else:
            for lst in [face_scores, sanct_scores, redline_scores]: lst.append(None)
            for lst in [face_quotes, sanct_quotes, redline_quotes]: lst.append(None)
            for lst in [face_reasonings, sanct_reasonings, redline_reasonings]: lst.append("API 失敗")
            engines_used.append("Failed")
            
        time.sleep(1.0) # 避免觸發 API 頻率限制

    # 將結果合併回原本的 DataFrame
    df['FZ_FACE_Score'] = face_scores
    df['FZ_FACE_Quotes'] = face_quotes
    df['FZ_FACE_Reasoning'] = face_reasonings
    
    df['FZ_SANCT_Score'] = sanct_scores
    df['FZ_SANCT_Quotes'] = sanct_quotes
    df['FZ_SANCT_Reasoning'] = sanct_reasonings
    
    df['FZ_RED_LINE_Score'] = redline_scores
    df['FZ_RED_LINE_Quotes'] = redline_quotes
    df['FZ_RED_LINE_Reasoning'] = redline_reasonings
    
    df['Engine_Used'] = engines_used

    # 存成新的 Excel 檔案
    output_file = "Scored_SMMR_Final_" + input_file
    df.to_excel(output_file, index=False)
    
    print("\n" + "="*65)
    print(f"🎉 雙層批量編碼大功告成！")
    print(f"📊 結果已自動儲存為全新檔案：{output_file}")
    print("="*65)