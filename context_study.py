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
你現在是一位頂尖的國際關係學者與中國外交政策專家。
你的任務是閱讀中國與其他國家發布的《聯合聲明》文本，並根據以下兩個變數的「學術編碼簿（Codebook）」進行評分。

【變數 1：FZ_FACE (面子與政治忠誠)】
定義：夥伴國是否在中國的「核心利益」上給予強烈且明確的政治背書。
評分標準：
- 1分：文本中明確提及「台灣是中國不可分割的一部分」、「反對任何形式的台獨」，或在「新疆、香港、人權」等敏感議題上，明確表態支持中國政府的立場。
- 0分：僅使用模糊的外交辭令（如「尊重彼此主權」），或完全沒有提及上述中國核心利益。

【變數 2：FZ_SANCT (制裁與外部威懾)】
定義：雙方是否面臨來自西方（如美國）的制裁壓力，或試圖透過聲明反擊外部壓力。
評分標準：
- 1分：文本中明確出現「反對單邊制裁」、「反對長臂管轄」、「反對干涉內政」、「反對霸權主義與強權政治」等具有強烈針對性的防禦或反擊詞彙。
- 0分：沒有提及反對制裁或反對干涉內政，僅談論一般的經貿或文化合作。

【輸出格式要求】
你必須以純 JSON 格式輸出，格式如下：
{
  "FZ_FACE": {
    "score": 1或0,
    "reason": "請引用文本中的關鍵句，並簡短解釋為何給此分數（50字以內）"
  },
  "FZ_SANCT": {
    "score": 1或0,
    "reason": "請引用文本中的關鍵句，並簡短解釋為何給此分數（50字以內）"
  }
}
"""

def extract_diplomatic_variables(text):
    """呼叫 OpenAI API 進行文本編碼"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"請分析以下《聯合聲明》文本：\n\n{text}"}
            ],
            temperature=0.1, # 極低溫度確保學術信度
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"   ❌ API 處理失敗: {e}")
        return None

# ==========================================
# 3. 批量化主程式
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("🤖 啟動：中國外交聲明 LLM 批量自動編碼引擎")
    print("="*50)
    
    # 讓使用者輸入檔案名稱
    input_file = input("👉 請輸入包含聲明文本的 Excel 檔名 (預設為 joint_declarations.xlsx): ").strip()
    if not input_file:
        input_file = "joint_declarations.xlsx"
        
    if not os.path.exists(input_file):
        print(f"❌ 找不到檔案 {input_file}！請確認檔案是否存在。")
        exit()

    print(f"\n📂 正在讀取資料集：{input_file}")
    df = pd.read_excel(input_file)
    
    # 確認是否有 Text 欄位
    text_col = 'Text' if 'Text' in df.columns else None
    if not text_col:
        print("❌ 錯誤：在 Excel 中找不到名為 'Text' 的欄位！請確認您的文本是放在 'Text' 欄位中。")
        exit()

    print(f"🚀 發現 {len(df)} 筆文本，準備開始批量 AI 編碼...\n")

    # 準備用來存儲結果的 List
    face_scores, face_reasons = [], []
    sanct_scores, sanct_reasons = [], []

    # 逐行進行編碼
    for index, row in df.iterrows():
        case_name = row.get('Country', f"Case_{index}") # 如果有 Country 欄位就印出來，否則印編號
        text_content = str(row[text_col])
        
        print(f"[{index+1}/{len(df)}] 正在分析: {case_name} ...")
        
        if pd.isna(text_content) or text_content.strip() == "":
            print("   ⚠️ 文本為空，跳過此筆。")
            face_scores.append(None); face_reasons.append("無文本")
            sanct_scores.append(None); sanct_reasons.append("無文本")
            continue
            
        # 呼叫 LLM
        result = extract_diplomatic_variables(text_content)
        
        if result:
            # 成功解析，寫入結果
            face_scores.append(result.get("FZ_FACE", {}).get("score"))
            face_reasons.append(result.get("FZ_FACE", {}).get("reason"))
            sanct_scores.append(result.get("FZ_SANCT", {}).get("score"))
            sanct_reasons.append(result.get("FZ_SANCT", {}).get("reason"))
            print(f"   ✅ 完成！(面子: {face_scores[-1]}, 反制裁: {sanct_scores[-1]})")
        else:
            # 解析失敗
            face_scores.append(None); face_reasons.append("API 失敗")
            sanct_scores.append(None); sanct_reasons.append("API 失敗")
            
        # 避免 API 頻率限制 (Rate Limit)
        time.sleep(1.5) 

    # 將結果合併回原本的 DataFrame
    df['FZ_FACE_Score'] = face_scores
    df['FZ_FACE_Reason'] = face_reasons
    df['FZ_SANCT_Score'] = sanct_scores
    df['FZ_SANCT_Reason'] = sanct_reasons

    # 存成新的 Excel 檔案
    output_file = "Scored_" + input_file
    df.to_excel(output_file, index=False)
    
    print("\n" + "="*50)
    print(f"🎉 批量編碼大功告成！")
    print(f"📊 結果已自動儲存為全新檔案：{output_file}")
    print("="*50)