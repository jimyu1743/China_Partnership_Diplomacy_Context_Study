import os
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# 1. 動態讀取 JSON 並組裝 System Prompt 的函數
# ==========================================
def load_dynamic_prompt(json_path="codebook.json"):
    """從外部 JSON 檔案讀取設定，並動態組裝成 LLM 需要的 Prompt 字串"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"⚠️ 找不到編碼簿檔案：{json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    # 1. 載入角色設定與防幻覺規則
    prompt = f"{config['system_role']}\n\n{config['anti_hallucination']}\n\n"
    
    # 2. 迴圈動態讀取所有變數 
    for var in config['variables']:
        prompt += f"【變數：{var['id']} ({var['name']})】\n"
        prompt += f"- 概念定義：{var['definition']}\n"
        prompt += f"- 1分條件：{var['score_1_condition']}\n"
        prompt += f"- 0分條件：{var['score_0_condition']}\n\n"
        
    # 3. 載入強制輸出格式
    prompt += f"【強制輸出格式：JSON 與雙重思維鏈 (CoT & Self-Evaluation)】\n{config['output_format']}"
    
    return prompt

# ==========================================
# 2. 系統初始化與 API 呼叫模組
# ==========================================
# 載入環境變數
load_dotenv("context_study.env")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_KEY:
    raise ValueError("⚠️ 找不到 OPENAI_API_KEY！請確認 context_study.env 檔案已正確設定。")

client = OpenAI(api_key=OPENAI_KEY)

# 啟動時即時編譯 Prompt
SYSTEM_PROMPT = load_dynamic_prompt("codebook.json")

def extract_variables_with_llm(text):
    """呼叫 LLM 進行結構化編碼 (低溫度、強約束)"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"請嚴格依據編碼簿分析以下《聯合聲明》文本：\n\n{text}"}
            ],
            temperature=0.0, # 【關鍵】溫度 0，完全消除隨機性與幻覺
            response_format={ "type": "json_object" } # 強制 JSON 輸出
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"   ❌ API 處理失敗: {e}")
        return None

# ==========================================
# 3. 批量處理與資料管線 (Data Pipeline)
# ==========================================
if __name__ == "__main__":
    print("="*55)
    print("🇨🇳 計算社會科學：外交文本 LLM 自動測量系統 (SMMR版)")
    print("="*55)
    
    # 讓使用者輸入檔案名稱
    input_file = input("👉 請輸入包含聲明文本的 Excel 檔名 (預設為 joint_declarations.xlsx): ").strip()
    if not input_file:
        input_file = "joint_declarations.xlsx"
        
    if not os.path.exists(input_file):
        print(f"❌ 找不到檔案 {input_file}！請確認您的資料夾中有此 Excel 檔案。")
        print("💡 提示：Excel 中需包含 'Case_ID' (案件名稱) 與 'Text' (聲明全文) 兩個欄位。")
        exit()

    print(f"\n📂 正在讀取資料集：{input_file}")
    df = pd.read_excel(input_file)
    
    # 檢查必要欄位
    if 'Text' not in df.columns or 'Case_ID' not in df.columns:
        print("❌ 錯誤：Excel 中必須包含 'Case_ID' 與 'Text' 欄位！")
        exit()

    print(f"🚀 發現 {len(df)} 筆外交文本，開始進行機制追蹤 (Process Tracing)...\n")

    # 準備資料欄位 (包含報價、推論、自我評估與最終分數)
    results_dict = {
        'FACE_Quote': [], 'FACE_Reason': [], 'FACE_SelfEval': [], 'FACE_Score': [],
        'SANCT_Quote': [], 'SANCT_Reason': [], 'SANCT_SelfEval': [], 'SANCT_Score': []
    }

    # 逐案執行 LLM 分析
    for index, row in df.iterrows():
        case_id = str(row['Case_ID'])
        text_content = str(row['Text'])
        
        print(f"[{index+1}/{len(df)}] 正在分析個案: {case_id} ...")
        
        if pd.isna(text_content) or text_content.strip() == "" or text_content == "nan":
            print("   ⚠️ 文本為空，跳過此筆。")
            for key in results_dict:
                results_dict[key].append(None)
            continue
            
        # 啟動 AI 萃取
        result = extract_variables_with_llm(text_content)
        
        if result:
            # FZ_FACE 寫入
            results_dict['FACE_Quote'].append(result.get("FZ_FACE", {}).get("exact_quote"))
            results_dict['FACE_Reason'].append(result.get("FZ_FACE", {}).get("reasoning"))
            results_dict['FACE_SelfEval'].append(result.get("FZ_FACE", {}).get("self_evaluation"))
            results_dict['FACE_Score'].append(result.get("FZ_FACE", {}).get("score"))
            
            # FZ_SANCT 寫入
            results_dict['SANCT_Quote'].append(result.get("FZ_SANCT", {}).get("exact_quote"))
            results_dict['SANCT_Reason'].append(result.get("FZ_SANCT", {}).get("reasoning"))
            results_dict['SANCT_SelfEval'].append(result.get("FZ_SANCT", {}).get("self_evaluation"))
            results_dict['SANCT_Score'].append(result.get("FZ_SANCT", {}).get("score"))
            
            print(f"   ✅ 完成！(政治忠誠: {results_dict['FACE_Score'][-1]}, 反制裁: {results_dict['SANCT_Score'][-1]})")
        else:
            for key in results_dict:
                results_dict[key].append("ERROR")
                
        # 遵守 API 速率限制 (避免被 OpenAI 阻擋)
        time.sleep(1.5) 

    # 將結構化數據併回原本的 DataFrame
    for key, values in results_dict.items():
        df[key] = values

    # 匯出分析結果
    output_file = "LLM_Scored_" + input_file
    df.to_excel(output_file, index=False)
    
    print("\n" + "="*55)
    print(f"🎉 批量文本測量大功告成！")
    print("⚠️ 學術提醒：請落實「人類迴圈(HITL)」，抽樣檢查輸出的 Excel 檔案。")
    print(f"📊 分析結果已儲存為：{output_file}")
    print("="*55)