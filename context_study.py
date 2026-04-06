import os
import json
import time
import pandas as pd
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

# ==========================================
# 1. 動態讀取 JSON 並組裝 System Prompt
# ==========================================
def load_dynamic_prompt(json_path="codebook.json"):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到編碼簿檔案：{json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    prompt  = f"{config['system_role']}\n\n{config['anti_hallucination']}\n\n"
    for var in config["variables"]:
        prompt += f"【變數：{var['id']} ({var['name']})】\n"
        prompt += f"- 概念定義：{var['definition']}\n"
        prompt += f"- 1分條件：{var['score_1_condition']}\n"
        prompt += f"- 0分條件：{var['score_0_condition']}\n\n"
    prompt += f"【強制輸出格式：JSON 與雙重思維鏈 (CoT & Self-Evaluation)】\n{config['output_format']}"
    return prompt

# ==========================================
# 2. 系統初始化
# ==========================================
load_dotenv("context_study.env")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise ValueError("找不到 OPENAI_API_KEY！請確認 context_study.env 已正確設定。")

client = OpenAI(api_key=OPENAI_KEY)
SYSTEM_PROMPT = load_dynamic_prompt("codebook.json")

MODEL = "gpt-4.1-mini"
MAX_OUTPUT_TOKENS = 400

# ==========================================
# 3. API 呼叫模組
# ==========================================
def extract_variables_with_llm(text: str, retries: int = 3):
    wait = 5
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"請嚴格依據編碼簿分析以下《聯合聲明》文本：\n\n{text}"}
                ],
                temperature=0.0,
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)

        except RateLimitError:
            if attempt < retries:
                print(f"   ⚠️ 限速，{wait}s 後重試（第 {attempt}/{retries} 次）...")
                time.sleep(wait)
                wait *= 2
            else:
                print("   ❌ 達到重試上限，略過此筆。")
                return None
        except Exception as e:
            print(f"   ❌ API 處理失敗: {e}")
            return None

# ==========================================
# 4. 批量處理與資料管線
# ==========================================
if __name__ == "__main__":
    print("=" * 55)
    print("  LLM-Augmented TAD: 外交文本自動測量系統")
    print(f"  模型：{MODEL}  |  輸出上限：{MAX_OUTPUT_TOKENS} tokens/筆")
    print("=" * 55)

    input_file = input("\n👉 請輸入 Excel 檔名（預設 joint_declarations.xlsx）: ").strip()
    if not input_file:
        input_file = "joint_declarations.xlsx"

    if not os.path.exists(input_file):
        print(f"❌ 找不到檔案 {input_file}！")
        exit()

    df = pd.read_excel(input_file)
    if "Text" not in df.columns or "Case_ID" not in df.columns:
        print("❌ Excel 必須包含 'Case_ID' 與 'Text' 欄位！")
        exit()

    print(f"\n🚀 共 {len(df)} 筆文本，開始分析...\n")

    results_dict = {
        "FACE_Quote": [], "FACE_Reason": [], "FACE_SelfEval": [],
        "FACE_NeedsReview": [], "FACE_Score": [],
        "SANCT_Quote": [], "SANCT_Reason": [], "SANCT_SelfEval": [],
        "SANCT_NeedsReview": [], "SANCT_Score": [],
    }

    for index, row in df.iterrows():
        case_id = str(row["Case_ID"])
        raw_text = row["Text"]

        if pd.isna(raw_text) or str(raw_text).strip() in ("", "nan"):
            print(f"[{index+1}/{len(df)}] ⚠️  {case_id}：文本為空，跳過。")
            for key in results_dict:
                results_dict[key].append(None)
            continue

        text_content = str(raw_text)
        print(f"[{index+1}/{len(df)}] 分析中: {case_id} ...")
        result = extract_variables_with_llm(text_content)

        if result:
            results_dict["FACE_Quote"].append(result.get("FZ_FACE", {}).get("exact_quote"))
            results_dict["FACE_Reason"].append(result.get("FZ_FACE", {}).get("reasoning"))
            results_dict["FACE_SelfEval"].append(result.get("FZ_FACE", {}).get("self_evaluation"))
            results_dict["FACE_NeedsReview"].append(result.get("FZ_FACE", {}).get("needs_human_review"))
            results_dict["FACE_Score"].append(result.get("FZ_FACE", {}).get("score"))

            results_dict["SANCT_Quote"].append(result.get("FZ_SANCT", {}).get("exact_quote"))
            results_dict["SANCT_Reason"].append(result.get("FZ_SANCT", {}).get("reasoning"))
            results_dict["SANCT_SelfEval"].append(result.get("FZ_SANCT", {}).get("self_evaluation"))
            results_dict["SANCT_NeedsReview"].append(result.get("FZ_SANCT", {}).get("needs_human_review"))
            results_dict["SANCT_Score"].append(result.get("FZ_SANCT", {}).get("score"))

            print(f"   ✅ 完成（面子: {results_dict['FACE_Score'][-1]}, 反制裁: {results_dict['SANCT_Score'][-1]}）")
        else:
            for key in results_dict:
                results_dict[key].append("ERROR")

        time.sleep(0.5)

    for key, values in results_dict.items():
        df[key] = values

    output_file = "LLM_Scored_" + input_file
    df.to_excel(output_file, index=False)

    print("\n" + "=" * 55)
    print(f"🎉 完成！輸出：{output_file}")
    print("   請篩選 NeedsReview = TRUE 的個案進行人工覆核。")
    print("=" * 55)