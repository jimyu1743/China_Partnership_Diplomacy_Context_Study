import os
import json
import time
import re
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# 1. 讀取 XML 提示詞模板
# ==========================================
def load_xml_template(file_path="prompt_template.xml"):
    """讀取 XML 標籤模板"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"⚠️ 找不到 XML 模板檔案：{file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ==========================================
# 2. API 初始化
# ==========================================
load_dotenv("context_study.env")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise ValueError("⚠️ 找不到 OPENAI_API_KEY！請確認 context_study.env 檔案。")

client = OpenAI(api_key=OPENAI_KEY)
XML_TEMPLATE = load_xml_template()

# ==========================================
# 3. LLM 呼叫模組（含指數退避重試）
# ==========================================
def extract_themes_with_xml(text: str, template: str, retries: int = 3) -> dict | None:
    """將文本注入 XML 模板並呼叫 LLM 進行主題萃取"""
    final_prompt = template.replace("{{TARGET_TEXT}}", text)

    wait = 5
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            raw_output = response.choices[0].message.content
            # 防呆：去除 markdown 程式碼區塊包裝
            clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip())
            result = json.loads(clean_json)

            # 驗證 4 個必要欄位
            required_keys = {"Primary_Focus", "Signature_Jargons",
                             "Has_Core_Interest_Support", "Key_Quote"}
            missing = required_keys - result.keys()
            if missing:
                print(f"   ⚠️  JSON 缺少欄位 {missing}，以 None 補齊。")
                for k in missing:
                    result[k] = None

            return result

        except Exception as e:
            err_str = str(e)
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
    print("  中國外交話語 — 主題分類與口號萃取系統（XML 模板版）")
    print("=" * 60)

    input_file = input(
        "\n👉 請輸入 Excel 檔名（預設: 文獻/joint_declarations.xlsx）: "
    ).strip() or "文獻/joint_declarations.xlsx"

    if not os.path.exists(input_file):
        print(f"❌ 找不到檔案 '{input_file}'，請確認路徑。")
        exit()

    df = pd.read_excel(input_file)

    # 驗證必要欄位
    required_cols = {"Country", "Text"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"❌ Excel 缺少必要欄位：{missing_cols}")
        print(f"   目前欄位：{list(df.columns)}")
        exit()

    print(f"\n✅ 載入 {len(df)} 筆文本，開始主題分析...\n")
    print(f"{'No.':<5} {'Country':<12} {'狀態'}")
    print("-" * 60)

    results = {
        "Primary_Focus": [],
        "Signature_Jargons": [],
        "Has_Core_Interest_Support": [],
        "Key_Quote": []
    }

    for idx, row in df.iterrows():
        country = str(row.get("Country", f"Row_{idx}"))
        text = row["Text"]
        display_no = f"[{idx+1}/{len(df)}]"

        # 空文本跳過
        if pd.isna(text) or str(text).strip() in ("", "nan"):
            print(f"{display_no:<5} {country:<12} ⚠️  文本為空，跳過")
            for key in results:
                results[key].append(None)
            continue

        print(f"{display_no:<5} {country:<12} 分析中...", end=" ", flush=True)
        result = extract_themes_with_xml(str(text), XML_TEMPLATE)

        if result:
            results["Primary_Focus"].append(result.get("Primary_Focus"))
            # Signature_Jargons 是陣列，轉為逗號分隔字串方便寫入 Excel
            jargons = result.get("Signature_Jargons", [])
            results["Signature_Jargons"].append(
                ", ".join(jargons) if isinstance(jargons, list) else str(jargons)
            )
            results["Has_Core_Interest_Support"].append(
                result.get("Has_Core_Interest_Support")
            )
            results["Key_Quote"].append(result.get("Key_Quote"))

            focus = result.get("Primary_Focus", "?")
            core  = result.get("Has_Core_Interest_Support", "?")
            print(f"✅  Focus={focus}  CoreInterest={core}")
        else:
            for key in results:
                results[key].append("ERROR")
            print("❌  分析失敗，標記為 ERROR")

        # 避免觸發 Rate Limit
        time.sleep(1.5)

    # 寫回 DataFrame
    for key, values in results.items():
        df[key] = values

    output_file = "Thematic_Scored_joint_declarations.xlsx"
    df.to_excel(output_file, index=False)

    # 統計摘要
    success_count = sum(1 for v in results["Primary_Focus"] if v not in (None, "ERROR"))
    error_count   = sum(1 for v in results["Primary_Focus"] if v == "ERROR")
    skip_count    = sum(1 for v in results["Primary_Focus"] if v is None)

    print("\n" + "=" * 60)
    print(f"🎉 分析完成！輸出檔案：{output_file}")
    print(f"   ✅ 成功：{success_count} 筆  ❌ 失敗：{error_count} 筆  ⚠️ 跳過：{skip_count} 筆")
    if error_count > 0:
        print("   → 請篩選 Primary_Focus == 'ERROR' 的列進行人工複核。")
    print("=" * 60)
