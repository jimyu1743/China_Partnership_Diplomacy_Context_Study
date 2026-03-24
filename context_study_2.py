import os
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

# ==========================================
# 1. 載入環境變數與 API 設定
# ==========================================
load_dotenv("context_study.env")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_KEY:
    raise ValueError("⚠️ 找不到 ANTHROPIC_API_KEY！請確認 .env 檔案。")

client = Anthropic(api_key=ANTHROPIC_KEY)

# ==========================================
# 2. 定義「批判性話語分析 (CDA)」專屬 Prompt
# ==========================================
CLAUDE_SYSTEM_PROMPT = """
你是一位深諳「中國外交與戰略文化」的頂尖政治學理論家與批判話語分析（CDA）專家。
你的任務是比較同一個國家在「胡溫時期（P1, 2002-2012）」與「習近平時期（P2, 2013-2023）」與中國簽署的兩份《聯合聲明》。

請根據以下三個學術維度進行深度對比分析，並為學術論文的「機制追蹤（Process Tracing）」章節提供可直接引用的質性論述：

1. 【話語框架的轉移 (Discursive Shift)】：
P1 時期的文本是否主要聚焦於「物質與經貿互賴」？P2 時期的文本是否被強行植入了中國的「規範性話語（如人類命運共同體、三大倡議）」？

2. 【面子與政治忠誠 (Face & Political Loyalty)】：
對比兩份文本對中國「核心利益（特別是台灣問題）」的表態。P2 是否呈現出中國對「絕對政治效忠」的剛性要求？夥伴國的措辭是被動敷衍還是主動迎合？

3. 【戰略避險與霸權抗拒 (Strategic Hedging & Resistance)】：
(最關鍵) 仔細審視 P2 文本中，夥伴國是否置入了防範性的「西方秩序話語」（如：基於規則的國際秩序、去風險）來抵銷中國的政治壓力？這如何解釋雙方雖然經濟熱絡，但政治互信實質下降？

【輸出格式】
請以嚴謹的學術論文語氣撰寫，使用 Markdown 格式，包含適當的標題。必須大量引用兩份文本中的「原文金句」作為證據。字數約 800-1000 字。
"""

def analyze_discourse_with_claude(country_name, text_p1, text_p2):
    """呼叫 Claude Sonnet 4.6 進行深度思考與話語分析"""
    print(f"🧠 正在交由 Claude Sonnet 4.6 (開啟 8000 Token 延伸思考) 進行深層話語解剖: {country_name} ...")
    
    user_prompt = f"""
    請比較以下【{country_name}】在兩個不同時期的联合声明：
    
    [P1 胡溫時期文本 (2002-2012)]：
    <document>
    {text_p1}
    </document>
    
    [P2 習近平時期文本 (2013-2023)]：
    <document>
    {text_p2}
    </document>
    """

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6", # 🌟 升級為最新 Sonnet 4.6
            max_tokens=16000,          # 🌟 總配額提高，容納思考與輸出
            temperature=1.0,           # 🌟 啟用 thinking 時的硬性規定
            thinking={
                "type": "enabled",
                "budget_tokens": 8000  # 🌟 給予 8000 token 的深度內部推理空間
            },
            system=[
                {
                    "type": "text",
                    "text": CLAUDE_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"} # 🌟 啟用 Prompt Caching，大幅降低重複呼叫成本
                }
            ],
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # 🌟 提取機制更新：開啟 thinking 後，回傳會包含 'thinking' 與 'text' 兩個區塊
        # 我們只需將最終分析結果 (text 區塊) 寫入檔案
        final_text = next((block.text for block in response.content if block.type == 'text'), None)
        return final_text
        
    except Exception as e:
        print(f"❌ Claude API 處理失敗: {e}")
        return None

# ==========================================
# 3. 執行機制追蹤
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("🕵️‍♂️ 啟動：SMMR 第二層 - Claude Sonnet 4.6 機制追蹤與話語分析")
    print("="*60)
    
    input_file = "Scored_SMMR_Final_joint_declarations.xlsx" # 對接第一階段的最終檔案
    
    if not os.path.exists(input_file):
        print(f"❌ 找不到檔案 {input_file}！請確認第一階段是否已完成。")
        exit()
        
    df = pd.read_excel(input_file)
    
    target_country = input("👉 請輸入要進行 P1 vs P2 深度對比的國家名稱 (例如: 法國): ").strip()
    
    df_country = df[df['Country'] == target_country]
    
    if len(df_country) < 2:
        print(f"❌ 找不到【{target_country}】完整的 P1 與 P2 資料，請確認 Excel 內容。")
    else:
        text_p1 = str(df_country[df_country['Period'] == 'P1']['Text'].values[0])
        text_p2 = str(df_country[df_country['Period'] == 'P2']['Text'].values[0])
        
        # Sonnet 4.6 支援超長上下文，不再需要手動暴力截斷，讓模型看見全貌
        qualitative_analysis = analyze_discourse_with_claude(target_country, text_p1, text_p2)
        
        if qualitative_analysis:
            output_filename = f"Mechanism_Tracing_{target_country}.md"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(qualitative_analysis)
            
            print(f"\n🎉 分析完成！")
            print(f"📄 論文級別的質性分析報告已儲存為：{output_filename}")