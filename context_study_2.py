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
    raise ValueError("⚠️ 找不到 ANTHROPIC_API_KEY！")

client = Anthropic(api_key=ANTHROPIC_KEY)

# ==========================================
# 2. 定義「批判性話語分析 (CDA)」專屬 Prompt
# ==========================================
# 這是餵給 Claude 的靈魂，要求它以建構主義與話語霸權的視角進行解讀
CLAUDE_SYSTEM_PROMPT = """
你是一位深諳「中國外交與戰略文化」的頂尖政治學理論家與批判話語分析（CDA）專家。
你的任務是比較同一個國家在「胡溫時期（P1, 2002-2012）」與「習近平時期（P2, 2013-2023）」與中國簽署的兩份《聯合聲明》。

請根據以下三個學術維度進行深度對比分析，並為學術論文的「機制追蹤（Process Tracing）」章節提供可直接引用的質性論述：

1. 【話語框架的轉移 (Discursive Shift)】：
P1 時期的文本是否主要聚焦於「物質與經貿互賴（如貿易額、基礎建設）」？P2 時期的文本是否被強行植入了中國的「規範性話語（如人類命運共同體、三大倡議）」？

2. 【面子與政治忠誠 (Face & Political Loyalty)】：
對比兩份文本對中國「核心利益（特別是台灣問題）」的表態。P2 是否呈現出中國對「絕對政治效忠」的剛性要求？夥伴國的措辭是被動敷衍還是主動迎合？

3. 【戰略避險與霸權抗拒 (Strategic Hedging & Resistance)】：
(最關鍵) 仔細審視 P2 文本中，夥伴國是否置入了防範性的「西方秩序話語」（如：基於規則的國際秩序、航行自由、去風險）來抵銷中國的政治壓力？這如何解釋雙方雖然經濟熱絡，但政治互信實質下降？

【輸出格式】
請以嚴謹的學術論文語氣撰寫，使用 Markdown 格式，包含適當的標題。必須大量引用兩份文本中的「原文金句」作為證據。字數約 800-1000 字。
"""

def analyze_discourse_with_claude(country_name, text_p1, text_p2):
    """呼叫 Claude 3.5 Sonnet / Opus 進行雙時期文本深度對比"""
    print(f"🧠 正在交由 Claude 進行深層話語分析: {country_name} ...")
    
    user_prompt = f"""
    請比較以下【{country_name}】在兩個不同時期的聯合聲明：
    
    [P1 胡溫時期文本 (2002-2012)]：
    {text_p1}
    
    [P2 習近平時期文本 (2013-2023)]：
    {text_p2}
    """

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620", # 亦可替換為 claude-3-opus-20240229 以獲取最高深度
            max_tokens=2000,
            temperature=0.3, # 質性分析可稍微提高溫度至 0.3，增加論述的豐富度
            system=CLAUDE_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        print(f"❌ Claude API 處理失敗: {e}")
        return None

# ==========================================
# 3. 執行機制追蹤 (以單一國家為例)
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("🕵️‍♂️ 啟動：SMMR 第二層 - Claude 機制追蹤與話語分析")
    print("="*50)
    
    # 假設您已經用 GPT-4o 跑完了第一層，產生了包含 P1 與 P2 的 Excel
    input_file = "Scored_joint_declarations.xlsx"
    df = pd.read_excel(input_file)
    
    # 讓使用者選擇要進行「深度解剖」的國家 (例如：德國、越南、巴基斯坦)
    target_country = input("👉 請輸入要進行 P1 vs P2 深度對比的國家名稱 (例如: 德國): ").strip()
    
    # 篩選該國的 P1 與 P2 文本
    df_country = df[df['Country'] == target_country]
    
    if len(df_country) < 2:
        print(f"❌ 找不到【{target_country}】完整的 P1 與 P2 資料，請確認 Excel 內容。")
    else:
        # 提取文本 (假設 DataFrame 中有 Period 欄位標示 P1 或 P2)
        text_p1 = df_country[df_country['Period'] == 'P1']['Text'].values[0]
        text_p2 = df_country[df_country['Period'] == 'P2']['Text'].values[0]
        
        # 進行長度防呆處理
        if len(text_p1) > 10000: text_p1 = text_p1[:5000] + "\n...\n" + text_p1[-5000:]
        if len(text_p2) > 10000: text_p2 = text_p2[:5000] + "\n...\n" + text_p2[-5000:]
        
        # 呼叫 Claude
        qualitative_analysis = analyze_discourse_with_claude(target_country, text_p1, text_p2)
        
        if qualitative_analysis:
            # 將結果存成 Markdown 檔，方便直接貼入您的論文中
            output_filename = f"Mechanism_Tracing_{target_country}.md"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(qualitative_analysis)
            
            print(f"\n🎉 分析完成！")
            print(f"📄 論文級別的質性分析報告已儲存為：{output_filename}")