import pandas as pd
import re
import os

# ==========================================
# 1. 設定檔案路徑
# ==========================================
p1_file = '聯合聲明_P1.txt'
p2_file = '聯合聲明_P2.txt'
output_file = 'joint_declarations.xlsx'

# ==========================================
# 2. 定義解析函數 (Parser)
# ==========================================
def parse_diplomatic_text(filepath, period_label):
    """讀取 txt 檔，並根據 '### 國家名' 進行切分"""
    if not os.path.exists(filepath):
        print(f"⚠️ 找不到檔案: {filepath}")
        return []

    # 嘗試不同的編碼格式 (防範 Windows 記事本的編碼問題)
    encodings = ['utf-8', 'utf-8-sig', 'cp950', 'gbk']
    content = ""
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            break # 成功讀取就跳出
        except UnicodeDecodeError:
            continue

    if not content:
        print(f"❌ 無法解析 {filepath} 的編碼，請確認檔案為 UTF-8 格式。")
        return []

    # 利用正則表達式尋找 "### 國家名" 進行切塊
    # 這會將文本切成 [空白/前言, 國家1, 文本1, 國家2, 文本2...]
    blocks = re.split(r'###\s*([^\n]+)', content)
    
    data = []
    # blocks[0] 通常是第一個 ### 之前的空白或雜訊，我們從索引 1 開始抓取
    for i in range(1, len(blocks), 2):
        country_name = blocks[i].strip()
        text_content = blocks[i+1].strip()
        
        data.append({
            'Country': country_name,
            'Period': period_label,
            'Text': text_content
        })
        print(f"  - 成功擷取: {country_name} ({period_label})，字數: {len(text_content)}")
        
    return data

# ==========================================
# 3. 執行轉換與合併
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("📂 啟動：中國外交聯合聲明 文本結構化轉換器")
    print("="*50)

    # 解析 P1 與 P2
    print("\n⏳ 正在處理 P1 文本...")
    p1_data = parse_diplomatic_text(p1_file, 'P1')
    
    print("\n⏳ 正在處理 P2 文本...")
    p2_data = parse_diplomatic_text(p2_file, 'P2')

    # 合併資料
    all_data = p1_data + p2_data

    if not all_data:
        print("\n❌ 轉換失敗：沒有找到任何資料。請確認文字檔內是否有加上 '### 國家名' 標籤。")
    else:
        # 轉換為 pandas DataFrame 並匯出 Excel
        df = pd.DataFrame(all_data)
        df.to_excel(output_file, index=False)
        print("\n" + "="*50)
        print(f"🎉 轉換大功告成！")
        print(f"📊 共處理 {len(df)} 筆個案，已成功匯出至：{output_file}")
        print("👉 您現在可以開始執行 SMMR_Automated_Coder_V2.py 來呼叫 AI 了！")
        print("="*50)