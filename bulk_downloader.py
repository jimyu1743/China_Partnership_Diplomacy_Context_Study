import os
import time
import pandas as pd
import requests
import urllib.parse
import re
from dotenv import load_dotenv

# ==========================================
# 1. 載入環境變數與安全檢查
# ==========================================
load_dotenv("context_study.env")

YOUR_EMAIL = os.getenv("UNPAYWALL_EMAIL")
OPENAI_KEY = os.getenv("OPENAI_API_KEY") 

if not YOUR_EMAIL:
    raise ValueError("⚠️ 找不到 UNPAYWALL_EMAIL！請確認 context_study.env 檔案已正確設定。")

print(f"✅ 環境變數載入成功！目前使用的信箱為: {YOUR_EMAIL}")

HEADERS = {
    "User-Agent": f"ResearchScript/1.0 (mailto:{YOUR_EMAIL})"
}

output_dir = "data/literature_pdfs"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 2. 互動式：讓使用者輸入檔案名稱
# ==========================================
print("\n" + "="*50)
print("📚 歡迎使用文獻自動獲取引擎 (進階探測版)")
print("="*50)

while True:
    excel_input = input("👉 請輸入您的 Excel 檔案名稱 (包含副檔名，例如 LLM_and_PS.xls)\n[直接按 Enter 預設讀取 'LLM_and_PS.xls']: ").strip()
    
    # 處理預設值
    if not excel_input:
        excel_file_path = "LLM_and_PS.xls"
    else:
        excel_file_path = excel_input

    # 檢查檔案是否存在
    if os.path.exists(excel_file_path):
        print(f"\n📂 成功找到檔案：{excel_file_path}，正在讀取中...")
        break
    else:
        print(f"\n❌ 錯誤：在當前資料夾找不到 '{excel_file_path}'！")
        print("💡 提示：請確認檔名是否正確（注意 .xls 與 .xlsx 的差異），並重試。\n")

# 讀取 Excel
df = pd.read_excel(excel_file_path)

# ==========================================
# 3. 智慧欄位偵測 (解決 Unknown 檔名問題)
# ==========================================
if 'Article Title' in df.columns:
    title_col = 'Article Title'
elif 'Title' in df.columns:
    title_col = 'Title'
else:
    title_col = None
    print(f"⚠️ 警告：找不到標題欄位。目前的欄位有：{df.columns.tolist()}")

doi_col = 'DOI' if 'DOI' in df.columns else None

# ==========================================
# 4. 定義核心 API 呼叫模組
# ==========================================
def get_doi_from_crossref(title):
    print(f"   🔍 正在 Crossref 搜尋 DOI: {str(title)[:30]}...")
    encoded_title = urllib.parse.quote(str(title))
    url = f"https://api.crossref.org/works?query.title={encoded_title}&select=DOI,title&rows=1"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            items = res.json().get("message", {}).get("items", [])
            if items:
                return items[0].get("DOI")
    except Exception as e:
        print(f"   ❌ Crossref 請求失敗: {e}")
    return None

def get_pdf_from_semantic_scholar(doi):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{doi}?fields=openAccessPdf"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data.get("openAccessPdf"):
                return data["openAccessPdf"].get("url")
    except Exception:
        pass
    return None

def get_pdf_from_openalex(doi):
    url = f"https://api.openalex.org/works/https://doi.org/{doi}?mailto={YOUR_EMAIL}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            oa_data = data.get("open_access", {})
            if oa_data.get("is_oa") and oa_data.get("oa_url"):
                return oa_data.get("oa_url")
    except Exception:
        pass
    return None

# ==========================================
# 5. 執行瀑布流下載主邏輯
# ==========================================
print(f"\n🚀 開始執行多模組合法批量下載工作流！共計 {len(df)} 筆資料。")

for index, row in df.iterrows():
    # 動態抓取標題，若無則編號
    if title_col and pd.notna(row[title_col]):
        raw_title = str(row[title_col])
    else:
        raw_title = f"Document_{index}"
        
    doi = str(row[doi_col]).strip() if doi_col and pd.notna(row[doi_col]) else ""
    
    print(f"\n[{index+1}/{len(df)}] 處理文獻: {raw_title[:50]}...")
    
    # 步驟 A：如果沒有 DOI，呼叫 Crossref 補充
    if not doi:
        doi = get_doi_from_crossref(raw_title)
        time.sleep(1)
        
    if not doi:
        print("   ⚠️ 無法找到對應的 DOI，跳過此篇。")
        continue
    else:
        print(f"   ✅ 確認 DOI: {doi}")

    # 步驟 B：瀑布流全文檢索
    pdf_url = get_pdf_from_semantic_scholar(doi)
    if pdf_url:
        print("   ✅ Semantic Scholar 找到開源全文！")
    else:
        time.sleep(1)
        pdf_url = get_pdf_from_openalex(doi)
        if pdf_url:
            print("   ✅ OpenAlex 找到開源全文！")

    # 步驟 C：下載與儲存 (加入防護破解與輔助模式)
    if pdf_url:
        try:
            print(f"   ⬇️ 正在下載 PDF...")
            # 開啟轉址追蹤 (allow_redirects=True)
            pdf_response = requests.get(pdf_url, headers=HEADERS, timeout=15, allow_redirects=True)
            content_type = pdf_response.headers.get('Content-Type', '').lower()
            
            # 只要 Content-Type 包含 pdf 或是通用的二進位格式 (octet-stream) 都放行
            if pdf_response.status_code == 200 and ('pdf' in content_type or 'octet-stream' in content_type):
                
                # 優化檔名處理 (保留空白，刪除不合法字元)
                safe_title = re.sub(r'[\\/*?:"<>|]', "", raw_title)
                safe_title = safe_title.replace('\n', ' ').replace('\r', '').strip()
                safe_title = safe_title[:80] # 限制長度避免路徑過長錯誤
                
                file_path = os.path.join(output_dir, f"{safe_title}.pdf")
                
                # 最終檢驗：檢查檔案的 Magic Number 是否真的是 PDF
                if pdf_response.content.startswith(b'%PDF'):
                    with open(file_path, 'wb') as f:
                        f.write(pdf_response.content)
                    print(f"   🎉 下載成功！已儲存為: {safe_title}.pdf")
                else:
                    print(f"   ⚠️ 伺服器回傳了網頁防護 (可能遇到機器人驗證)。")
                    print(f"   👉 請按住 Ctrl 並點擊手動下載: {pdf_url}")
            else:
                print(f"   ⚠️ 目標為出版社跳轉網頁或被阻擋。")
                print(f"   👉 請按住 Ctrl 並點擊手動下載: {pdf_url}")
        except Exception as e:
            print(f"   ❌ 下載過程發生錯誤: {e}")
    else:
        print("   🔒 此文獻可能無合法的 Open Access 版本 (請透過學校圖書館查詢)。")

    # 學術倫理紅線：跨文獻處理間隔
    time.sleep(1.5)

print(f"\n🏁 批量下載作業結束！請至 {output_dir}/ 資料夾查看。")