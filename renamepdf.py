import os
import pandas as pd

# 檔案路徑設定
excel_file_path = "LLM_and_PS.xls"
output_dir = "data/literature_pdfs"

print("🚀 開始執行批量重新命名腳本...\n")

try:
    # 讀取 Excel 檔案
    df = pd.read_excel(excel_file_path)
    
    rename_count = 0
    
    # 逐行檢查並對應檔名
    for index, row in df.iterrows():
        # 這是我們剛剛下載時產生的「未知檔名」格式
        old_filename = f"UnknownTitle{index}.pdf"
        old_filepath = os.path.join(output_dir, old_filename)
        
        # 檢查該檔案是否存在於資料夾中
        if os.path.exists(old_filepath):
            # 抓取真實的 Article Title
            real_title = str(row.get('Article Title', f"Recovered_Title_{index}"))
            
            # 清理檔名 (與下載時的保護機制一模一樣)
            safe_title = "".join([c for c in real_title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
            safe_title = safe_title[:80] # 限制長度
            new_filename = f"{safe_title}.pdf"
            new_filepath = os.path.join(output_dir, new_filename)
            
            # 執行重新命名
            try:
                os.rename(old_filepath, new_filepath)
                print(f"✅ 成功: {old_filename}  --->  {new_filename}")
                rename_count += 1
            except FileExistsError:
                print(f"⚠️ 跳過: {new_filename} 已經存在。")
            except Exception as e:
                print(f"❌ 錯誤: 無法重命名 {old_filename} ({e})")
                
    print(f"\n🎉 重新命名完成！共成功修改了 {rename_count} 個檔案。")

except Exception as e:
    print(f"程式執行發生錯誤: {e}")