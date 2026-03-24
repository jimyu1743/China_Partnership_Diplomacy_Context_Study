import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

# ==========================================
# 1. 載入資料與自訂停用詞 (過濾外交廢話)
# ==========================================
input_file = "Scored_SMMR_Final_joint_declarations.xlsx" # 讀取您剛跑完的 Excel

if not os.path.exists(input_file):
    print(f"❌ 找不到檔案 {input_file}，請確認檔名是否正確。")
    exit()

df = pd.read_excel(input_file)

# 這些是沒有政治學分析價值的「外交過渡詞」，我們強迫 AI 忽略它們
stop_words = set([
    "双方", "两国", "合作", "发展", "关系", "关于", "强调", "指出", "表示",
    "同意", "认为", "进一步", "全面", "战略", "伙伴", "我们", "他们",
    "进行", "支持", "领域", "促进", "加强", "推动", "实现", "共同", 
    "深化", "继续", "重申", "致力于", "高度评价", "一致同意"
])

# 載入一些專有名詞以防被切斷 (例如不要把 人类命运共同体 切成 人类/命运/共同体)
jieba.add_word("人类命运共同体")
jieba.add_word("全球发展倡议")
jieba.add_word("全球安全倡议")
jieba.add_word("全球文明倡议")
jieba.add_word("一带一路")

def preprocess_text(text):
    if pd.isna(text): return ""
    # 使用 jieba 進行中文斷詞
    words = jieba.lcut(str(text))
    # 過濾停用詞與單一字元 (例如 '的', '在')
    return " ".join([w for w in words if w not in stop_words and len(w) > 1])

print("⏳ 正在進行中文斷詞與清理，請稍候...")
df['Processed_Text'] = df['Text'].apply(preprocess_text)

# ==========================================
# 2. 計算 TF-IDF 矩陣
# ==========================================
# max_df=0.8 表示如果一個詞在 80% 的文章都出現，它就沒鑑別度；min_df=2 表示至少要在兩篇文章出現過
vectorizer = TfidfVectorizer(max_df=0.8, min_df=2) 
tfidf_matrix = vectorizer.fit_transform(df['Processed_Text'])
feature_names = vectorizer.get_feature_names_out()

# ==========================================
# 3. 萃取 P1 與 P2 的專屬特徵詞
# ==========================================
def get_top_keywords(period_name, top_n=20):
    idx = df[df['Period'] == period_name].index
    if len(idx) == 0: return []
    
    # 計算該時期 TF-IDF 的平均值
    period_tfidf = tfidf_matrix[idx].mean(axis=0)
    period_tfidf = np.array(period_tfidf).flatten()
    
    # 排序找出分數最高的詞
    top_indices = period_tfidf.argsort()[-top_n:][::-1]
    return [(feature_names[i], round(period_tfidf[i], 4)) for i in top_indices]

print("\n" + "="*60)
print("🔍 中國外交聲明 潛在變數挖掘引擎 (TF-IDF)")
print("="*60)

print("\n📉 [P1 胡溫時期] 核心特徵詞 (代表過去的舊議程):")
for word, score in get_top_keywords('P1', 15):
    print(f" - {word:<10} (TF-IDF: {score})")

print("\n📈 [P2 習近平時期] 核心特徵詞 (🔥 您的新變數金礦):")
for word, score in get_top_keywords('P2', 15):
    print(f" - {word:<10} (TF-IDF: {score})")

print("\n💡 導師建議：請觀察 P2 的關鍵字。是否出現了『氣候變化』、『生物多樣性』或『南南合作』？這就是您可以加入 Codebook 的新變數！")