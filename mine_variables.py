import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

# ==========================================
# 1. 學術編碼簿：中國外交話語特徵七大維度詞典
# 💡 導師優化：已自動為您補齊「簡體中文」對應詞，確保能精準捕捉中共官方文本
# ==========================================
USER_LEXICON = {
    "經濟貿易": [
        "貿易", "贸易", "經濟", "经济", "投資", "投资", "市場", "市场", 
        "自由貿易", "自由贸易", "進出口", "进出口", "出口", "進口", "进口", 
        "互利", "雙邊貿易", "双边贸易", "關稅", "关税", "貿易協定", "贸易协定", 
        "產業鏈", "产业链", "供應鏈", "供应链", "trade", "economic", 
        "investment", "market", "tariff", "supply chain"
    ],
    "金融貨幣": [
        "金融", "銀行", "银行", "貨幣", "货币", "匯率", "汇率", "融資", "融资", 
        "資本", "资本", "債務", "债务", "股市", "人民幣", "人民币", "外匯", "外汇", 
        "開發銀行", "开发银行", "亞投行", "亚投行", "financial", "banking", 
        "currency", "capital", "debt", "monetary"
    ],
    "安全防衛": [
        "安全", "防衛", "防卫", "軍事", "军事", "反恐", "維和", "维和", 
        "武器", "核", "軍備", "军备", "非傳統安全", "非传统安全", 
        "海洋安全", "網路安全", "网络安全", "security", "defense", 
        "military", "counter-terrorism", "nuclear", "cybersecurity", "peacekeeping"
    ],
    "科技創新": [
        "科技", "技術", "技术", "數位", "数位", "數字", "数字", "人工智慧", "人工智能", 
        "創新", "创新", "研發", "研发", "網路", "网络", "數據", "数据", "航太", "航天", 
        "衛星", "卫星", "5G", "智慧", "智能", "數字經濟", "数字经济", "technology", 
        "innovation", "digital", "artificial intelligence", "data", "satellite", 
        "cyber", "R&D"
    ],
    "政治主權": [
        "主權", "主权", "領土", "领土", "核心利益", "台灣", "台湾", "一個中國", "一个中国", 
        "不干涉", "尊重", "主權平等", "主权平等", "領土完整", "领土完整", "反對分裂", "反对分裂", 
        "sovereignty", "territorial integrity", "non-interference", "one-China", "Taiwan"
    ],
    "規範性話語": [
        "人類命運共同體", "人类命运共同体", "命運共同體", "命运共同体", "一帶一路", "一带一路", 
        "絲綢之路", "丝绸之路", "全球發展倡議", "全球发展倡议", "全球安全倡議", "全球安全倡议", 
        "全球文明倡議", "全球文明倡议", "新型大國關係", "新型大国关系", "新型國際關係", "新型国际关系", 
        "互聯互通", "互联互通", "Belt and Road", "Global Development Initiative", 
        "Global Security Initiative", "community of shared future"
    ],
    "多邊國際秩序": [
        "聯合國", "联合国", "多邊", "多边", "國際秩序", "国际秩序", "全球治理", "多極", "多极", 
        "國際法", "国际法", "基於規則", "基于规则", "世貿組織", "世贸组织", "氣候", "气候", 
        "可持續發展", "可持续发展", "United Nations", "multilateral", "international order", 
        "rules-based", "WTO", "climate", "sustainable development"
    ]
}

# 強制 jieba 把這些專有名詞當作「一個完整的詞」來切，不要切碎
for category, words in USER_LEXICON.items():
    for word in words:
        jieba.add_word(word)

# 外交廢話停用詞 (包含繁簡體)
stop_words = set([
    "雙方", "双方", "兩國", "两国", "合作", "發展", "发展", "關係", "关系", "關於", "关于", 
    "強調", "强调", "指出", "表示", "同意", "認為", "认为", "進一步", "进一步", "全面", 
    "戰略", "战略", "夥伴", "伙伴", "我們", "我们", "他們", "他们", "進行", "进行", 
    "支持", "領域", "领域", "促進", "促进", "加強", "加强", "推動", "推动", "實現", "实现", 
    "共同", "深化", "繼續", "继续", "重申", "致力於", "致力于", "高度評價", "高度评价", 
    "一致同意", "的", "和", "與", "与", "在", "等"
])

# ==========================================
# 2. 資料載入與前處理 (計算標準化頻率)
# ==========================================
input_file = "joint_declarations.xlsx" 

if not os.path.exists(input_file):
    print(f"❌ 找不到檔案 {input_file}。")
    exit()

df = pd.read_excel(input_file)

if 'Text' not in df.columns and 'Declaration_Text' in df.columns:
    df.rename(columns={'Declaration_Text': 'Text'}, inplace=True)

def preprocess_and_count(text):
    if pd.isna(text): 
        return "", 0, {k: 0 for k in USER_LEXICON.keys()}
    
    words = jieba.lcut(str(text))
    total_words = len(words)
    
    # 統計七大維度詞彙出現次數
    category_counts = {k: 0 for k in USER_LEXICON.keys()}
    for word in words:
        for category, lexicon in USER_LEXICON.items():
            if word in lexicon:
                category_counts[category] += 1
                
    clean_text = " ".join([w for w in words if w not in stop_words and len(w) > 1])
    return clean_text, total_words, category_counts

print("⏳ 正在進行中文斷詞、七大維度詞典計數與資料標準化，請稍候...")

df[['Processed_Text', 'Total_Words', 'Lexicon_Counts']] = df['Text'].apply(
    lambda x: pd.Series(preprocess_and_count(x))
)

# 展開字典並計算「每萬字出現次數」
for category in USER_LEXICON.keys():
    df[f'{category}_RawCount'] = df['Lexicon_Counts'].apply(lambda x: x[category])
    df[f'{category}_Normalized (per 10k)'] = np.where(
        df['Total_Words'] > 0, 
        (df[f'{category}_RawCount'] / df['Total_Words']) * 10000, 
        0
    )

# ==========================================
# 3. 軌道一：驗證性研究 (七大維度的跨期板塊移動)
# ==========================================
print("\n" + "="*70)
print("📊 軌道一：七大理論維度標準化頻率比較 (每萬字平均提及次數)")
print("="*70)

if 'Period' not in df.columns and 'Year' in df.columns:
    df['Period'] = np.where(df['Year'] <= 2012, 'P1', 'P2')

if 'Period' in df.columns:
    summary_df = df.groupby('Period')[[f'{c}_Normalized (per 10k)' for c in USER_LEXICON.keys()]].mean().round(2)
    print(summary_df.T)
    print("\n💡 學術解讀：")
    print("若 P2 在「政治主權」與「規範性話語」的數值顯著高於 P1，")
    print("而「經濟貿易」維持不變或下降，即完美證實習近平外交的『泛政治化』與『論述霸權擴張』！")

# ==========================================
# 4. 軌道二：探索性研究 (TF-IDF 潛在變數挖掘)
# ==========================================
print("\n" + "="*70)
print("🔍 軌道二：潛在議題挖掘引擎 (TF-IDF 演算法)")
print("="*70)

vectorizer = TfidfVectorizer(max_df=0.8, min_df=2) 
tfidf_matrix = vectorizer.fit_transform(df['Processed_Text'])
feature_names = vectorizer.get_feature_names_out()

def get_top_keywords(period_name, top_n=15):
    idx = df[df['Period'] == period_name].index
    if len(idx) == 0: return []
    period_tfidf = tfidf_matrix[idx].mean(axis=0)
    period_tfidf = np.array(period_tfidf).flatten()
    top_indices = period_tfidf.argsort()[-top_n:][::-1]
    return [(feature_names[i], round(period_tfidf[i], 4)) for i in top_indices]

if 'Period' in df.columns:
    print("\n📉 [P1 胡溫時期] 獨特高頻特徵詞:")
    for word, score in get_top_keywords('P1', 10):
        print(f" - {word:<10} (TF-IDF: {score})")

    print("\n📈 [P2 習近平時期] 獨特高頻特徵詞:")
    for word, score in get_top_keywords('P2', 10):
        print(f" - {word:<10} (TF-IDF: {score})")

# 匯出結果
output_file = "Lexicon_7Dimensions_Results.xlsx"
df.drop(columns=['Lexicon_Counts', 'Processed_Text']).to_excel(output_file, index=False)
print("\n" + "="*70)
print(f"✅ 分析完成！所有跨期標準化詞頻數據已匯出至：{output_file}")
print("="*70)