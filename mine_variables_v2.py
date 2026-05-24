"""
mine_variables_v2.py — 修正版

修正項目：
  1. 分母虛增：total_words 排除標點、空白等無意義 token
  2. 英文多字詞組：改用 regex 於原始文本搜尋，不依賴 jieba 切分穩定性
  3. 詞典查詢：list → set，查詢複雜度從 O(n) 降至 O(1)
  4. 繁簡轉換：安裝 zhconv 後自動將文本統一為簡體，提升 jieba 準確度

說明：
  跨類別計數（同一 token 可計入多個語意重疊的類別）屬學術設計選擇，
  本版保留此行為，但在計數迴圈中加入說明；若需互斥分類，取消 break 的註解即可。
"""

import os
import re
import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# 0. 可選依賴：繁→簡轉換（pip install zhconv）
# ==========================================
try:
    import zhconv
    def to_simplified(text: str) -> str:
        return zhconv.convert(text, 'zh-hans')
    print("✅ 已載入 zhconv，啟用繁→簡自動轉換。")
except ImportError:
    def to_simplified(text: str) -> str:
        return text
    print("⚠️  未安裝 zhconv，跳過繁簡轉換（可執行 pip install zhconv 啟用）。")

# ==========================================
# 1. 學術編碼簿：七大維度詞典
# ==========================================
USER_LEXICON = {
    "經濟貿易": [
        "貿易", "贸易", "經濟", "经济", "投資", "投资", "市場", "市场",
        "自由貿易", "自由贸易", "進出口", "进出口", "出口", "進口", "进口",
        "互利", "雙邊貿易", "双边贸易", "關稅", "关税", "貿易協定", "贸易协定",
        "產業鏈", "产业链", "供應鏈", "供应链",
        "trade", "economic", "investment", "market", "tariff", "supply chain"
    ],
    "金融貨幣": [
        "金融", "銀行", "银行", "貨幣", "货币", "匯率", "汇率", "融資", "融资",
        "資本", "资本", "債務", "债务", "股市", "人民幣", "人民币", "外匯", "外汇",
        "開發銀行", "开发银行", "亞投行", "亚投行",
        "financial", "banking", "currency", "capital", "debt", "monetary"
    ],
    "安全防衛": [
        "安全", "防衛", "防卫", "軍事", "军事", "反恐", "維和", "维和",
        "武器", "核", "軍備", "军备", "非傳統安全", "非传统安全",
        "海洋安全", "網路安全", "网络安全",
        "總體國家安全觀", "总体国家安全观", # 補入習時期重要安全概念
        "security", "defense", "military", "counter-terrorism", "nuclear",
        "cybersecurity", "peacekeeping", "overall national security concept", "holistic national security concept"
    ],
    "科技創新": [
        "科技", "技術", "技术", "數位", "数位", "數字", "数字", "人工智慧", "人工智能",
        "創新", "创新", "研發", "研发", "網路", "网络", "數據", "数据", "航太", "航天",
        "衛星", "卫星", "5G", "智慧", "智能", "數字經濟", "数字经济",
        "中國製造2025", "中国制造2025", "軍民融合", "军民融合", # 補入習時期重要科技/軍民政策
        "technology", "innovation", "digital", "artificial intelligence", "data",
        "satellite", "cyber", "R&D", "Made in China 2025", "Military-Civil Fusion"
    ],
    "政治主權": [
        "主權", "主权", "領土", "领土", "核心利益", "台灣", "台湾",
        "一個中國", "一个中国", "不干涉", "尊重", "主權平等", "主权平等",
        "領土完整", "领土完整", "反對分裂", "反对分裂",
        "sovereignty", "territorial integrity", "non-interference", "one-China", "Taiwan"
    ],
    "規範性話語": [
        # --- 習近平時期 (P2) 核心話語 ---
        "人類命運共同體", "人类命运共同体", "命運共同體", "命运共同体",
        "一帶一路", "一带一路", "絲綢之路", "丝绸之路",
        "全球發展倡議", "全球发展倡议", "全球安全倡議", "全球安全倡议",
        "全球文明倡議", "全球文明倡议", "新型大國關係", "新型大国关系",
        "新型國際關係", "新型国际关系", "互聯互通", "互联互通",
        "大國外交", "大国外交", "奮發有為", "奋发有为", # 補入習時期核心外交定調
        "Belt and Road", "Global Development Initiative",
        "Global Security Initiative", "community of shared future", "major country diplomacy", "striving for achievement",
        
        # --- 江澤民、胡錦濤時期 (P1) 核心話語 ---
        "和諧世界", "和谐世界", 
        "和平發展", "和平发展", "和平崛起", 
        "國際關係民主化", "国际关系民主化",
        "國際政治經濟新秩序", "国际政治经济新秩序",
        "和平共處五項原則", "和平共处五项原则",
        "韜光養晦", "韬光养晦",
        "harmonious world", "peaceful development", "peaceful rise",
        "democratization of international relations", 
        "new international political and economic order",
        "five principles of peaceful coexistence"
    ],
    "多邊國際秩序": [
        "聯合國", "联合国", "多邊", "多边", "國際秩序", "国际秩序",
        "全球治理", "多極", "多极", "國際法", "国际法", "基於規則", "基于规则",
        "世貿組織", "世贸组织", "氣候", "气候", "可持續發展", "可持续发展",
        "United Nations", "multilateral", "international order",
        "rules-based", "WTO", "climate", "sustainable development"
    ]
}

# ==========================================
# 2. 前處理：分離英文多字詞組 vs. 其餘詞彙
#
# 修正說明（問題三）：
#   含空格的英文短語（如 "Belt and Road"、"supply chain"）無法穩定由
#   jieba.add_word 合併為單一 token，因此改以 regex 直接搜尋原始文本，
#   並從 jieba 自訂詞典中排除，避免雙重計算。
# ==========================================
def _is_english_multiword(term: str) -> bool:
    has_space = ' ' in term
    has_chinese = any('一' <= c <= '鿿' for c in term)
    return has_space and not has_chinese

ENGLISH_MULTIWORD: dict[str, list[str]] = {}
JIEBA_TERMS: dict[str, list[str]] = {}

for _cat, _terms in USER_LEXICON.items():
    ENGLISH_MULTIWORD[_cat] = [t for t in _terms if _is_english_multiword(t)]
    JIEBA_TERMS[_cat]       = [t for t in _terms if not _is_english_multiword(t)]

# 修正說明（問題四）：詞典改用 set，查詢由 O(n) 降為 O(1)
JIEBA_LEXICON_SETS: dict[str, set[str]] = {
    cat: set(terms) for cat, terms in JIEBA_TERMS.items()
}

# 僅將非多字英文詞組加入 jieba 自訂詞典
for _cat, _terms in JIEBA_TERMS.items():
    for _word in _terms:
        jieba.add_word(_word)

# 停用詞（僅用於 TF-IDF 前處理，不影響詞典計數）
STOP_WORDS = {
    "雙方", "双方", "兩國", "两国", "合作", "發展", "发展", "關係", "关系", "關於", "关于",
    "強調", "强调", "指出", "表示", "同意", "認為", "认为", "進一步", "进一步", "全面",
    "戰略", "战略", "夥伴", "伙伴", "我們", "我们", "他們", "他们", "進行", "进行",
    "支持", "領域", "领域", "促進", "促进", "加強", "加强", "推動", "推动", "實現", "实现",
    "共同", "深化", "繼續", "继续", "重申", "致力於", "致力于", "高度評價", "高度评价",
    "一致同意", "的", "和", "與", "与", "在", "等",
}

# ==========================================
# 3. 核心函數：前處理 + 計數
# ==========================================
_PUNCTUATION_RE = re.compile(r'^[^\w]+$')

def _is_meaningful_token(token: str) -> bool:
    """
    修正說明（問題一）：
      原版以 len(tokens) 作分母，包含標點與空白等無意義 token，
      會系統性壓低標準化頻率且跨文本影響程度不一。
      現改為只計長度 > 1 且非純標點的有意義 token。
    """
    return len(token.strip()) > 1 and not _PUNCTUATION_RE.match(token)

def preprocess_and_count(raw_text) -> tuple[str, int, dict]:
    if pd.isna(raw_text):
        return "", 0, {k: 0 for k in USER_LEXICON}

    # 可選：統一轉為簡體，提升 jieba 對繁體文本的切分準確度
    text = to_simplified(str(raw_text))

    category_counts = {k: 0 for k in USER_LEXICON}

    # ── ① 英文多字詞組：regex 搜尋原始文本 ──────────────────────
    #    在 jieba 切分之前處理，完全不依賴 jieba 對英文短語的識別
    for cat, phrases in ENGLISH_MULTIWORD.items():
        for phrase in phrases:
            category_counts[cat] += len(
                re.findall(re.escape(phrase), text, re.IGNORECASE)
            )

    # ── ② jieba 切分 ─────────────────────────────────────────────
    tokens = jieba.lcut(text)

    # ── ③ 分母：僅計有意義的 token（修正問題一）─────────────────
    meaningful_tokens = [t for t in tokens if _is_meaningful_token(t)]
    total_words = len(meaningful_tokens)

    # ── ④ Token 詞典比對 ─────────────────────────────────────────
    #    設計選擇：同一 token 可歸屬多個語意重疊的類別（如「网络安全」
    #    同時計入安全防衛與科技創新）。
    #    若研究設計要求互斥分類，取消下方 `break` 的註解。
    for token in meaningful_tokens:
        for cat, lexicon_set in JIEBA_LEXICON_SETS.items():
            if token in lexicon_set:
                category_counts[cat] += 1
                # break  # 互斥模式：每個 token 只歸入第一個匹配的類別

    # ── ⑤ 清洗文本（供 TF-IDF 使用）────────────────────────────
    clean_text = " ".join(
        t for t in meaningful_tokens if t not in STOP_WORDS
    )

    return clean_text, total_words, category_counts

# ==========================================
# 4. 資料載入與前處理
# ==========================================
input_file = "joint_declarations.xlsx"

if not os.path.exists(input_file):
    print(f"❌ 找不到檔案 {input_file}。")
    exit()

df = pd.read_excel(input_file)

if 'Text' not in df.columns and 'Declaration_Text' in df.columns:
    df.rename(columns={'Declaration_Text': 'Text'}, inplace=True)

print("⏳ 正在進行中文斷詞、七大維度詞典計數與資料標準化，請稍候...")

df[['Processed_Text', 'Total_Words', 'Lexicon_Counts']] = df['Text'].apply(
    lambda x: pd.Series(preprocess_and_count(x))
)

# 展開字典並計算「每萬字出現次數」
for category in USER_LEXICON:
    df[f'{category}_RawCount'] = df['Lexicon_Counts'].apply(lambda x: x[category])
    df[f'{category}_Normalized (per 10k)'] = np.where(
        df['Total_Words'] > 0,
        (df[f'{category}_RawCount'] / df['Total_Words']) * 10000,
        0
    )

# ==========================================
# 5. 軌道一：七大維度跨期比較
# ==========================================
print("\n" + "=" * 70)
print("📊 軌道一：七大理論維度標準化頻率比較（每萬字平均提及次數）")
print("=" * 70)

if 'Period' not in df.columns and 'Year' in df.columns:
    df['Period'] = np.where(df['Year'] <= 2012, 'P1', 'P2')

if 'Period' in df.columns:
    summary_df = df.groupby('Period')[
        [f'{c}_Normalized (per 10k)' for c in USER_LEXICON]
    ].mean().round(2)
    print(summary_df.T)
    print("\n💡 學術解讀：")
    print("若 P2 在「政治主權」與「規範性話語」的數值顯著高於 P1，")
    print("而「經濟貿易」維持不變或下降，即完美證實習近平外交的『泛政治化』與『論述霸權擴張』！")

# ==========================================
# 6. 軌道二：TF-IDF 潛在議題挖掘
# ==========================================
print("\n" + "=" * 70)
print("🔍 軌道二：潛在議題挖掘引擎（TF-IDF 演算法）")
print("=" * 70)

vectorizer = TfidfVectorizer(max_df=0.8, min_df=2)
tfidf_matrix = vectorizer.fit_transform(df['Processed_Text'])
feature_names = vectorizer.get_feature_names_out()

def get_top_keywords(period_name: str, top_n: int = 15) -> list[tuple[str, float]]:
    idx = df[df['Period'] == period_name].index
    if len(idx) == 0:
        return []
    period_tfidf = np.array(tfidf_matrix[idx].mean(axis=0)).flatten()
    top_indices = period_tfidf.argsort()[-top_n:][::-1]
    return [(feature_names[i], round(period_tfidf[i], 4)) for i in top_indices]

if 'Period' in df.columns:
    print("\n📉 [P1 胡溫時期] 獨特高頻特徵詞:")
    for word, score in get_top_keywords('P1', 10):
        print(f"  - {word:<15} (TF-IDF: {score})")

    print("\n📈 [P2 習近平時期] 獨特高頻特徵詞:")
    for word, score in get_top_keywords('P2', 10):
        print(f"  - {word:<15} (TF-IDF: {score})")

# ==========================================
# 7. 匯出結果
# ==========================================
output_file = "Lexicon_7Dimensions_Results_v2.xlsx"
df.drop(columns=['Lexicon_Counts', 'Processed_Text']).to_excel(output_file, index=False)

print("\n" + "=" * 70)
print(f"✅ 分析完成！所有跨期標準化詞頻數據已匯出至：{output_file}")
print("=" * 70)
