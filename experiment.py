import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sentence_transformers import SentenceTransformer

# 简单停用词
stop_words = {"的","了","是","就","也","都","而","和","在","我","有"}

def preprocess(text):
    """文本预处理：分词+去停用词"""
    words = jieba.lcut(text)
    res = [w for w in words if w not in stop_words and len(w.strip())>0]
    return res

# 1.Jaccard相似度
def jaccard_sim(set_a,set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter/union if union>0 else 0

# 2.编辑距离相似度
def edit_distance(a,b):
    la,lb = len(a),len(b)
    dp = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1): dp[i][0]=i
    for j in range(lb+1): dp[0][j]=j
    for i in range(1,la+1):
        for j in range(1,lb+1):
            if a[i-1]==b[j-1]:
                dp[i][j]=dp[i-1][j-1]
            else:
                dp[i][j]=1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])
    dist = dp[la][lb]
    sim = 1 - dist/max(la,lb) if max(la,lb)>0 else 0
    return sim

# 3.TF‑IDF余弦相似度
tfidf = TfidfVectorizer(tokenizer=lambda x:x,token_pattern=None)

# 4.Sentence‑BERT语义相似度
model = SentenceTransformer('all‑MiniLM‑L6‑v2')

if __name__=="__main__":
    print("NLP多相似度测度‑短文本情感分类实验")
    # 此处可以接入数据集，计算特征，训练逻辑回归，输出指标
