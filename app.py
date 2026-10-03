import os
import re
import json
import time
import base64
import zipfile
import tempfile
import sqlite3
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st
import xgboost as xgb
from PIL import Image
from bs4 import BeautifulSoup
from textblob import TextBlob

# ───────────────────────── 基礎設定 ─────────────────────────
st.set_page_config(page_title="EcoFit AI", layout="wide")

BASE = Path(__file__).parent
DB_PATH = BASE / "wardrobe.db"
CHART_DIR = BASE / "saved_charts"
CHART_DIR.mkdir(exist_ok=True)
GEMINI_MODEL = "gemini-2.5-flash"


def get_api_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")


API_KEY = get_api_key()


# ───────────────────────── 資料庫 ─────────────────────────
def db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS my_wardrobe (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand TEXT, product_name TEXT, size TEXT,
                ai_summary TEXT, user_feedback TEXT,
                size_chart_path TEXT, url TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
        cols = [r[1] for r in conn.execute("PRAGMA table_info(my_wardrobe)")]
        if "url" not in cols:  # 舊版資料庫升級
            conn.execute("ALTER TABLE my_wardrobe ADD COLUMN url TEXT")


init_db()


# ───────────────────────── Gemini ─────────────────────────
def img_to_b64(file) -> str:
    img = Image.open(file).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def gemini(parts: list, model: str = GEMINI_MODEL, tools=None) -> str:
    if not API_KEY:
        return "⚠️ 尚未設定 GEMINI_API_KEY"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {"contents": [{"parts": parts}]}
    if tools:
        body["tools"] = tools
    try:
        r = requests.post(
            url,
            headers={"Content-Type": "application/json", "x-goog-api-key": API_KEY},
            json=body,
            timeout=120,
        )
        data = r.json()
        if r.status_code != 200:
            return f"API 錯誤：{data.get('error', {}).get('message', '未知錯誤')}"
        ps = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in ps)
    except Exception as e:
        return f"連線異常：{e}"


# ───────────────────────── 蝦皮爬蟲 ─────────────────────────
class FetchError(Exception):
    pass


class ShopeeError(FetchError):
    pass


class PageError(FetchError):
    pass


UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def parse_shopee_url(url: str):
    """從蝦皮網址取出 (shopid, itemid)，支援短網址。"""
    url = url.strip()
    if re.search(r"(shp\.ee|s\.shopee\.tw)", url):
        try:
            url = requests.get(url, headers={"User-Agent": UA},
                               allow_redirects=True, timeout=10).url
        except Exception:
            raise ShopeeError("短網址展開失敗，請改貼完整商品網址")
    m = re.search(r"i\.(\d+)\.(\d+)", url) or re.search(r"/product/(\d+)/(\d+)", url)
    if not m:
        raise ShopeeError("無法從網址解析商品 ID，請確認是蝦皮商品頁網址")
    return int(m.group(1)), int(m.group(2))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_shopee_reviews(shopid: int, itemid: int, max_reviews: int = 200):
    """抓取商品評價。蝦皮有反爬蟲機制，失敗時會丟出 ShopeeError。"""
    headers = {
        "User-Agent": UA,
        "Referer": f"https://shopee.tw/product/{shopid}/{itemid}",
        "Accept": "application/json",
        "x-api-source": "pc",
        "x-requested-with": "XMLHttpRequest",
        "af-ac-enc-dat": "null",
    }
    out, offset = [], 0
    while len(out) < max_reviews:
        params = {
            "itemid": itemid, "shopid": shopid, "filter": 1,  # 1 = 有留言的評價
            "flag": 1, "limit": 50, "offset": offset, "type": 0,
        }
        r = requests.get("https://shopee.tw/api/v4/item/get_ratings",
                         params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            raise ShopeeError(f"蝦皮拒絕請求（HTTP {r.status_code}），可能被反爬蟲擋下")
        try:
            data = r.json()
        except Exception:
            raise ShopeeError("蝦皮回傳非 JSON（可能是驗證頁面）")
        if data.get("error"):
            raise ShopeeError(f"蝦皮回傳錯誤碼 {data['error']}（多半是反爬蟲）")
        ratings = (data.get("data") or {}).get("ratings") or []
        if not ratings:
            break
        for x in ratings:
            comment = (x.get("comment") or "").strip()
            if not comment:
                continue
            items = x.get("product_items") or [{}]
            out.append({
                "star": x.get("rating_star"),
                "comment": comment,
                "variant": items[0].get("model_name", ""),
                "time": x.get("ctime"),
            })
        offset += 50
        time.sleep(0.8)  # 放慢速度，避免被封
    if not out:
        raise ShopeeError("沒有抓到任何文字評價")
    return out[:max_reviews]


# ───────────────────────── 通用網頁（非蝦皮）─────────────────────────
def _walk_jsonld(node, found):
    if isinstance(node, dict):
        if node.get("reviewBody"):
            rating = node.get("reviewRating")
            star = rating.get("ratingValue") if isinstance(rating, dict) else None
            found.append({"star": star, "comment": str(node["reviewBody"]).strip(),
                          "variant": "", "time": node.get("datePublished")})
        for v in node.values():
            _walk_jsonld(v, found)
    elif isinstance(node, list):
        for v in node:
            _walk_jsonld(v, found)


def extract_jsonld_reviews(soup):
    found = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            _walk_jsonld(json.loads(tag.string or ""), found)
        except Exception:
            continue
    return found


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_generic_page(url: str):
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8"},
                         timeout=15)
    except Exception as e:
        raise PageError(f"無法連線到該網站：{e}")
    if r.status_code != 200:
        raise PageError(f"網站拒絕請求（HTTP {r.status_code}）")
    if not r.encoding or r.encoding.lower() == "iso-8859-1":
        r.encoding = r.apparent_encoding
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    reviews = extract_jsonld_reviews(soup)  # 要在移除 script 之前
    for t in soup(["script", "style", "noscript", "header", "footer", "nav", "svg"]):
        t.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    return {"title": title, "reviews": reviews, "text": text}


def fetch_via_gemini(url: str) -> str:
    """頁面內容靠 JS 動態載入時，改請 Gemini 的 URL 讀取功能代抓。"""
    out = gemini([{"text": (
        f"請讀取這個商品頁：{url}\n"
        "列出頁面上所有買家評論原文（特別是身高體重、尺寸、品質相關的描述），每則一行並編號；"
        "另外列出頁面上的尺寸表與材質說明。不要編造內容；"
        "若無法讀取頁面或找不到評論，只回覆「無法讀取」。")}],
        tools=[{"url_context": {}}])
    if out.startswith(("API", "⚠️", "連線")) or "無法讀取" in out[:20] or len(out) < 50:
        raise PageError("此網站的評論無法自動讀取（可能需登入或由 JS 動態載入）")
    return out


def load_reviews_from_url(url: str):
    """依網址類型取得評論，回傳 (給 AI 的文字, 顯示給使用者的說明)。"""
    url = url.strip()
    if re.search(r"shopee\.|shp\.ee", url):
        sid, iid = parse_shopee_url(url)
        allr = fetch_shopee_reviews(sid, iid)
        picked = select_relevant(allr)
        return reviews_to_text(picked), f"蝦皮：共抓到 {len(allr)} 則文字評論，挑選 {len(picked)} 則送交 AI"

    if not url.startswith("http"):
        raise PageError("網址需以 http:// 或 https:// 開頭")
    page = fetch_generic_page(url)
    chunks, info = [], []
    if page["reviews"]:
        picked = select_relevant(page["reviews"]) or page["reviews"][:60]
        chunks.append("【結構化評論】\n" + reviews_to_text(picked))
        info.append(f"找到 {len(page['reviews'])} 則內建評論")
    if len(page["text"]) >= 800:
        chunks.append("【商品頁文字（可能含尺寸表、說明、評論）】\n" + page["text"][:12000])
        info.append("已讀取頁面文字")
    if not page["reviews"] and len(page["text"]) < 800:
        chunks.append("【Gemini 讀取結果】\n" + fetch_via_gemini(url))
        info.append("頁面為動態載入，改由 Gemini 讀取")
    if not chunks:
        raise PageError("沒有讀到任何可用內容")
    return "\n\n".join(chunks), f"{page['title'][:40]}：" + "、".join(info)


FIT_KEYWORDS = ["身高", "體重", "公斤", "kg", "cm", "公分", "偏小", "偏大", "好穿", "合身",
                "寬鬆", "緊", "鬆", "短", "長", "尺寸", "size", "S號", "M號", "L號", "XL",
                "縮水", "毛球", "起毛", "色差", "線頭", "薄", "透", "硬", "扎", "悶", "味"]


def select_relevant(reviews, n=60):
    """挑出與版型/品質最相關的評論，避免把無用的「很好」「已收到」丟給 AI。"""
    def score(r):
        c = r["comment"].lower()
        s = sum(2 for k in FIT_KEYWORDS if k.lower() in c)
        s += 3 if re.search(r"\d{3}\s*(cm|公分)|\d{2,3}\s*(kg|公斤)", c) else 0
        s += min(len(c) // 20, 3)
        s += 1 if r["star"] and r["star"] <= 3 else 0  # 負評優先保留
        return s
    ranked = sorted(reviews, key=score, reverse=True)
    return [r for r in ranked if len(r["comment"]) >= 6][:n]


def reviews_to_text(reviews) -> str:
    lines = []
    for i, r in enumerate(reviews, 1):
        v = f"｜規格:{r['variant']}" if r["variant"] else ""
        lines.append(f"[{i}] {r['star']}★{v}｜{r['comment']}")
    return "\n".join(lines)


# ───────────────────────── AI 分析 ─────────────────────────
def build_prompt(stats, habit, review_text):
    u = (f"身高{stats['h']}cm、體重{stats['weight']}kg、"
         f"三圍{stats['chest']}/{stats['waist']}/{stats['hip']}cm"
         + ("（部分數值為預設平均值，僅供參考）" if stats["is_uncertain"] else ""))
    return f"""你是嚴謹的服飾品管與版型顧問。以下是某商品的買家評論或商品頁內容，請根據評論（以及附圖的尺寸表，若有）判斷是否適合該使用者。

【使用者】{u}
【使用者過往偏好】{habit}

【買家評論】
{review_text if review_text else "（無文字評論，請依附圖判斷）"}

【規則】
- 只能根據提供的評論與圖片作答，禁止編造；評論沒提到的就寫「評論未提及」。
- 引用評論時請標註編號，例如 [3]、[12]。
- 優先參考身材與使用者接近的買家（身高體重相近）的回饋。
- 不要略過任何負面評價，包含細小問題。
- 使用繁體中文，關鍵字加粗。

【輸出格式】
1. 💎 **尺寸建議**：最推薦尺寸與理由（附身材相近買家的評論編號）
2. ⚠️ **風險警告**
   - 🚨 即時瑕疵（收到貨就有的問題）
   - 🐢 長期耐用度（洗後毛球、縮水、變形等，寫出次數與狀況）
   - 📏 版型偏差（偏大/偏小/長度）
3. ✅ **優點**
4. 💡 **穿搭建議**：針對此使用者身材
5. 📊 **結論**：適合度 1~5 星 + 一句話總結
"""


def analyze(stats, habit, review_text, review_imgs, chart):
    parts = [{"text": build_prompt(stats, habit, review_text)}]
    for f in (review_imgs or [])[:6]:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_b64(f)}})
    if chart:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_to_b64(chart)}})
    return gemini(parts)


@st.cache_data(ttl=600, show_spinner=False)
def summarize_habits(_marker: float):
    with db() as conn:
        df = pd.read_sql_query(
            "SELECT size, user_feedback FROM my_wardrobe "
            "WHERE user_feedback IS NOT NULL AND user_feedback != ''", conn)
    if df.empty:
        return "尚無足夠評價紀錄。"
    fb = "\n".join(f"- {r['size']}: {r['user_feedback']}" for _, r in df.iterrows())
    return gemini([{"text": f"請根據以下購買回饋，用 3 句話總結此人的穿衣偏好（寬鬆/修身等）：\n{fb}"}])


# ───────────────────────── XGBoost ─────────────────────────
@st.cache_resource(show_spinner=False)
def load_xgb_model():
    """載入模型；若只有 zip 壓縮檔，會自動解壓縮到暫存資料夾。"""
    ep = BASE / "label_encoder.pkl"
    mp = BASE / "size_model_2.json"
    zp = BASE / "size_model_2.json.zip"
    if not mp.exists() and zp.exists():
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist()
                     if n.endswith(".json") and "__MACOSX" not in n and not Path(n).name.startswith("._")]
            if not names:
                return None
            mp = Path(tempfile.gettempdir()) / "size_model_2.json"
            mp.write_bytes(z.read(names[0]))
    if not mp.exists() or not ep.exists():
        return None
    bst = xgb.XGBClassifier()
    bst.load_model(str(mp))
    return bst, joblib.load(ep)


def predict_size_xgb(stats, text=""):
    try:
        loaded = load_xgb_model()
        if loaded is None:
            return "⚙️ 找不到模型檔（size_model_2.json(.zip) / label_encoder.pkl）"
        bst, le = loaded
        h, b, w, hip = stats["h"], stats["chest"], stats["waist"], stats["hip"]
        sent = TextBlob(text).sentiment.polarity if text else 0
        best = None
        for label, code in [("S", 0), ("M", 1), ("L", 2)]:
            row = pd.DataFrame([[code, 1, h, b, w, hip, b / (w + 0.1), hip / (w + 0.1), 3, sent]],
                               columns=["size_code", "category_code", "height_cm", "bust_size",
                                        "waist_size", "hips_size", "bust_to_waist",
                                        "hips_to_waist", "length_score", "sentiment_score"])
            if le.inverse_transform([bst.predict(row)[0]])[0] == "fit":
                best = label
        return f"🎯 最佳推薦：{best} 號" if best else "找不到完全合適的尺寸"
    except Exception as e:
        return f"預測錯誤：{e}"


# ───────────────────────── 側邊欄 ─────────────────────────
with db() as _c:
    _marker = _c.execute("SELECT COALESCE(MAX(id),0) + COUNT(user_feedback) FROM my_wardrobe").fetchone()[0]
habit = summarize_habits(_marker)

st.sidebar.header("🔍 個人版型總結")
st.sidebar.info(habit)

with st.sidebar:
    st.header("👤 身材數據")
    st.caption("不確定的數值請勾選，系統會用平均體型代替")
    h = st.number_input("身高 (cm)", 100, 250, 160)

    def field(label, lo, hi, default):
        unsure = st.checkbox(f"不確定{label}", key=f"u_{label}")
        v = st.number_input(f"{label}", lo, hi, default, disabled=unsure, key=f"v_{label}")
        return default if unsure else v, unsure

    weight, u1 = field("體重(kg)", 30, 200, 55)
    chest, u2 = field("胸圍(cm)", 50, 150, 85)
    waist, u3 = field("腰圍(cm)", 40, 150, 66)
    hip, u4 = field("臀圍(cm)", 50, 150, 90)
    stats = dict(h=h, weight=weight, chest=chest, waist=waist, hip=hip,
                 is_uncertain=any([u1, u2, u3, u4]))

# ───────────────────────── 主畫面 ─────────────────────────
st.title("👗 EcoFit AI 雙引擎管家")
tab1, tab2 = st.tabs(["🚀 深度分析", "🗄️ 智慧衣櫃"])

with tab1:
    mode = st.radio("評論來源", ["🔗 商品連結（蝦皮／其他網站）", "📋 貼上評論文字", "🖼️ 上傳截圖"],
                    horizontal=True)
    shop_url, pasted, shots = "", "", None
    if mode.startswith("🔗"):
        shop_url = st.text_input("貼上商品網址",
                                 placeholder="https://shopee.tw/… 或任何購物網站的商品頁")
    elif mode.startswith("📋"):
        pasted = st.text_area("每則評論一行", height=200)
    else:
        shots = st.file_uploader("評論截圖", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    chart = st.file_uploader("尺寸表截圖（選填）", type=["png", "jpg", "jpeg"])

    if st.button("🚀 開始分析", type="primary"):
        review_text, n_total = "", 0
        try:
            if shop_url:
                with st.spinner("正在抓取評論…"):
                    review_text, info = load_reviews_from_url(shop_url)
                st.caption(f"📥 {info}")
            elif pasted.strip():
                review_text = "\n".join(f"[{i}] {l}" for i, l in
                                        enumerate([x for x in pasted.splitlines() if x.strip()], 1))
            elif not shots:
                st.error("請提供商品連結、評論文字或截圖")
                st.stop()
        except FetchError as e:
            st.error(f"❌ {e}\n\n部分網站會封鎖雲端主機的爬蟲請求。請改用「貼上評論文字」或「上傳截圖」模式。")
            st.stop()
        except Exception as e:
            st.error(f"抓取失敗：{e}")
            st.stop()

        with st.spinner("AI 分析中…"):
            report = analyze(stats, habit, review_text, shots, chart)
            xgb_res = predict_size_xgb(stats, report)
        st.session_state.res = dict(report=report, xgb=xgb_res, url=shop_url)

    if "res" in st.session_state:
        r = st.session_state.res
        st.divider()
        a, b = st.columns([2, 1])
        a.markdown(f"### 🤖 AI 報告\n{r['report']}")
        b.info(f"### 📊 XGBoost 預測\n{r['xgb']}")

        with st.form("buy_form"):
            st.subheader("🛒 記錄購買商品")
            brand = st.text_input("品牌")
            name = st.text_input("品名")
            sz = st.selectbox("購買尺寸", ["S", "M", "L", "XL", "F"])
            if st.form_submit_button("確認購買並存入衣櫃"):
                path = ""
                if chart:
                    safe = re.sub(r"[^\w\-]", "_", brand or "item")
                    path = str(CHART_DIR / f"{int(time.time())}_{safe}_{sz}.jpg")
                    Image.open(chart).convert("RGB").save(path, "JPEG")
                with db() as conn:
                    conn.execute(
                        "INSERT INTO my_wardrobe (brand, product_name, size, ai_summary, size_chart_path, url) "
                        "VALUES (?,?,?,?,?,?)",
                        (brand, name, sz, r["report"][:300], path, r.get("url", "")))
                st.success(f"✅ {brand} - {name} 已記錄")

with tab2:
    st.subheader("📋 智慧衣櫃售後回饋")
    with db() as conn:
        df = pd.read_sql_query("SELECT * FROM my_wardrobe ORDER BY timestamp DESC", conn)
    if df.empty:
        st.caption("衣櫃還是空的")
    for _, row in df.iterrows():
        with st.expander(f"📦 {row['brand']} | {row['product_name']}（尺寸：{row['size']}）"):
            ci, cm = st.columns([2, 1])
            with ci:
                st.write(f"🕒 **購買時間**：{row['timestamp']}")
                if row["url"]:
                    st.write(f"🔗 [商品連結]({row['url']})")
                summary = row["ai_summary"] or ""
                if any(k in summary for k in ["⚠️", "縮水", "毛球"]):
                    st.error(f"**先前注意**：{summary}")
                else:
                    st.write(f"🤖 **AI 分析回顧**：{summary}")
                fb = st.text_area("心得回饋", value=row["user_feedback"] or "", key=f"f_{row['id']}")
                if st.button("更新回饋", key=f"b_{row['id']}"):
                    with db() as conn:
                        conn.execute("UPDATE my_wardrobe SET user_feedback=? WHERE id=?", (fb, int(row["id"])))
                    st.cache_data.clear()
                    st.rerun()
            with cm:
                p = row["size_chart_path"]
                if p and os.path.exists(p):
                    st.image(p, caption="當時參考的尺寸表")
                else:
                    st.caption("無尺寸表圖片")
