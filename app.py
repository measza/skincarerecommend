import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Styling background ---
st.markdown("""
    <style>
    .stApp {
        background-color: #ffe6f0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load model dan data ---
vectorizer = joblib.load("tfidf_vectorizer.pkl")
data = pd.read_csv("cosmetic_p.csv")
data.columns = data.columns.str.lower().str.strip()
data["ingredients"] = data["ingredients"].str.lower()

# --- Aturan sistem pakar ---
rules = {
    "Mencerahkan Wajah": ["niacinamide", "vitamin c", "alpha arbutin"],
    "Mengurangi Jerawat": ["salicylic acid", "benzoyl peroxide", "tea tree", "azelaic acid", "sulfur"],
    "Anti Aging": ["retinol", "peptide", "coenzyme q10", "bakuchiol"],
    "Melembapkan Kulit": ["hyaluronic acid", "ceramide", "glycerin", "squalane"],
}

# --- UI ---
st.title("🔍 Rekomendasi Produk Skincare")

kebutuhan = st.selectbox("🎯 Apa tujuan skincare Anda?", list(rules.keys()))
produk_type = st.selectbox("🧴 Pilih jenis produk", data["label"].unique())

jenis_kulit = st.multiselect(
    "💧 Jenis Kulit Anda (opsional):",
    ["combination", "dry", "normal", "oily", "sensitive"]
)

similarity_weight = st.slider("⚖️ Prioritaskan bahan atau rating", 0.0, 1.0, 0.7, help="0 = Hanya rating, 1 = Hanya bahan aktif")
rating_weight = 1.0 - similarity_weight

# --- Tampilkan rekomendasi ---
if st.button("✨ Tampilkan Rekomendasi"):
    keywords = rules[kebutuhan]
    produk_subset = data[data["label"].str.lower() == produk_type.lower()].copy()

    # TF-IDF
    tfidf_matrix = vectorizer.fit_transform(produk_subset["ingredients"])
    query_vec = vectorizer.transform([" ".join(keywords)])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    produk_subset["similarity"] = similarities

    # Tandai apakah produk mengandung keyword
    def contains_keywords(text):
        return any(kw in text for kw in keywords)

    produk_subset["has_keywords"] = produk_subset["ingredients"].apply(contains_keywords)

    # Filter jenis kulit (jika diisi)
    for skin in jenis_kulit:
        if skin in produk_subset.columns:
            produk_subset = produk_subset[produk_subset[skin] == 1]

    # Hitung skor gabungan
    produk_subset["combined_score"] = (
        similarity_weight * produk_subset["similarity"] +
        rating_weight * (produk_subset["rank"] / 5)
    )

    # Tambah skor jika mengandung keyword
    produk_subset.loc[produk_subset["has_keywords"], "combined_score"] += 0.05

    # Urutkan hasil
    recommended = produk_subset.sort_values(by="combined_score", ascending=False)

    # --- Tampilkan hasil ---
    if not recommended.empty:
        st.success(f"👍 Ditemukan {len(recommended)} produk yang cocok:")
        for _, row in recommended.iterrows():
            st.markdown(f"### 🧴 {row['name']}")
            st.markdown(f"**Brand:** {row['brand']} &nbsp;&nbsp; | &nbsp;&nbsp; 💰 **Harga:** ${row['price']} &nbsp;&nbsp; | ⭐ **Rating:** {row['rank']}")
            st.markdown(f"**Ingredients:** {row['ingredients'][:300]}...")
            st.markdown("---")
    else:
        st.warning("❌ Tidak ditemukan produk yang sesuai dengan kriteria Anda.")
