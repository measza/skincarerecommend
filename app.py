import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Styling background ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffe6f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("cosmetic_p.csv")
    df.columns = df.columns.str.lower().str.strip()  # Normalize column names
    df["ingredients"] = df["ingredients"].str.lower()
    return df

data = load_data()

# --- Aturan sistem pakar ---
rules = {
    "Mencerahkan Wajah": ["niacinamide", "vitamin c", "alpha arbutin"],
    "Mengurangi Jerawat": ["salicylic acid", "benzoyl peroxide", "tea tree", "azelaic acid", "sulfur"],
    "Anti Aging": ["retinol", "peptide", "coenzyme q10", "bakuchiol"],
    "Melembapkan Kulit": ["hyaluronic acid", "ceramide", "glycerin", "squalane"],
}

# --- UI ---
st.title("🔍 Rekomendasi Produk Skincare")

# Input pengguna
kebutuhan = st.selectbox("🎯 Apa tujuan skincare Anda?", list(rules.keys()))
produk_type = st.selectbox("🧴 Pilih jenis produk", data["label"].unique())

jenis_kulit = st.multiselect(
    "💧 Jenis Kulit Anda (opsional):",
    ["combination", "dry", "normal", "oily", "sensitive"]
)

similarity_weight = st.slider("⚖️ Prioritaskan bahan atau rating", 0.0, 1.0, 0.7, help="0 = Hanya rating, 1 = Hanya bahan aktif")
rating_weight = 1.0 - similarity_weight

# --- Rekomendasi ---
if st.button("✨ Tampilkan Rekomendasi"):
    keywords = rules[kebutuhan]
    produk_subset = data[data["label"].str.lower() == produk_type.lower()]

    def contains_ingredients(ingredients_text):
        return any(ingredient in ingredients_text for ingredient in keywords)

    rule_filtered = produk_subset[produk_subset["ingredients"].apply(contains_ingredients)]

    if rule_filtered.empty:
        st.warning("⚠️ Tidak ada produk yang mengandung keyword langsung. Menampilkan produk serupa...")
        rule_filtered = produk_subset.copy()

    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(rule_filtered["ingredients"])
    query_vec = vectorizer.transform([" ".join(keywords)])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    rule_filtered["similarity"] = similarities

    # Filter kulit
    for skin_type in jenis_kulit:
        rule_filtered = rule_filtered[rule_filtered[skin_type] == 1]

    # Skor gabungan
    rule_filtered["combined_score"] = (
        similarity_weight * rule_filtered["similarity"] +
        rating_weight * (rule_filtered["rank"] / 5)
    )

    recommended = rule_filtered.sort_values(by="combined_score", ascending=False)

    # --- Tampilkan hasil ---
    if not recommended.empty:
        st.success(f"👍 Ditemukan {len(recommended)} produk yang cocok:")
        for _, row in recommended.iterrows():
            st.markdown(f"### 🧴 {row['name']}")
            st.markdown(f"**Brand:** {row['brand']} &nbsp;&nbsp; | &nbsp;&nbsp; 💰 **Harga:** ${row['price']} &nbsp;&nbsp; | ⭐ **Rating:** {row['rank']}")
            st.markdown(f"**Ingredients:** {row['ingredients'][:250]}...")
            st.markdown("---")
    else:
        st.warning("❌ Tidak ditemukan produk yang sesuai dengan kriteria Anda.")
