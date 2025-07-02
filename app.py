import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load model dan data
vectorizer = joblib.load("tfidf_vectorizer.pkl")
data = pd.read_csv("cosmetic_p.csv")
data["ingredients"] = data["ingredients"].str.lower()

# Aturan rule-based
rules = {
    "Mencerahkan Wajah": ["niacinamide", "vitamin c", "alpha arbutin"],
    "Mengurangi Jerawat": ["salicylic acid", "benzoyl peroxide", "tea tree", "azelaic acid", "sulfur"],
    "Anti Aging": ["retinol", "peptide", "coenzyme q10", "bakuchiol"],
    "Melembapkan Kulit": ["hyaluronic acid", "ceramide", "glycerin", "squalane"],
}

# Konfigurasi Streamlit
st.set_page_config(page_title="Rekomendasi Skincare", page_icon="💄", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background-color: #ffe6ee;
    }
    .product-card {
        background-color: #fff;
        border-radius: 1rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Rekomendasi Produk Skincare")

# Input form
kebutuhan = st.selectbox("🎯 Apa tujuan skincare Anda?", list(rules.keys()))
produk_type = st.selectbox("🍶 Pilih jenis produk", data["label"].unique())
jenis_kulit = st.multiselect("💧 Jenis Kulit Anda (opsional):", 
                              ["combination", "dry", "normal", "oily", "sensitive"])

similarity_weight = st.slider("🧬 Prioritaskan bahan atau rating", 0.0, 1.0, 0.7)
rating_weight = 1.0 - similarity_weight

if st.button("✅ Tampilkan Rekomendasi"):
    keywords = rules[kebutuhan]
    produk_subset = data[data["label"] == produk_type]

    def contains_ingredients(text):
        return any(ing in text for ing in keywords)

    rule_filtered = produk_subset[produk_subset["ingredients"].apply(contains_ingredients)]

    if rule_filtered.empty:
        st.warning("⚠️ Tidak ada produk dengan bahan tersebut. Menampilkan alternatif...")
        rule_filtered = produk_subset.copy()

    tfidf_matrix = vectorizer.fit_transform(rule_filtered["ingredients"])
    query_doc = " ".join(keywords)
    query_vec = vectorizer.transform([query_doc])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    rule_filtered["similarity"] = similarities

    # Filter jenis kulit
    for skin in jenis_kulit:
        if skin in rule_filtered.columns:
            rule_filtered = rule_filtered[rule_filtered[skin] == 1]

    rule_filtered["combined_score"] = (
        similarity_weight * rule_filtered["similarity"] +
        rating_weight * (rule_filtered["rank"] / 5)
    )

    recommended = rule_filtered.sort_values(by="combined_score", ascending=False)

    if not recommended.empty:
        st.success(f"✅ Ditemukan {len(recommended)} produk yang cocok:")
        for _, row in recommended.iterrows():
            st.markdown("### 🧴 " + row['name'])
            st.markdown(f"**Brand:** {row['brand']}  |  💰 **Harga:** ${row['price']}  |  ⭐ **Rating:** {row['rank']}")
            st.markdown(f"**Ingredients:** {row['ingredients'][:300]}...")
            st.markdown("---")
    else:
        st.warning("❌ Tidak ditemukan produk yang sesuai dengan kriteria.")
