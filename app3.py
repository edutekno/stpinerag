import streamlit as st
from astrapy import DataAPIClient
from openai import OpenAI
from googletrans import Translator
import asyncio
import requests
import json
import struct

# Konfigurasi DataStax AstraDB
ASTRA_TOKEN = st.secrets["ASTRA_TOKEN"]
ASTRA_API_ENDPOINT = st.secrets["ASTRA_API_ENDPOINT"]
KEYSPACE = "default_keyspace"
COLLECTION_NAME = "book_embeddings"

# Konfigurasi OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Inisialisasi DataStax AstraDB
astra_client = DataAPIClient(ASTRA_TOKEN)
db = astra_client.get_database_by_api_endpoint(ASTRA_API_ENDPOINT, keyspace=KEYSPACE)
collection = db.get_collection(COLLECTION_NAME)

# Konfigurasi OpenRouter
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# Fungsi konversi vektor ke biner
def vector_to_binary(vector):
    return struct.pack(f"<{len(vector)}f", *vector)

# Fungsi terjemahan
async def async_translate_to_english(text):
    translator = Translator()
    translation = await translator.translate(text, src='id', dest='en')
    return translation.text

def translate_to_english(text):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(async_translate_to_english(text))
    loop.close()
    return result

# Fungsi embedding
def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding[:1000]  # Truncate ke 1000 dimensi

# Fungsi pencarian di AstraDB
def search_in_astra(query_embedding, top_k=5):
    query_binary = vector_to_binary(query_embedding)
    results = collection.find(
        filter={},
        sort={"embedding": {"$vector": query_binary}},
        limit=top_k,
        projection={"text": 1}
    )
    return [doc["text"] for doc in results["data"]["documents"]]

# Fungsi respons AI
def generate_response(prompt):
    data = {
        "model": "google/gemma-3-12b-it:free",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(OPENROUTER_API_URL, headers=HEADERS, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# UI Streamlit
st.set_page_config(
    page_title="QA Solve for Happy",
    page_icon="📖",
    layout="wide",
)

with st.sidebar:
    st.write("QA Book Gemma 3")
    st.image("solve.jpeg", caption="Mo Gawdat")

st.title("QA Buku Solve for Happy")
st.write("(Mo Gawdat)")

query = st.text_input("Masukkan pertanyaan Anda:")
if st.button("Cari"):
    if query:
        with st.spinner("Sedang memproses..."):
            try:
                # Terjemahkan pertanyaan
                query_en = translate_to_english(query)
                
                # Dapatkan embedding query
                query_embedding = get_embedding(query_en)
                
                # Cari di AstraDB
                contexts = search_in_astra(query_embedding, top_k=5)
                context_str = "\n".join(contexts)
                
                # Buat prompt
                prompt = f"""
                Berdasarkan konteks berikut (bahasa Inggris):
                {context_str}

                Pertanyaan (bahasa Inggris): {query_en}

                Jawablah pertanyaan di atas dalam bahasa Indonesia yang jelas dan lengkap.
                """
                
                # Dapatkan respons AI
                response = generate_response(prompt)
                
                # Tampilkan hasil
                st.write("Konteks yang relevan:")
                st.write(context_str)
                st.write("Jawaban:")
                st.write(response)

            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
    else:
        st.warning("Masukkan pertanyaan terlebih dahulu.")
