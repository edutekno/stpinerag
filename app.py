import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
from googletrans import Translator
import asyncio

st.set_page_config(
        page_title="QA Solve for Happy",
        page_icon="open_book",
        layout="wide",
    )
with st.sidebar:
    #with st.echo():
    #st.write("QA Book")
    st.image("solve.jpeg", caption="Solve for Happy - Mo Gawdat")
# Inisialisasi Pinecone dan OpenAI
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
index_name = "mydb"  # Ganti dengan nama index Anda
index = pc.Index(index_name)

# Fungsi asynchronous untuk menerjemahkan teks ke bahasa Inggris
async def async_translate_to_english(text):
    translator = Translator()
    translation = await translator.translate(text, src='id', dest='en')
    return translation.text

# Fungsi synchronous untuk memanggil fungsi asynchronous
def translate_to_english(text):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(async_translate_to_english(text))
    loop.close()
    return result

# Fungsi untuk menghasilkan embedding dari teks
def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

# Fungsi untuk menghasilkan respons dari model AI
def generate_response(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Aplikasi Streamlit
st.title("QA Buku Solve for Happy")
st.write("(Mo Gawdat)")
# Input pengguna
query = st.text_input("Masukkan pertanyaan Anda:")
if st.button("Cari"): 
    if query:
        # Tambahkan tampilan loading
        with st.spinner("Sedang memproses pertanyaan Anda..."):
            try:
                # Terjemahkan pertanyaan ke bahasa Inggris
                query_en = translate_to_english(query)

                # Buat vektor dari query yang diterjemahkan
                query_vector = get_embedding(query_en)
                st.write(query_en)
                # Cari di PineconeDB
                results = index.query(
                    vector=query_vector,
                    top_k=3,
                    include_metadata=True,
                    namespace="my-works"  # Ganti namespace sesuai kebutuhan Anda
                )

                # Ambil teks dari metadata hasil pencarian
                contexts = [match['metadata']['text'] for match in results['matches'] if 'text' in match['metadata']]
                context_str = "\n".join(contexts)

                # Buat prompt untuk model AI
                prompt = f"""
                Berdasarkan informasi berikut (dalam bahasa Inggris):
                {context_str}

                Pertanyaan (dalam bahasa Inggris): {query_en}

                Tolong jawab pertanyaan di atas dalam bahasa Indonesia.
                """

                # Dapatkan respons dari model AI
                response = generate_response(prompt)

                # Tampilkan hasil
                st.write("Jawaban:")
                st.write(response)

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    else:
        st.warning("Masukkan pertanyaan terlebih dahulu.")
