import streamlit as st
import os
import psycopg2
import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- 💡 NUEVAS IMPORTACIONES DE ML ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Comprobación de NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

stop_words = set(stopwords.words('spanish'))


def limpiar_texto(texto):
    texto = str(texto).lower()
    tokens = word_tokenize(texto)
    tokens_filtrados = [
        t for t in tokens if t.isalnum() and t not in stop_words]
    return ' '.join(tokens_filtrados)


# --- 💡 MEJORA DE BACKEND (Carga de Modelo y Vectores) ---
@st.cache_resource
def cargar_conocimiento_y_modelo():
    faq_data = []
    question_vectors = []
    
    try:
        # 1. Cargar el Modelo de ML
        # (Esto puede tardar la primera vez que se descarga)
        print("Cargando modelo de lenguaje...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("Modelo cargado.")

        # 2. Conectar a la BD (igual que antes)
        load_dotenv() 
        DB_URL = os.getenv('DATABASE_URL')
        if not DB_URL:
            if 'DATABASE_URL' in st.secrets:
                DB_URL = st.secrets['DATABASE_URL']
        
        if not DB_URL:
            st.error("Error: No se pudo encontrar la variable DATABASE_URL.")
            return None, None

        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT pregunta, palabras_clave, respuesta FROM chatbot_conocimiento")
        data_db = cursor.fetchall()
        conn.close()

        # 3. Procesar y "Vectorizar" las preguntas
        print("Vectorizando preguntas de la BD...")
        preguntas_limpias = []
        for row in data_db:
            pregunta_original = row[0]
            pregunta_limpia = limpiar_texto(pregunta_original)
            preguntas_limpias.append(pregunta_limpia)
            
            faq_data.append({
                'pregunta': pregunta_original,
                'palabras_clave': [p.strip().lower() for p in (row[1].split(';') if row[1] else [])],
                'respuesta': row[2]
            })

        # Convertimos todas las preguntas limpias en vectores de una sola vez
        question_vectors = model.encode(preguntas_limpias)
        print(f"Conocimiento cargado y vectorizado: {len(faq_data)} preguntas.")
        
        # Devolvemos el modelo, los datos y los vectores
        return model, faq_data, question_vectors

    except Exception as e:
        print(f"Error fatal al cargar el conocimiento: {e}")
        st.error(f"Error de conexión o carga de modelo: {e}")
        return None, None, None


# --- 💡 MEJORA DE LÓGICA (Búsqueda Semántica) ---
def responder(pregunta_usuario, model, faq_data, question_vectors):
    
    # Combinamos la búsqueda por palabra clave (rápida)
    # y la búsqueda por ML (inteligente)
    
    texto_filtrado = limpiar_texto(pregunta_usuario)

    if not texto_filtrado:
        return "Disculpa, no detecté ninguna palabra clave."

    # 1️⃣ Búsqueda por palabra clave (igual que antes, para coincidencias perfectas)
    for item in faq_data:
        for palabra in item['palabras_clave']:
            if palabra in texto_filtrado:
                return item['respuesta']

    # 2️⃣ Búsqueda por ML (Similitud Semántica)
    if model:
        # Convertir la pregunta del usuario en un vector
        user_vector = model.encode([texto_filtrado])
        
        # Calcular la similitud del coseno entre el vector del usuario
        # y TODOS los vectores de la base de datos
        similarities = cosine_similarity(user_vector, question_vectors)
        
        # Encontrar la pregunta más similar
        best_match_index = np.argmax(similarities)
        best_score = similarities[0][best_match_index]
        
        # 3️⃣ Devolución con umbral
        # (Usamos un umbral más bajo porque es semántico)
        if best_score >= 0.65: 
            return faq_data[best_match_index]['respuesta']
        else:
            return "Lo siento, no estoy seguro de entender tu pregunta. 😅 ¿Podrías reformularla?"
    else:
        return "Error: El modelo de IA no está cargado."


# --- Interfaz Gráfica (Con cambios menores) ---

st.title("🤖 Chatbot de Repuestos Verese Sac (v2.0 con IA)")
st.caption("Tu asistente virtual para consultas sobre repuestos, horarios y servicios.")

# 1. Cargar el modelo Y los datos
model, faq_data, question_vectors = cargar_conocimiento_y_modelo()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu consulta aquí..."):

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Comprobar que todo se haya cargado
    if not faq_data or model is None:
        response = "Error: No pude cargar mi base de conocimiento. El bot no está operativo."
    else:
        # 3. Pasar el modelo y los vectores a la función de respuesta
        response = responder(prompt, model, faq_data, question_vectors)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
