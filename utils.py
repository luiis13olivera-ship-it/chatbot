import streamlit as st
import os
import psycopg2
import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- IMPORTACIONES DE ML ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Configuración de NLTK ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Descargando recursos de NLTK...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

stop_words = set(stopwords.words('spanish'))

# --- 1. Lógica de Limpieza ---
def limpiar_texto(texto):
    """Limpia y tokeniza el texto."""
    texto = str(texto).lower()
    tokens = word_tokenize(texto)
    tokens_filtrados = [
        t for t in tokens if t.isalnum() and t not in stop_words]
    return ' '.join(tokens_filtrados)

# --- 2. Lógica de Conexión a BD ---
def get_db_connection():
    """Establece y devuelve una conexión a la base de datos."""
    load_dotenv()
    DB_URL = os.getenv('DATABASE_URL')
    if not DB_URL and 'DATABASE_URL' in st.secrets:
        DB_URL = st.secrets['DATABASE_URL']
    
    if not DB_URL:
        st.error("Error: No se encontró DATABASE_URL. La aplicación no puede conectarse a la BD.")
        print("Error al conectar: No se encontró DATABASE_URL.")
        return None
        
    try:
        conn = psycopg2.connect(DB_URL)
        return conn
    except Exception as e:
        st.error(f"Error al conectar con la base de datos: {e}")
        print(f"Error al conectar con la base de datos: {e}")
        return None

# --- 3. Lógica de Registro de Fallos ---
def registrar_pregunta_fallida(pregunta):
    """Registra la pregunta que el bot no entendió."""
    print(f"Registrando pregunta fallida: {pregunta}")
    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn.cursor() as cursor:
            # 💡 NOTA: Tu tabla 'preguntas_sin_respuesta' debería tener un ID
            # CREATE TABLE IF NOT EXISTS preguntas_sin_respuesta (
            #     id SERIAL PRIMARY KEY,
            #     pregunta_usuario TEXT NOT NULL,
            #     fecha_creacion TIMESTAMP DEFAULT NOW()
            # );
            sql_query = "INSERT INTO preguntas_sin_respuesta (pregunta_usuario) VALUES (%s)"
            cursor.execute(sql_query, (pregunta,))
            conn.commit()
    except Exception as e:
        print(f"Error al registrar la pregunta fallida en la BD: {e}")
    finally:
        if conn:
            conn.close()

# --- 4. Lógica de Carga de Conocimiento (IA) ---
def cargar_conocimiento_y_modelo():
    """Carga el modelo de ML y vectoriza el conocimiento de la BD."""
    faq_data = []
    question_vectors = []
    model = None
    
    try:
        # 1. Cargar el Modelo de ML
        print("Cargando modelo de lenguaje...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("Modelo cargado.")

        # 2. Conectar a la BD
        conn = get_db_connection()
        if conn is None:
            st.error("Fallo al cargar conocimiento: No hay conexión a la BD.")
            return None, None, None

        with conn.cursor() as cursor:
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

        question_vectors = model.encode(preguntas_limpias)
        print(f"Conocimiento cargado y vectorizado: {len(faq_data)} preguntas.")
        
        return model, faq_data, question_vectors

    except Exception as e:
        print(f"Error fatal al cargar el conocimiento: {e}")
        st.error(f"Error de conexión o carga de modelo: {e}")
        return None, None, None

# --- 5. Lógica de Respuesta (IA) ---
def responder(pregunta_usuario, model, faq_data, question_vectors):
    """Genera una respuesta basada en la entrada del usuario."""
    texto_filtrado = limpiar_texto(pregunta_usuario)

    # 1. Búsqueda por palabra clave
    for item in faq_data:
        for palabra in item['palabras_clave']:
            if palabra in texto_filtrado:
                return item['respuesta']

    # 2. Búsqueda por ML (Similitud Semántica)
    if model and question_vectors is not None:
        user_vector = model.encode([texto_filtrado])
        similarities = cosine_similarity(user_vector, question_vectors)
        best_match_index = np.argmax(similarities)
        best_score = similarities[0][best_match_index]
        
        # 3. Devolución con umbral
        if best_score >= 0.65:  
            return faq_data[best_match_index]['respuesta']
        else:
            # Si la IA no está segura, registra la pregunta
            registrar_pregunta_fallida(pregunta_usuario)
            return "Lo siento, no estoy seguro de entender tu pregunta. 😅 ¿Podrías reformularla?"
    else:
        return "Error: El modelo de IA no está cargado."

