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
    print("Descargando recursos de NLTK...")
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

def registrar_pregunta_fallida(pregunta):
    """
    Se conecta a la BD y registra la pregunta que el bot no entendió.
    """
    print(f"Registrando pregunta fallida: {pregunta}")
    try:
        DB_URL = None
        load_dotenv() 
        DB_URL = os.getenv('DATABASE_URL')
        if not DB_URL and 'DATABASE_URL' in st.secrets:
            DB_URL = st.secrets['DATABASE_URL']

        if not DB_URL:
            print("Error al registrar: No se encontró DATABASE_URL.")
            return

        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        
        # Insertamos la pregunta en la nueva tabla
        sql_query = "INSERT INTO preguntas_sin_respuesta (pregunta_usuario) VALUES (%s)"
        cursor.execute(sql_query, (pregunta,))
        
        conn.commit() # ¡Importante! Guardamos los cambios
        
    except Exception as e:
        # Si falla el registro, no detenemos el bot, solo lo imprimimos en el log
        print(f"Error al registrar la pregunta fallida en la BD: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

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
            return None, None, None

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

# --- 💡 LÓGICA DE RESPUESTA (Semántica + Palabras Clave) ---
def responder(pregunta_usuario, model, faq_data, question_vectors):
    texto_filtrado = limpiar_texto(pregunta_usuario)

    if not texto_filtrado:
        return "Disculpa, no detecté ninguna palabra clave."

    # 1️⃣ Búsqueda por palabra clave (rápida, para coincidencias exactas)
    for item in faq_data:
        for palabra in item['palabras_clave']:
            if palabra in texto_filtrado:
                return item['respuesta']

    # 2️⃣ Búsqueda por ML (Similitud Semántica)
    if model:
        user_vector = model.encode([texto_filtrado])
        similarities = cosine_similarity(user_vector, question_vectors)
        best_match_index = np.argmax(similarities)
        best_score = similarities[0][best_match_index]
        
        # 3️⃣ Devolución con umbral
        if best_score >= 0.65: 
            return faq_data[best_match_index]['respuesta']
        else:
            # Si la IA no está segura, registra la pregunta
            registrar_pregunta_fallida(pregunta_usuario)
            return "Lo siento, no estoy seguro de entender tu pregunta. 😅 ¿Podrías reformularla?"
    else:
        return "Error: El modelo de IA no está cargado."

# --- INTERFAZ GRÁFICA MEJORADA ---

# 1. Cargar el modelo Y los datos
model, faq_data, question_vectors = cargar_conocimiento_y_modelo()

# --- 🎨 MEJORA DE DISEÑO 2: BARRA LATERAL CON ESTATUS E INSTRUCCIONES ---
with st.sidebar:
    st.image("https://placehold.co/150x50/3498db/ffffff?text=VERESE+SAC", use_column_width=True)
    st.markdown("---")
    st.header("⚙️ Estado y Configuración")
    
    if faq_data is not None and model is not None:
         st.success(f"✅ Base de Conocimiento Activa ({len(faq_data)} temas)")
         
         st.markdown("---")
         st.subheader("💡 Tips para Preguntar")
         st.markdown("""
         - *Sé específico* (ej: "¿Qué repuestos tienen para motor Toyota?").
         - Pregunta por *servicios, horarios o ubicación*.
         - Evita la jerga compleja.
         """)
    else:
         st.error("⚠️ El bot no pudo cargar la base de conocimiento o el modelo. Por favor, verifica la variable DATABASE_URL.")
    
    st.markdown("---")
    st.caption("Desarrollado para Verese Sac (v2.0 IA)")


# --- 🎨 MEJORA DE DISEÑO 3: TÍTULO PRINCIPAL CON ICONO ---
st.title("🔩 Asistente Virtual Verese Sac 🤖")
st.caption("Tu experto en repuestos: Consulta la disponibilidad, servicios y detalles de nuestros productos.")

# --- 🎨 MEJORA DE DISEÑO 4: MENSAJE DE BIENVENIDA INICIAL ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy tu Asistente de Repuestos Verese Sac. ¿En qué puedo ayudarte hoy? Por ejemplo, puedes preguntarme por un repuesto o nuestros horarios."}
    ]

# Mostrar el historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Manejar la entrada del usuario
if prompt := st.chat_input("Escribe tu consulta aquí..."):

    # Mostrar la pregunta del usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generar la respuesta del bot
    if not faq_data or model is None:
        response = "Error: El bot no está operativo. Por favor, revisa la conexión a la base de datos."
    else:
        response = responder(prompt, model, faq_data, question_vectors)

    # Mostrar la respuesta del asistente
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
