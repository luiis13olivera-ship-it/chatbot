import streamlit as st
import os
import psycopg2
import nltk
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Comprobación de NLTK ---
# --- Comprobación de NLTK (Actualizado) ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab') # <-- Añadimos la comprobación
except LookupError:
    print("Descargando recursos de NLTK (esto solo pasa una vez)...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab') # <-- Añadimos la descarga faltante
    print("Recursos de NLTK descargados.")

stop_words = set(stopwords.words('spanish'))


def limpiar_texto(texto):
    texto = str(texto).lower()
    tokens = word_tokenize(texto)
    tokens_filtrados = [
        t for t in tokens if t.isalnum() and t not in stop_words]
    return ' '.join(tokens_filtrados)


# --- Conexión a la BD y Carga de Datos (Corregido) ---
@st.cache_resource
def cargar_conocimiento():
    faq = []
    try:
        DB_URL = None

        # 💡 MEJORA: Priorizamos el .env para desarrollo local
        # 1. Intentar cargar desde el archivo .env
        load_dotenv()
        DB_URL = os.getenv('DATABASE_URL')

        # 2. Si no se encontró en .env, intentar con st.secrets (para la nube)
        if not DB_URL:
            if 'DATABASE_URL' in st.secrets:
                DB_URL = st.secrets['DATABASE_URL']

        # 3. Si sigue sin encontrarse, fallar y avisar al usuario
        if not DB_URL:
            print("Error: No se pudo encontrar la variable DATABASE_URL.")
            st.error(
                "Error de configuración: No se pudo encontrar la variable DATABASE_URL. Asegúrate de crear el archivo .env localmente o de añadir el Secret en Streamlit Cloud.")
            return []

        # --- ¡Tu contraseña ya NO está en el código! ---
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT pregunta, palabras_clave, respuesta FROM chatbot_conocimiento")

        data_db = cursor.fetchall()

        for row in data_db:
            pregunta_original = row[0]
            palabras_clave_raw = row[1].split(';') if row[1] else []

            faq.append({
                'pregunta_limpia': limpiar_texto(pregunta_original),
                'palabras_clave': [p.strip().lower() for p in palabras_clave_raw if p.strip()],
                'respuesta': row[2]
            })

        conn.close()
        print(f"Conocimiento cargado exitosamente: {len(faq)} preguntas.")
        return faq

    except Exception as e:
        print(f"Error fatal al cargar el conocimiento desde la BD: {e}")
        st.error(f"Error de conexión: {e}")
        return []


# --- Lógica del Bot (sin cambios) ---
def responder(pregunta, faq_data):
    texto_filtrado = limpiar_texto(pregunta)

    if not texto_filtrado:
        return "Disculpa, no detecté ninguna palabra clave."

    # 1️⃣ Búsqueda por palabra clave
    for item in faq_data:
        for palabra in item['palabras_clave']:
            if palabra in texto_filtrado:
                return item['respuesta']

    # 2️⃣ Búsqueda difusa
    mejor_puntaje = 0
    mejor_respuesta = None
    for item in faq_data:
        similitud = fuzz.token_set_ratio(texto_filtrado, item['pregunta_limpia'])
        if similitud > mejor_puntaje:
            mejor_puntaje = similitud
            mejor_respuesta = item['respuesta']

    # 3️⃣ Devolución
    if mejor_puntaje >= 70:
        return mejor_respuesta
    else:
        return "Lo siento, no entiendo tu pregunta. 😅 ¿Podrías reformularla?"


# --- Interfaz Gráfica de la App Web (Nuevo) ---

# Título de la App
st.title("🤖 Chatbot de Repuestos Verese Sac")
st.caption("Tu asistente virtual para consultas sobre repuestos, horarios y servicios.")

# Cargar los datos (esto usará el caché)
# (Se mueve aquí para que el título aparezca primero,
# y si hay un error, se muestre debajo)
faq_data = cargar_conocimiento()

# Inicializar el historial del chat en la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes antiguos en el historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Obtener nueva entrada del usuario (reemplaza a input())
if prompt := st.chat_input("Escribe tu consulta aquí..."):

    # 1. Mostrar el mensaje del usuario y guardarlo
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generar y mostrar la respuesta del bot
    if not faq_data:
        # Manejo de error si la BD no cargó
        response = "Error: No pude cargar mi base de conocimiento. El bot no está operativo."
    else:
        response = responder(prompt, faq_data)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})