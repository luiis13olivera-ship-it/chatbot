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


# --- INTERFAZ GRÁFICA MEJORADA ---

# Configuración de la página
st.set_page_config(
    page_title="Chatbot Verese Sac",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem !important;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #e9ecef;
        max-height: 500px;
        overflow-y: auto;
    }
    .user-message {
        background: linear-gradient(135deg, #007bff, #0056b3);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 0px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        animation: fadeIn 0.3s ease-in;
    }
    .bot-message {
        background: linear-gradient(135deg, #e9ecef, #dee2e6);
        color: #333;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 0px;
        margin: 8px 0;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        animation: fadeIn 0.3s ease-in;
    }
    .stChatInput {
        border-radius: 25px !important;
        border: 2px solid #e9ecef !important;
        padding: 12px 20px !important;
    }
    .stChatInput:focus {
        border-color: #007bff !important;
        box-shadow: 0 0 0 2px rgba(0,123,255,0.25) !important;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    .online {
        background-color: #28a745;
    }
    .offline {
        background-color: #dc3545;
    }
    .suggestion-chip {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border: 1px solid #90caf9;
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 5px;
        display: inline-block;
    }
    .suggestion-chip:hover {
        background: linear-gradient(135deg, #bbdefb, #90caf9);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.7; }
        100% { transform: scale(1); opacity: 1; }
    }
    .welcome-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header mejorado con banner de bienvenida
st.markdown("""
<div class="welcome-banner">
    <h1 style="margin:0; font-size: 2.5rem;">🤖 Chatbot Verese Sac</h1>
    <p style="margin:0; font-size: 1.2rem; opacity: 0.9;">v2.0 con IA • Tu asistente virtual inteligente</p>
</div>
""", unsafe_allow_html=True)

# Barra de estado mejorada
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Indicador de estado dinámico
    status_text = st.empty()
    if len(st.session_state.messages) > 0:
        status_text.markdown(
            f'<div style="text-align: center; background: #f8f9fa; padding: 10px; border-radius: 10px; border-left: 4px solid #28a745;">'
            f'<span class="status-indicator online"></span>'
            f'<span style="color: #28a745; font-weight: bold;">En línea • {len(st.session_state.messages)} mensajes en la conversación</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        status_text.markdown(
            '<div style="text-align: center; background: #f8f9fa; padding: 10px; border-radius: 10px; border-left: 4px solid #28a745;">'
            '<span class="status-indicator online"></span>'
            '<span style="color: #28a745; font-weight: bold;">En línea • Listo para ayudarte con tus consultas</span>'
            '</div>',
            unsafe_allow_html=True
        )

# Preguntas sugeridas mejoradas
st.markdown("### 💡 ¿En qué puedo ayudarte?")
suggested_questions = [
    "¿Cuál es su horario de atención?",
    "¿Qué métodos de pago aceptan?",
    "¿Tienen repuestos para Toyota?",
    "¿Realizan envíos a domicilio?",
    "¿Ofrecen garantía en los repuestos?",
    "¿Dónde están ubicados?"
]

# Crear chips de sugerencias
suggestions_html = '<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0;">'
for i, question in enumerate(suggested_questions):
    suggestions_html += f'''
    <div class="suggestion-chip" onclick="document.getElementById('suggestion-{i}').click()">
        {question}
    </div>
    '''
suggestions_html += '</div>'
st.markdown(suggestions_html, unsafe_allow_html=True)

# Botones ocultos para las sugerencias
for i, question in enumerate(suggested_questions):
    if st.button(question, key=f"sugg_{i}", help=f"Hacer esta pregunta: {question}"):
        st.session_state.suggested_question = question

# Contenedor del chat mejorado
st.markdown("### 💬 Conversación")
st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)

# Mostrar historial de mensajes con mejor formato
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">👤 *Tú:* {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">🤖 *Asistente:* {message["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input de chat mejorado
if "suggested_question" in st.session_state:
    prompt = st.chat_input("Escribe tu consulta aquí...", value=st.session_state.suggested_question)
    del st.session_state.suggested_question
else:
    prompt = st.chat_input("Escribe tu consulta aquí...")

# Procesar mensaje
if prompt:
    # Mostrar mensaje del usuario inmediatamente
    st.markdown(f'<div class="user-message">👤 *Tú:* {prompt}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Mostrar indicador de typing con mejor diseño
    with st.spinner("🤖 *El bot está procesando tu consulta...*"):
        # Cargar modelo y datos
        model, faq_data, question_vectors = cargar_conocimiento_y_modelo()
        
        # Obtener respuesta
        if not faq_data or model is None:
            response = "❌ *Lo siento, hay un problema técnico.* No pude cargar mi base de conocimiento. Por favor, intenta más tarde."
        else:
            response = responder(prompt, model, faq_data, question_vectors)

    # Mostrar respuesta del bot
    st.markdown(f'<div class="bot-message">🤖 *Asistente:* {response}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Auto-scroll con JavaScript
    st.markdown("""
    <script>
        function scrollToBottom() {
            const chatContainer = document.getElementById('chat-container');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
        setTimeout(scrollToBottom, 100);
    </script>
    """, unsafe_allow_html=True)

    # Recargar la página para actualizar la visualización
    st.rerun()

# Footer mejorado
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 10px;">
        <span>🔍 <strong>Búsqueda inteligente</strong></span>
        <span>🤖 <strong>IA avanzada</strong></span>
        <span>⚡ <strong>Respuestas rápidas</strong></span>
    </div>
    <div>
        💡 <strong>Tip:</strong> Sé específico en tus preguntas para obtener mejores respuestas<br>
        <em>Powered by Streamlit & Sentence Transformers</em>
    </div>
</div>
""", unsafe_allow_html=True)
