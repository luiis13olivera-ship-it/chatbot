import streamlit as st
# Importamos las funciones clave desde utils.py
from utils import cargar_conocimiento_y_modelo, responder
import streamlit as st
# Importamos las funciones clave desde utils.py
from utils import cargar_conocimiento_y_modelo, responder

# --- 🎨 CONFIGURACIÓN DE PÁGINA Y ESTILOS (NUEVO) ---
st.set_page_config(layout="wide")

# Inyectamos el CSS personalizado inspirado en index.html
st.markdown("""
    <style>
        /* Variables de color extraídas de index.html */
        :root {
            --primary: #E4202B; /* Rojo Verese */
            --primary-dark: #C61C26;
            --sidebar: #1a1b26;
            --chat-bg: #161722;
            --card-bg: #1e1f2e;
            --text-primary: #e0e6f0;
            --text-secondary: #a0a6b8;
            --border: #2a2b3c;
            --success: #10B981;
            --warning: #F59E0B;
        }
        
        /* Fondo principal de la App */
        [data-testid="stAppViewContainer"] {
            background-color: var(--chat-bg);
            color: var(--text-primary);
        }
        
        /* Barra lateral */
        [data-testid="stSidebar"] {
            background-color: var(--sidebar);
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
            color: var(--text-primary);
        }
        
        /* Contenedores de mensajes de chat */
        [data-testid="stChatMessage"] {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        /* Color del texto y cabeceras */
        h1, h2, h3, h4, h5, h6, .stMarkdown, label {
            color: var(--text-primary);
        }
        
        /* Inputs y textareas */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div {
            background-color: var(--card-bg) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px;
        }
        
        /* Botones (general) */
        .stButton > button {
            border-radius: 10px;
            border: 1px solid var(--border);
        }

        /* Botón Primario (rojo) */
        .stButton > button[kind="primary"] {
            background-color: var(--primary);
            color: white;
            border: none;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: var(--primary-dark);
            color: white;
            border: none;
        }

        /* Botón Secundario (oscuro) */
        .stButton > button[kind="secondary"] {
            background-color: var(--card-bg);
            color: var(--text-primary);
        }
        .stButton > button[kind="secondary"]:hover {
            background-color: var(--border);
            color: var(--text-primary);
        }
        
        /* Éxito (verde) y Advertencia (amarillo) */
        [data-testid="stSuccess"] {
            background-color: rgba(16, 185, 129, 0.2);
            color: var(--success);
            border: 1px solid var(--success);
        }
        [data-testid="stWarning"] {
            background-color: rgba(245, 158, 11, 0.2);
            color: var(--warning);
            border: 1px solid var(--warning);
        }

    </style>
""", unsafe_allow_html=True)
# --- FIN DE ESTILOS ---


# --- 🎨 INTERFAZ GRÁFICA ---

# 1. Cargar el modelo Y los datos (esto usa la caché de utils.py)
model, faq_data, question_vectors = cargar_conocimiento_y_modelo()

# 2. Barra Lateral
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


# 3. Título Principal
st.title("🔩 Asistente Virtual Verese Sac 🤖")
st.caption("Tu experto en repuestos: Consulta la disponibilidad, servicios y detalles de nuestros productos.")

# 4. Mensaje de Bienvenida Inicial
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy tu Asistente de Repuestos Verese Sac. ¿En qué puedo ayudarte hoy? Por ejemplo, puedes preguntarme por un repuesto o nuestros horarios."}
    ]

# 5. Mostrar historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Manejar la entrada del usuario
if prompt := st.chat_input("Escribe tu consulta aquí..."):

    # Mostrar la pregunta del usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generar la respuesta del bot
    if not faq_data or model is None:
        response = "Error: El bot no está operativo. Por favor, revisa la conexión a la base de datos."
    else:
        # Usamos la función 'responder' importada
        response = responder(prompt, model, faq_data, question_vectors)

    # Mostrar la respuesta del asistente
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
