import streamlit as st
# Importamos las funciones clave desde utils.py
from utils import cargar_conocimiento_y_modelo, responder

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
