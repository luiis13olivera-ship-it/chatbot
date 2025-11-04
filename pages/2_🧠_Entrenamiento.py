import streamlit as st
from utils import get_db_connection # Importamos solo la conexión a la BD
import pandas as pd

st.set_page_config(layout="wide")

st.title("🧠 Centro de Entrenamiento del Asistente")
st.caption("Revisa las preguntas que el bot no entendió y añádelas a su base de conocimiento.")


# --- Funciones de esta página ---

def cargar_preguntas_pendientes():
    """Obtiene todas las preguntas de la tabla 'preguntas_sin_respuesta'."""
    conn = get_db_connection()
    if conn is None:
        st.error("No se puede conectar a la BD.")
        return pd.DataFrame()
    try:
        # Usamos pandas para leer fácilmente la tabla
        query = "SELECT id, pregunta_usuario, fecha_registro FROM preguntas_sin_respuesta ORDER BY fecha_registro DESC"
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        # Esto pasa si la tabla o columnas 'id', 'fecha_registro' no existen
        if "column \"id\" does not exist" in str(e) or "column \"fecha_registro\" does not exist" in str(e):
             st.error("Error: Tu tabla 'preguntas_sin_respuesta' no tiene las columnas 'id' o 'fecha_registro'. Por favor, mira el consejo de arriba y actualiza tu base de datos.")
             return pd.DataFrame()
        st.error(f"Error al cargar preguntas: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def agregar_a_conocimiento(pregunta, respuesta, palabras_clave):
    """Inserta la nueva Q&A en la tabla principal 'chatbot_conocimiento'."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO chatbot_conocimiento (pregunta, respuesta, palabras_clave)
            VALUES (%s, %s, %s)
            """
            cursor.execute(sql, (pregunta, respuesta, palabras_clave))
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Error al guardar en conocimiento: {e}")
        return False
    finally:
        if conn: conn.close()

def resolver_pregunta_pendiente(id_pregunta):
    """Elimina la pregunta de la tabla 'preguntas_sin_respuesta'."""
    conn = get_db_connection()
    if conn is None: return False
    try:
        with conn.cursor() as cursor:
            sql = "DELETE FROM preguntas_sin_respuesta WHERE id = %s"
            cursor.execute(sql, (id_pregunta,))
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Error al eliminar pregunta pendiente: {e}")
        return False
    finally:
        if conn: conn.close()


# --- Interfaz de la Página ---

df_pendientes = cargar_preguntas_pendientes()

if df_pendientes.empty:
    st.success("¡Buen trabajo! No hay preguntas pendientes de entrenar. 🎉")
else:
    st.info(f"Tienes {len(df_pendientes)} preguntas pendientes por revisar.")
    
    # Usamos un 'st.data_editor' para ver la tabla
    st.subheader("Listado de Preguntas sin Respuesta")
    st.dataframe(df_pendientes, use_container_width=True)

    st.markdown("---")

    # --- 1. Selector de Pregunta ---
    st.subheader("Selecciona una pregunta para gestionar")
    
    # Creamos un diccionario para mostrar texto amigable en el selectbox
    # (Ej: "ID 123: El usuario preguntó...")
    opciones = {
        row['id']: f"ID {row['id']}: {row['pregunta_usuario'][:70]}..." 
        for index, row in df_pendientes.iterrows()
    }
    
    selected_id = st.selectbox(
        "Elige la pregunta que quieres resolver o descartar:",
        options=opciones.keys(),
        format_func=lambda id: opciones[id] # Muestra el texto amigable
    )

    # Obtenemos la fila completa de la pregunta seleccionada
    pregunta_seleccionada = df_pendientes[df_pendientes['id'] == selected_id].iloc[0]
    
    
    # --- 2. Formulario de Acción ---
    st.markdown(f"**Gestionando Pregunta ID: `{pregunta_seleccionada['id']}`**")
    
    # Formulario para agregar la respuesta
    with st.form(key=f"form_entrenamiento_{pregunta_seleccionada['id']}"):
        st.markdown(f"**Pregunta original del usuario:**")
        st.info(f"*{pregunta_seleccionada['pregunta_usuario']}*")
        
        st.markdown("**Completa la información para el bot:**")
        
        pregunta_oficial = st.text_area(
            "1. Pregunta (edita la original para que sea una buena 'pregunta oficial'):",
            value=pregunta_seleccionada['pregunta_usuario']
        )
        
        respuesta_nueva = st.text_area(
            "2. Respuesta (la respuesta que el bot debe dar):",
            placeholder="Escribe la respuesta completa aquí..."
        )
        
        palabras_clave = st.text_input(
            "3. Palabras Clave (opcional, separadas por ; ):",
            placeholder="ej: motor;repuesto;toyota"
        )
        
        # --- 3. Botones de Acción ---
        col1, col2 = st.columns(2)
        with col1:
            submit_aprender = st.form_submit_button(
                label="✅ Aprender y Resolver Pregunta", 
                use_container_width=True,
                type="primary"
            )
        with col2:
            submit_descartar = st.form_submit_button(
                label="❌ Descartar Pregunta (Eliminar)", 
                use_container_width=True,
                type="secondary"
            )

    # --- 4. Lógica de los Botones ---
    if submit_aprender:
        if not pregunta_oficial or not respuesta_nueva:
            st.warning("Debes completar al menos la 'Pregunta' y la 'Respuesta' para entrenar.")
        else:
            # 1. Agregar a la base de conocimiento
            if agregar_a_conocimiento(pregunta_oficial, respuesta_nueva, palabras_clave):
                # 2. Eliminar de la lista de pendientes
                if resolver_pregunta_pendiente(pregunta_seleccionada['id']):
                    st.success("¡Éxito! El bot ha aprendido la nueva respuesta.")
                    
                    # 3. MUY IMPORTANTE: Limpiar la caché del modelo
                    st.cache_resource.clear()
                    
                    # 4. Refrescar la página
                    st.rerun()
                else:
                    st.error("Se guardó la respuesta, pero no se pudo eliminar la pregunta de 'pendientes'.")
            else:
                st.error("No se pudo guardar la nueva respuesta en la base de conocimiento.")

    if submit_descartar:
        # Simplemente eliminamos la pregunta de "pendientes"
        if resolver_pregunta_pendiente(pregunta_seleccionada['id']):
            st.success(f"Pregunta ID {pregunta_seleccionada['id']} descartada exitosamente.")
            
            # No limpiamos caché de modelo, pero sí refrescamos la UI
            st.rerun()
        else:
            st.error("No se pudo descartar la pregunta.")

