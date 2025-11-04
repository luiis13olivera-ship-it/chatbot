import streamlit as st
from utils import get_db_connection # Importamos solo la conexión a la BD
import pandas as pd

st.set_page_config(layout="wide")

st.title("🧠 Centro de Entrenamiento del Asistente")
st.caption("Revisa las preguntas que el bot no entendió y añádelas a su base de conocimiento.")

# --- 💡 Consejo Importante sobre tu BD ---
with st.expander("🔑 ¡IMPORTANTE! Asegúrate que tu tabla 'preguntas_sin_respuesta' tenga un ID"):
    st.code("""
    -- Si aún no tienes la tabla, usa esto:
    CREATE TABLE IF NOT EXISTS preguntas_sin_respuesta (
        id SERIAL PRIMARY KEY,
        pregunta_usuario TEXT NOT NULL,
        fecha_registro TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Si ya la tenías pero sin 'id', ejecuta esto UNA VEZ:
    -- ALTER TABLE preguntas_sin_respuesta ADD COLUMN id SERIAL PRIMARY KEY;
    -- ALTER TABLE preguntas_sin_respuesta ADD COLUMN fecha_registro TIMESTAMP WITH TIME ZONE DEFAULT NOW();
    """, language="sql")

# --- 💡 Consejo 2: Tu tabla 'chatbot_conocimiento' ---
with st.expander("🔑 ¡IMPORTANTE! Revisa tu tabla 'chatbot_conocimiento'"):
    st.code("""
    -- ¡Entendido! Veo que tu tabla 'chatbot_conocimiento' usa un 'id' de tipo TEXT (ej. 'R001', 'R002').
    -- ¡Esto es perfectamente válido!
    
    -- El error 'null value in column "id"' sucedía porque el código anterior 
    -- no estaba generando este ID de texto antes de insertarlo.
    
    -- He modificado el código para que genere automáticamente el siguiente
    -- ID (ej. si el máximo es 'R019', generará 'R020').
    
    -- Tu estructura de tabla es correcta, asegúrate de que 'id' sea PRIMARY KEY:
    CREATE TABLE IF NOT EXISTS chatbot_conocimiento (
        id TEXT PRIMARY KEY,
        pregunta TEXT,
        respuesta TEXT,
        palabras_clave TEXT
    );
    
    """, language="sql")

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
            
            # --- 💡 NUEVO: Generar el siguiente ID de TEXTO (ej. R020) ---
            # Extrae la parte numérica del ID, la convierte a entero, y saca el máximo
            cursor.execute("SELECT MAX(CAST(SUBSTRING(id FROM 2) AS INTEGER)) FROM chatbot_conocimiento WHERE id LIKE 'R%';")
            max_id_num = cursor.fetchone()[0]
            
            if max_id_num is None:
                # Si la tabla está vacía
                next_id_num = 1
            else:
                next_id_num = max_id_num + 1
            
            # Formatea el nuevo ID a 3 dígitos (ej. R001, R020, R123)
            new_id = f"R{next_id_num:03d}"
            
            # --- 💡 MODIFICADO: Insertar con el nuevo ID de texto ---
            sql = """
            INSERT INTO chatbot_conocimiento (id, pregunta, respuesta, palabras_clave)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (new_id, pregunta, respuesta, palabras_clave))
            conn.commit()
        return True
    except Exception as e:
        # --- 💡 MEJORA: Detectar error de ID duplicado ---
        if "duplicate key value violates unique constraint" in str(e):
            st.error(f"Error de ID duplicado: Se intentó guardar con un ID ('{new_id}') que ya existe. Revisa la base de datos o intenta de nuevo.")
        elif "violates not-null constraint" in str(e) and "\"id\"" in str(e):
            st.error(f"Error al guardar: La columna 'id' en tu tabla 'chatbot_conocimiento' no puede ser nula.")
            st.warning("El código intentó generar un ID pero algo falló.")
        else:
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
            # El ID se pasa aquí, nos aseguramos que sea un 'int'
            sql = "DELETE FROM preguntas_sin_respuesta WHERE id = %s"
            cursor.execute(sql, (id_pregunta,))
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Error al eliminar pregunta pendiente: {e}")
        # Imprimimos el error en la consola para más depuración
        print(f"Error en resolver_pregunta_pendiente: {e}") 
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
    # .any() es para evitar un error de pandas si no hay selección
    if selected_id:
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
                    #    💡 AQUÍ ESTÁ EL ARREGLO: convertimos a int()
                    if resolver_pregunta_pendiente(int(pregunta_seleccionada['id'])):
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
            # 💡 AQUÍ ESTÁ EL ARREGLO: convertimos a int()
            if resolver_pregunta_pendiente(int(pregunta_seleccionada['id'])):
                st.success(f"Pregunta ID {pregunta_seleccionada['id']} descartada exitosamente.")
                
                # No limpiamos caché de modelo, pero sí refrescamos la UI
                st.rerun()
            else:
                st.error("No se pudo descartar la pregunta.")

