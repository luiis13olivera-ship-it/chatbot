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

# --- 💡 SUGERENCIA: Definir el umbral como una constante ---
UMBRAL_CONFIANZA_IA = 0.75 # Puedes ajustar este valor (ej. 0.7, 0.8)

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
            #    id SERIAL PRIMARY KEY,
            #    pregunta_usuario TEXT NOT NULL,
            #    fecha_creacion TIMESTAMP DEFAULT NOW()
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

# --- 💡 CORRECCIÓN CRÍTICA: Añadir caché de recursos ---
def cargar_conocimiento_y_modelo():
    """Carga el modelo de ML y vectoriza el conocimiento de la BD."""
    faq_data = []
    question_vectors = []
    model = None
    
    try:
        # 1. Cargar el Modelo de ML
        print("Cargando modelo de lenguaje (esto solo pasará una vez)...")
        st.toast("Cargando modelo de IA...") # Feedback para el usuario
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
        conn.close() # Cerramos la conexión, ya tenemos los datos.

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

        # Codificamos en lote (más rápido)
        question_vectors = model.encode(preguntas_limpias)
        print(f"Conocimiento cargado y vectorizado: {len(faq_data)} preguntas.")
        
        return model, faq_data, question_vectors

    except Exception as e:
        print(f"Error fatal al cargar el conocimiento: {e}")
        st.error(f"Error de conexión o carga de modelo: {e}")
        return None, None, None
# --- 💡 NUEVA FUNCIÓN (CORREGIDA): Búsqueda de Productos en BD ---
# --- 💡 NUEVA FUNCIÓN (OPTIMIZADA Y MÁS FLEXIBLE) ---
# --- 💡 NUEVA FUNCIÓN (FINAL): Búsqueda de Productos en BD ---
def buscar_productos(texto_filtrado):
    """
    Busca en la tabla 'productos' usando los términos del texto_filtrado.
    Versión corregida y optimizada.
    """
    if not texto_filtrado or len(texto_filtrado.strip()) < 3:
        print("DEBUG: Texto de búsqueda muy corto o vacío")
        return []

    conn = get_db_connection()
    if conn is None:
        print("DEBUG: No hay conexión a la BD")
        return []

    try:
        # Limpiar y preparar términos de búsqueda
        texto_filtrado = texto_filtrado.strip()
        terminos = texto_filtrado.split()
        
        print(f"DEBUG: Términos de búsqueda: {terminos}")
        
        # Filtrar términos válidos (más de 2 caracteres y no stopwords)
        terminos_validos = []
        for termino in terminos:
            if len(termino) > 2 and termino not in stop_words:
                terminos_validos.append(termino)
        
        if not terminos_validos:
            print("DEBUG: No hay términos válidos para buscar")
            return []

        # Construir consulta SQL dinámica
        condiciones = []
        params = []
        
        for termino in terminos_validos:
            # Buscar en múltiples columnas
            condiciones.append("(nombre ILIKE %s OR categoria ILIKE %s OR descripcion ILIKE %s)")
            params.extend([f"%{termino}%", f"%{termino}%", f"%{termino}%"])

        # Construir consulta final
        sql_base = """
            SELECT nombre, stock, precio, categoria 
            FROM productos 
            WHERE {}
            ORDER BY 
                stock DESC,
                CASE WHEN stock > 0 THEN 0 ELSE 1 END,
                nombre ASC
            LIMIT 8
        """.format(" OR ".join(condiciones))

        print(f"DEBUG SQL: {sql_base}")
        print(f"DEBUG Parámetros: {params}")

        # Ejecutar consulta
        with conn.cursor() as cursor:
            cursor.execute(sql_base, tuple(params))
            resultados_db = cursor.fetchall()
            
            print(f"DEBUG: Encontrados {len(resultados_db)} productos")
            
            if not resultados_db:
                return []

            # Formatear resultados
            resultados_formateados = []
            for row in resultados_db:
                nombre_prod = row[0]
                stock_prod = row[1]
                precio_prod = row[2] if row[2] is not None else 0
                categoria_prod = row[3] if row[3] else "Sin categoría"
                
                # Formatear precio
                if precio_prod > 0:
                    precio_str = f"S/ {precio_prod:.2f}"
                else:
                    precio_str = "Consultar precio"
                
                # Determinar estado de stock
                if stock_prod > 10:
                    stock_str = f"Stock: {stock_prod} (Disponible)"
                    icono = "✅"
                elif stock_prod > 0:
                    stock_str = f"Stock: {stock_prod} (Últimas unidades)"
                    icono = "⚠️"
                else:
                    stock_str = "Agotado"
                    icono = "❌"
                
                resultado_linea = f"{icono} **{nombre_prod}** - {precio_str} - {stock_str} - *{categoria_prod}*"
                resultados_formateados.append(resultado_linea)

            return resultados_formateados

    except Exception as e:
        print(f"ERROR en buscar_productos: {str(e)}")
        # Información adicional para debugging
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return []
    
    finally:
        if conn:
            conn.close()
            print("DEBUG: Conexión cerrada")

# --- 5. Lógica de Respuesta (IA) ---
def responder(pregunta_usuario, model, faq_data, question_vectors):
    """Genera una respuesta basada en la entrada del usuario."""
    texto_filtrado = limpiar_texto(pregunta_usuario)

    if not texto_filtrado:
        registrar_pregunta_fallida(pregunta_usuario)
        return "Lo siento, no estoy seguro de entender tu pregunta. 😅 ¿Podrías reformularla?"

    # --- 1. Búsqueda por palabra clave (igual que antes) ---
    for item in faq_data:
        palabras_filtradas = texto_filtrado.split()
        for palabra_clave in item['palabras_clave']:
            if palabra_clave in palabras_filtradas:
                return item['respuesta']

    # --- 💡 NUEVO: Búsqueda dinámica de productos ---
    # Buscamos productos *antes* de decidir si la IA falló
    resultados_productos = buscar_productos(texto_filtrado)


    # --- 2. Búsqueda por ML (Similitud Semántica) ---
    respuesta_ia = None
    if model and question_vectors is not None and len(question_vectors) > 0:
        user_vector = model.encode([texto_filtrado])
        similarities = cosine_similarity(user_vector, question_vectors)
        best_match_index = np.argmax(similarities)
        best_score = similarities[0][best_match_index]
        
        if best_score >= UMBRAL_CONFIANZA_IA:  
            respuesta_ia = faq_data[best_match_index]['respuesta']
            
    elif not model:
        return "Error: El modelo de IA no está cargado."
    # (Si el modelo está cargado pero la BD de IA está vacía, se maneja abajo)


    # --- 3. 💡 NUEVA LÓGICA DE COMBINACIÓN ---

    # Caso 1: La IA encontró una respuesta Y encontramos productos
    if respuesta_ia and resultados_productos:
        respuesta_final = (
            f"{respuesta_ia}\n\n"
            "Además, he encontrado estos productos relacionados en nuestro inventario:\n"
            + "\n".join(resultados_productos)
        )
        return respuesta_final

    # Caso 2: La IA encontró una respuesta, pero NO hay productos
    elif respuesta_ia:
        return respuesta_ia # Comportamiento original

    # Caso 3: La IA NO encontró respuesta, PERO SÍ encontramos productos
    elif not respuesta_ia and resultados_productos:
        respuesta_final = (
            "No he encontrado una respuesta exacta a tu pregunta, "
            "pero sí encontré estos productos que podrían interesarte:\n"
            + "\n".join(resultados_productos)
        )
        return respuesta_final

    # Caso 4: Ni la IA entendió, ni encontramos productos (Fallo total)
    else:
        registrar_pregunta_fallida(pregunta_usuario)
        if not faq_data: # Si la BD de conocimiento está vacía
             return "Lo siento, no tengo información sobre eso en mi base de conocimiento. ¿Puedes preguntar de otra forma?"
        else: # Si la IA simplemente no tuvo confianza
             return "Lo siento, no estoy seguro de entender tu pregunta. 😅 ¿Podrías reformularla?"
