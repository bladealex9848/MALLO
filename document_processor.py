"""
Módulo para procesamiento de documentos y OCR en MALLO.
Proporciona funcionalidades para:
1. Validación de formatos de archivo
2. Procesamiento OCR con Mistral
3. Extracción de texto de diferentes tipos de documentos
4. Preparación de imágenes para OCR
"""

import os
import base64
import json
import requests
import tempfile
import logging
import traceback
import io
import uuid
import time
from datetime import datetime
from pathlib import Path
from io import BytesIO
from PIL import Image
import PyPDF2
import yaml
import streamlit as st

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("mallo.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Cargar configuración
def load_config():
    """Carga la configuración desde el archivo YAML."""
    try:
        with open("config.yaml", "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error("No se pudo encontrar el archivo de configuración 'config.yaml'.")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error al leer el archivo de configuración: {str(e)}")
        return {}


# Cargar configuración
config = load_config()

# Obtener configuración de OCR
ocr_config = config.get("utilities", {}).get("ocr", {})
file_handling_config = config.get("utilities", {}).get("file_handling", {})

# Definición de formatos permitidos y sus extensiones
ALLOWED_FILE_FORMATS = {
    "PDF": [".pdf"],
    "Imagen": [".jpg", ".jpeg", ".png"],
    "Texto": [".txt"],
}

# Lista plana de todas las extensiones permitidas
ALLOWED_EXTENSIONS = ocr_config.get(
    "allowed_formats", [".pdf", ".jpg", ".jpeg", ".png", ".txt"]
)


def validate_file_format(file):
    """
    Valida que el archivo tenga un formato permitido y que su contenido
    sea consistente con la extensión declarada.

    Parámetros:
        file: Objeto de archivo cargado por el usuario mediante Streamlit

    Retorno:
        tuple: (es_válido, tipo_documento, mensaje_error)
    """
    file_type = None

    # Verificar que el archivo tenga nombre
    if not hasattr(file, "name"):
        return False, None, "El archivo no tiene nombre"

    # Obtener extensión y verificar que esté permitida
    file_name = file.name.lower()
    file_ext = os.path.splitext(file_name)[1]

    if file_ext not in ALLOWED_EXTENSIONS:
        allowed_exts = ", ".join(ALLOWED_EXTENSIONS)
        return False, None, f"Formato de archivo no permitido. Use: {allowed_exts}"

    # Determinar el tipo de documento según la extensión
    for doc_type, extensions in ALLOWED_FILE_FORMATS.items():
        if file_ext in extensions:
            file_type = doc_type
            break

    # Verificar contenido según el tipo de archivo
    try:
        # Guardar posición del cursor
        position = file.tell()

        # Verificar contenido según tipo
        if file_type == "PDF":
            # Verificar firma de PDF
            header = file.read(8)
            file.seek(position)  # Restaurar posición

            if not header.startswith(b"%PDF"):
                return False, None, "El archivo no es un PDF válido"

        elif file_type == "Imagen":
            # Intentar abrir como imagen
            try:
                img = Image.open(file)
                img.verify()  # Verificar que la imagen sea válida
                file.seek(position)  # Restaurar posición
            except Exception as e:
                file.seek(position)  # Restaurar posición
                return False, None, f"El archivo no es una imagen válida: {str(e)}"

    except Exception as e:
        # Restaurar posición en caso de error
        try:
            file.seek(position)
        except:
            pass
        return False, None, f"Error validando el archivo: {str(e)}"

    # Si llegamos aquí, el archivo es válido
    return True, file_type, None


def detect_document_type(file):
    """
    Detecta automáticamente si un archivo es un PDF o una imagen
    con múltiples verificaciones para mayor precisión

    Parámetros:
        file: Objeto de archivo cargado por el usuario mediante Streamlit

    Retorno:
        string: Tipo de documento detectado ("PDF" o "Imagen")
    """
    # 1. Verificar por MIME type
    if hasattr(file, "type"):
        mime_type = file.type
        if mime_type.startswith("application/pdf"):
            return "PDF"
        elif mime_type.startswith("image/"):
            return "Imagen"

    # 2. Verificar por extensión del nombre
    if hasattr(file, "name"):
        name = file.name.lower()
        if name.endswith(".pdf"):
            return "PDF"
        elif name.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")):
            return "Imagen"

    # 3. Verificar contenido con análisis de bytes
    try:
        # Guardar posición del cursor
        position = file.tell()
        # Leer los primeros bytes
        header = file.read(8)
        file.seek(position)  # Restaurar posición

        # Verificar firmas de archivo comunes
        if header.startswith(b"%PDF"):
            return "PDF"
        elif header.startswith(b"\x89PNG") or header.startswith(b"\xff\xd8"):
            return "Imagen"
    except:
        pass

    # 4. Intentar abrir como imagen (último recurso)
    try:
        Image.open(file)
        file.seek(0)  # Restaurar el puntero
        return "Imagen"
    except:
        file.seek(0)  # Restaurar el puntero

    # Asumir PDF por defecto
    return "PDF"


def prepare_image_for_ocr(file_data):
    """
    Prepara una imagen para ser procesada con OCR,
    optimizando formato y calidad para mejorar resultados

    Parámetros:
        file_data: Datos binarios de la imagen

    Retorno:
        tuple: (datos_optimizados, mime_type)
    """
    try:
        # Abrir la imagen con PIL
        img = Image.open(BytesIO(file_data))

        # Optimizaciones avanzadas para OCR
        # 1. Convertir a escala de grises si tiene más de un canal
        if img.mode != "L" and img.mode != "1":
            img = img.convert("L")

        # 2. Ajustar tamaño si es muy grande (límite 4000px)
        max_dimension = 4000
        if img.width > max_dimension or img.height > max_dimension:
            ratio = min(max_dimension / img.width, max_dimension / img.height)
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)

        # 3. Evaluar y determinar mejor formato
        # JPEG para imágenes fotográficas, PNG para documentos/texto
        save_format = "JPEG"
        save_quality = 95

        # Detectar si es más probable que sea un documento (blanco/negro predominante)
        histogram = img.histogram()
        if img.mode == "L" and (histogram[0] + histogram[-1]) > sum(histogram) * 0.8:
            save_format = "PNG"

        # 4. Guardar con parámetros optimizados
        buffer = BytesIO()
        if save_format == "JPEG":
            img.save(buffer, format=save_format, quality=save_quality, optimize=True)
        else:
            img.save(buffer, format=save_format, optimize=True)

        buffer.seek(0)
        return buffer.read(), f"image/{save_format.lower()}"

    except Exception as e:
        logger.warning(f"Optimización de imagen fallida: {str(e)}")
        return file_data, "image/jpeg"  # Formato por defecto


def extract_text_from_ocr_response(response):
    """
    Extrae texto de diferentes formatos de respuesta OCR
    con soporte para múltiples estructuras de datos

    Parámetros:
        response: Respuesta JSON del servicio OCR

    Retorno:
        dict: Diccionario con el texto extraído
    """
    # Registro para diagnóstico
    logger.info(f"Procesando respuesta OCR de tipo: {type(response)}")

    # Caso 1: Si hay páginas con markdown (formato preferido)
    if "pages" in response and isinstance(response["pages"], list):
        pages = response["pages"]
        if pages and "markdown" in pages[0]:
            markdown_text = "\n\n".join(page.get("markdown", "") for page in pages)
            if markdown_text.strip():
                return {"text": markdown_text, "format": "markdown"}

    # Caso 2: Si hay un texto plano en la respuesta
    if "text" in response:
        return {"text": response["text"], "format": "text"}

    # Caso 3: Si hay elementos estructurados
    if "elements" in response:
        elements = response["elements"]
        if isinstance(elements, list):
            text_parts = []
            for element in elements:
                if "text" in element:
                    text_parts.append(element["text"])
            return {"text": "\n".join(text_parts), "format": "elements"}

    # Caso 4: Si hay un campo 'content' principal
    if "content" in response:
        return {"text": response["content"], "format": "content"}

    # Caso 5: Extracción recursiva de todos los campos de texto
    try:
        response_str = json.dumps(response, indent=2)
        # Si la respuesta es muy grande, limitar extracción
        if len(response_str) > 10000:
            response_str = response_str[:10000] + "... [truncado]"

        extracted_text = extract_all_text_fields(response)
        if extracted_text:
            return {"text": extracted_text, "format": "extracted"}

        return {
            "text": "No se pudo encontrar texto estructurado en la respuesta OCR. Vea los detalles técnicos.",
            "format": "unknown",
            "raw_response": response_str,
        }
    except Exception as e:
        logger.error(f"Error al procesar respuesta OCR: {str(e)}")
        return {"error": f"Error al procesar la respuesta: {str(e)}"}


def extract_all_text_fields(data, prefix="", max_depth=5, current_depth=0):
    """
    Función recursiva optimizada para extraer todos los campos de texto
    de un diccionario anidado con límites de profundidad

    Parámetros:
        data: Diccionario o lista de datos
        prefix: Prefijo para la ruta de acceso (uso recursivo)
        max_depth: Profundidad máxima de recursión
        current_depth: Profundidad actual (uso recursivo)

    Retorno:
        string: Texto extraído
    """
    # Evitar recursión infinita
    if current_depth > max_depth:
        return []

    result = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key

            if isinstance(value, str) and len(value) > 1:
                result.append(f"{new_prefix}: {value}")
            elif (
                isinstance(value, (dict, list)) and value
            ):  # Solo recursión si hay contenido
                child_results = extract_all_text_fields(
                    value, new_prefix, max_depth, current_depth + 1
                )
                result.extend(child_results)

    elif isinstance(data, list):
        # Limitar número de elementos procesados en listas muy grandes
        max_items = 20
        for i, item in enumerate(data[:max_items]):
            new_prefix = f"{prefix}[{i}]"
            if isinstance(item, (dict, list)) and item:
                child_results = extract_all_text_fields(
                    item, new_prefix, max_depth, current_depth + 1
                )
                result.extend(child_results)
            elif isinstance(item, str) and len(item) > 1:
                result.append(f"{new_prefix}: {item}")

        # Indicar si se truncó la lista
        if len(data) > max_items:
            result.append(
                f"{prefix}: [... {len(data) - max_items} elementos adicionales omitidos]"
            )

    return "\n".join(result)


def process_document_with_mistral_ocr(api_key, file_bytes, file_type, file_name):
    """
    Procesa un documento con OCR de Mistral
    con sistema de recuperación ante fallos

    Parámetros:
        api_key: API key de Mistral
        file_bytes: Bytes del archivo
        file_type: Tipo de archivo ("PDF", "Imagen", "Texto")
        file_name: Nombre del archivo

    Retorno:
        dict: Texto extraído del documento
    """
    job_id = str(uuid.uuid4())
    logger.info(f"Procesando documento {file_name} con Mistral OCR (ID: {job_id})")

    # Crear un objeto de estado para seguimiento
    status_obj = {"label": f"Procesando documento {file_name}...", "state": "running"}

    # Verificar contexto para decidir cómo mostrar el estado
    # Siempre usamos un enfoque seguro para evitar problemas de anidamiento
    use_streamlit_status = (
        False  # Por defecto, no usamos st.status para evitar problemas
    )

    status = None
    status_container = None
    if use_streamlit_status:
        status = st.status(status_obj["label"], expanded=True)
    else:
        # Si no podemos usar st.status, usamos st.write para mostrar el progreso
        status_container = st.empty()
        status_container.write(f"🔄 {status_obj['label']}")

    # Función para actualizar el estado
    def update_status(label, state="running"):
        status_obj["label"] = label
        status_obj["state"] = state
        if use_streamlit_status and status:
            status.update(label=label, state=state)
        else:
            icon = "✅" if state == "complete" else "❌" if state == "error" else "🔄"
            if status_container:
                status_container.write(f"{icon} {label}")

    try:
        update_status("Preparando documento para OCR...")

        # Guardar una copia del archivo para depuración
        debug_dir = os.path.join(tempfile.gettempdir(), "mallo_debug")
        os.makedirs(debug_dir, exist_ok=True)
        debug_file_path = os.path.join(debug_dir, f"debug_{job_id}_{file_name}")

        with open(debug_file_path, "wb") as f:
            f.write(file_bytes)

        logger.info(f"Archivo de depuración guardado en: {debug_file_path}")

        # Sistema de procesamiento con verificación según tipo
        if file_type == "PDF":
            # Verificar que el PDF sea válido
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                page_count = len(reader.pages)
                sample_text = ""
                if page_count > 0:
                    sample_text = reader.pages[0].extract_text()[:100]
                logger.info(
                    f"PDF válido con {page_count} páginas. Muestra: {sample_text}"
                )

                # Codificar PDF en base64
                encoded_file = base64.b64encode(file_bytes).decode("utf-8")
                document = {
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{encoded_file}",
                }
            except Exception as e:
                logger.error(f"Error al validar PDF: {str(e)}")
                update_status(f"Error al validar PDF: {str(e)}", "error")
                return {"error": f"El archivo no es un PDF válido: {str(e)}"}
        elif file_type == "Imagen":
            # Optimizar imagen para mejores resultados
            try:
                optimized_bytes, mime_type = prepare_image_for_ocr(file_bytes)

                # Codificar en base64
                encoded_file = base64.b64encode(optimized_bytes).decode("utf-8")
                document = {
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{encoded_file}",
                }
            except Exception as e:
                logger.error(f"Error al procesar imagen: {str(e)}")
                update_status(f"Error al procesar imagen: {str(e)}", "error")
                return {"error": f"El archivo no es una imagen válida: {str(e)}"}
        elif file_type == "Texto":
            # Para archivos de texto, extraer contenido directamente
            try:
                # Intentar leer con diferentes codificaciones
                try:
                    text_content = file_bytes.decode("utf-8")
                    return {"text": text_content, "format": "text"}
                except UnicodeDecodeError:
                    # Intentar con otras codificaciones comunes
                    for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                        try:
                            text_content = file_bytes.decode(encoding)
                            return {"text": text_content, "format": "text"}
                        except UnicodeDecodeError:
                            continue

                # Si llegamos aquí, no pudimos decodificar el texto
                # Intentar enviar como documento plano
                update_status(
                    f"Convirtiendo documento de texto {file_name} para OCR..."
                )

                # Codificar en base64 y enviar como documento
                encoded_file = base64.b64encode(file_bytes).decode("utf-8")
                document = {
                    "type": "document_url",
                    "document_url": f"data:text/plain;base64,{encoded_file}",
                }
            except Exception as e:
                logger.error(f"Error al procesar documento de texto: {str(e)}")
                update_status(f"Error al procesar documento: {str(e)}", "error")
                return {"error": f"Error al procesar documento de texto: {str(e)}"}
        else:
            # Tipo de documento no soportado
            error_msg = f"Tipo de documento no soportado: {file_type}"
            logger.error(error_msg)
            update_status(error_msg, "error")
            return {"error": error_msg}

        update_status("Enviando documento a la API de Mistral...")

        # Configurar los headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # Preparar payload
        payload = {
            "model": ocr_config.get("model", "mistral-ocr-latest"),
            "document": document,
        }

        # Guardar payload para depuración (excluyendo contenido base64 por tamaño)
        debug_payload = {
            "model": payload["model"],
            "document": {
                "type": payload["document"]["type"],
                "content_size": len(encoded_file),
                "content_format": "base64",
            },
        }
        logger.info(f"Payload para OCR: {json.dumps(debug_payload)}")

        # Sistema de retry interno para la API de Mistral
        max_retries = 2
        retry_delay = 2
        last_error = None

        for retry in range(max_retries + 1):
            try:
                # Hacer la solicitud a Mistral OCR API
                response = requests.post(
                    "https://api.mistral.ai/v1/ocr",
                    json=payload,
                    headers=headers,
                    timeout=90,  # Timeout ampliado para documentos grandes
                )

                logger.info(f"Respuesta de OCR API - Estado: {response.status_code}")

                if response.status_code == 200:
                    try:
                        result = response.json()
                        # Guardar respuesta para depuración
                        with open(
                            os.path.join(
                                debug_dir, f"response_{job_id}_{file_name}.json"
                            ),
                            "w",
                        ) as f:
                            json.dump(result, f, indent=2)

                        update_status(
                            f"Documento {file_name} procesado exitosamente", "complete"
                        )

                        # Verificar existencia de contenido
                        if not result or (isinstance(result, dict) and not result):
                            return {
                                "error": "La API no devolvió contenido",
                                "raw_response": str(result),
                            }

                        # Extraer texto de la respuesta
                        extracted_content = extract_text_from_ocr_response(result)

                        if "error" in extracted_content:
                            update_status(
                                f"Error al extraer texto: {extracted_content['error']}",
                                "error",
                            )
                            return {
                                "error": extracted_content["error"],
                                "raw_response": result,
                            }

                        return extracted_content
                    except Exception as e:
                        error_message = f"Error al procesar respuesta JSON: {str(e)}"
                        logger.error(error_message)
                        # Guardar respuesta cruda para depuración
                        with open(
                            os.path.join(
                                debug_dir, f"raw_response_{job_id}_{file_name}.txt"
                            ),
                            "w",
                        ) as f:
                            f.write(response.text[:10000])  # Limitar tamaño
                        update_status(error_message, "error")
                        last_error = e
                elif response.status_code == 429:  # Rate limit
                    if retry < max_retries:
                        wait_time = retry_delay * (retry + 1)
                        logger.warning(
                            f"Rate limit alcanzado. Esperando {wait_time}s antes de reintentar..."
                        )
                        update_status(
                            f"Límite de tasa alcanzado. Reintentando en {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        error_message = "Límite de tasa alcanzado. No se pudo procesar después de reintentos."
                        logger.error(error_message)
                        update_status(error_message, "error")
                        return {
                            "error": error_message,
                            "raw_response": response.text,
                        }
                else:
                    error_message = f"Error en API OCR ({response.status_code}): {response.text[:500]}"
                    logger.error(error_message)
                    update_status(f"Error: {error_message}", "error")
                    last_error = Exception(error_message)
                    break
            except requests.exceptions.Timeout:
                if retry < max_retries:
                    wait_time = retry_delay * (retry + 1)
                    logger.warning(
                        f"Timeout al contactar API. Esperando {wait_time}s antes de reintentar..."
                    )
                    update_status(f"Timeout. Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    error_message = (
                        "Timeout al contactar API después de múltiples intentos."
                    )
                    logger.error(error_message)
                    update_status(error_message, "error")
                    return {"error": error_message}
            except Exception as e:
                if retry < max_retries:
                    wait_time = retry_delay * (retry + 1)
                    logger.warning(
                        f"Error: {str(e)}. Esperando {wait_time}s antes de reintentar..."
                    )
                    update_status(f"Error. Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    error_message = f"Error al procesar documento: {str(e)}"
                    logger.error(error_message)
                    update_status(f"Error: {error_message}", "error")
                    last_error = e
                    break

        # Si llegamos aquí después de reintentos, devolver último error
        return {
            "error": f"Error después de reintentos: {str(last_error)}",
            "details": traceback.format_exc(),
        }

    except Exception as e:
        error_message = f"Error general al procesar documento: {str(e)}"
        # Registrar el error completo en el log para depuración
        logger.error(error_message)
        logger.error(traceback.format_exc())

        # Mensaje de error más amigable para el usuario
        user_friendly_message = "Ocurrió un error al procesar el documento. Por favor, inténtalo de nuevo o prueba con otro documento."
        update_status(user_friendly_message, "error")

        # Devolver información detallada para depuración interna
        return {
            "error": user_friendly_message,
            "technical_details": error_message,
            "traceback": traceback.format_exc(),
        }


def manage_document_context():
    """
    Permite al usuario gestionar qué documentos mantener en el contexto actual
    con manejo seguro de actualización de estado
    """
    if "document_contents" in st.session_state and st.session_state.document_contents:
        st.write("Documentos en contexto actual:")

        # Crear checkboxes para cada documento
        docs_to_keep = {}
        for doc_name in st.session_state.document_contents:
            docs_to_keep[doc_name] = st.checkbox(
                f"{doc_name}", value=True, key=f"keep_{doc_name}"
            )

        # Botón para aplicar cambios
        if st.button("Actualizar contexto", key="update_context"):
            # Eliminar documentos no seleccionados
            docs_to_remove = [doc for doc, keep in docs_to_keep.items() if not keep]
            if docs_to_remove:
                for doc in docs_to_remove:
                    if doc in st.session_state.document_contents:
                        del st.session_state.document_contents[doc]
                    if doc in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.remove(doc)

                st.success(
                    f"Se eliminaron {len(docs_to_remove)} documentos del contexto."
                )
                # Usar sistema seguro de reinicio
                st.rerun()
            else:
                st.info("No se seleccionaron documentos para eliminar.")
    else:
        st.info("No hay documentos cargados en el contexto actual.")
