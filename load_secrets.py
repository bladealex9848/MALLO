import os
import json
import toml
import logging
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_secrets():
    secrets = {}
    logger.info("Iniciando carga de secretos")

    # Cargar variables de entorno desde .env
    load_dotenv()
    logger.info("Variables de entorno cargadas desde .env (si existe)")

    # Definir rutas de búsqueda para archivos de secretos
    search_paths = [
        '/opt/render/.streamlit',
        '/opt/render/project/src/.streamlit',
        '/app/.streamlit',
        '.',
        '.streamlit',
        '/etc/secrets',
    ]

    # Buscar y cargar secrets.toml
    for path in search_paths:
        secrets_path = Path(path) / 'secrets.toml'
        if secrets_path.is_file():
            logger.info(f"Archivo secrets.toml encontrado en {secrets_path}")
            try:
                with open(secrets_path, 'r') as f:
                    secrets.update(toml.load(f))
                break  # Si se encuentra y carga correctamente, salimos del bucle
            except Exception as e:
                logger.error(f"Error al leer {secrets_path}: {str(e)}")

    # Cargar desde /etc/secrets si existe
    etc_secrets = Path('/etc/secrets')
    if etc_secrets.is_dir():
        logger.info("Directorio /etc/secrets encontrado")
        for file_path in etc_secrets.iterdir():
            if file_path.is_file():
                logger.info(f"Procesando archivo: {file_path}")
                try:
                    with open(file_path, 'r') as f:
                        if file_path.suffix == '.json':
                            secrets.update(json.load(f))
                        elif file_path.suffix == '.toml':
                            secrets.update(toml.load(f))
                        else:
                            secrets[file_path.name] = f.read().strip()
                except Exception as e:
                    logger.error(f"Error al leer {file_path}: {str(e)}")

    # Cargar desde variables de entorno
    for key, value in os.environ.items():
        if key.startswith('MALLO_'):
            secrets[key[6:]] = value
        elif key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'TOGETHER_API_KEY']:
            secrets[key] = value

    logger.info(f"Carga de secretos completada. {len(secrets)} secretos cargados.")
    
    # Actualizar st.secrets de forma segura
    for key, value in secrets.items():
        if key not in st.secrets:
            st.secrets[key] = value

def get_secret(key):
    """Obtiene un secreto específico."""
    return st.secrets.get(key)