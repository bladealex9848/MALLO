import os
import json
import toml
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_file(filename, search_paths):
    """Busca un archivo en las rutas especificadas."""
    for path in search_paths:
        file_path = Path(path) / filename
        if file_path.is_file():
            return file_path
    return None

def load_secret_file(file_path):
    """Carga un archivo de secretos basado en su extensión."""
    try:
        with open(file_path, 'r') as f:
            if file_path.suffix == '.json':
                return json.load(f)
            elif file_path.suffix == '.toml':
                return toml.load(f)
            else:
                return {file_path.name: f.read().strip()}
    except Exception as e:
        logger.error(f"Error al leer el archivo {file_path}: {str(e)}")
        return {}

def load_secrets():
    secrets = {}
    logger.info("Iniciando carga de secretos")

    # Cargar variables de entorno desde .env
    load_dotenv()
    logger.info("Variables de entorno cargadas desde .env (si existe)")

    # Definir rutas de búsqueda para archivos de secretos
    search_paths = [
        '.',
        '.streamlit',
        '/etc/secrets',
        '/opt/render/project/src/.streamlit',
        '/app/.streamlit'
    ]

    # Buscar y cargar secrets.toml
    secrets_toml = find_file('secrets.toml', search_paths)
    if secrets_toml:
        logger.info(f"Archivo secrets.toml encontrado en {secrets_toml}")
        secrets.update(load_secret_file(secrets_toml))
    else:
        logger.warning("No se encontró el archivo secrets.toml")

    # Buscar y cargar archivos en /etc/secrets
    etc_secrets = Path('/etc/secrets')
    if etc_secrets.is_dir():
        logger.info("Directorio /etc/secrets encontrado")
        for file_path in etc_secrets.iterdir():
            if file_path.is_file():
                logger.info(f"Procesando archivo: {file_path}")
                secrets.update(load_secret_file(file_path))

    # Cargar desde variables de entorno
    env_secrets = {k[6:]: v for k, v in os.environ.items() if k.startswith('MALLO_')}
    secrets.update(env_secrets)
    logger.info(f"{len(env_secrets)} variables de entorno MALLO_ cargadas")

    # Cargar secretos específicos si no se encontraron en los pasos anteriores
    required_secrets = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'TOGETHER_API_KEY']
    for secret in required_secrets:
        if secret not in secrets:
            env_value = os.getenv(secret)
            if env_value:
                secrets[secret] = env_value
                logger.info(f"Secreto {secret} cargado desde variables de entorno")
            else:
                logger.warning(f"Secreto requerido {secret} no encontrado")

    logger.info(f"Carga de secretos completada. {len(secrets)} secretos cargados.")
    return secrets

# Ejemplo de uso
if __name__ == "__main__":
    secrets = load_secrets()
    logger.info(f"Secretos cargados: {', '.join(secrets.keys())}")