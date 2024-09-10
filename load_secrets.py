import os
import toml
import json
from dotenv import load_dotenv

def load_secrets():
    secrets = {}
    
    # Intenta cargar desde .env (desarrollo local o servidor)
    load_dotenv()  # Esto cargará las variables de .env en os.environ
    
    # Intenta cargar desde .streamlit/secrets.toml (desarrollo local)
    try:
        with open('.streamlit/secrets.toml', 'r') as f:
            secrets.update(toml.load(f))
    except FileNotFoundError:
        pass

    # Intenta cargar desde /etc/secrets (Render)
    if os.path.exists('/etc/secrets'):
        for filename in os.listdir('/etc/secrets'):
            with open(os.path.join('/etc/secrets', filename), 'r') as f:
                if filename.endswith('.json'):
                    secrets.update(json.load(f))
                elif filename.endswith('.toml'):
                    secrets.update(toml.load(f))
                else:
                    # Asume que es un archivo de texto plano con un solo valor
                    secrets[filename] = f.read().strip()

    # Carga desde variables de entorno (incluyendo las cargadas desde .env)
    for key, value in os.environ.items():
        if key.startswith('MALLO_'):
            secrets[key[6:]] = value
        else:
            # Opcionalmente, puedes decidir incluir todas las variables de entorno
            # Descomenta la siguiente línea si quieres incluir todas las variables
            secrets[key] = value

    return secrets