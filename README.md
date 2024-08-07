# MALLO: MultiAgent LLM Orchestrator

## Descripción
MALLO (MultiAgent LLM Orchestrator) es un sistema avanzado de orquestación de múltiples agentes de Modelos de Lenguaje de Gran Escala (LLMs). Diseñado para manejar consultas complejas, MALLO utiliza una variedad de agentes, desde modelos locales hasta asistentes especializados y APIs de terceros, para proporcionar respuestas precisas y contextuales.

## Características principales
- Integración con modelos locales de Ollama
- Uso de APIs de OpenAI y otros proveedores de LLM
- Asistentes especializados para dominios específicos
- Búsqueda web integrada utilizando DuckDuckGo
- Procesamiento de texto y análisis de sentimientos
- Manejo de archivos (PDF, DOCX) e imágenes

## Requisitos previos
- Python 3.8+
- Ollama instalado localmente (para modelos locales)
- Clave API de OpenAI (para servicios de OpenAI)

## Instalación
1. Clona el repositorio:
   ```
   git clone https://github.com/bladealex9848/MALLO.git
   cd MALLO
   ```

2. Crea un entorno virtual y actívalo:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

4. Configura tus claves API:
   Crea un archivo `.streamlit/secrets.toml` y añade tu clave API de OpenAI:
   ```
   OPENAI_API_KEY = "tu_clave_api_aqui"
   ```

## Uso
Para iniciar la aplicación, ejecuta:
```
streamlit run main.py
```

Luego, abre tu navegador y ve a `http://localhost:8501`.

## Estructura del proyecto
- `main.py`: Punto de entrada principal y interfaz de usuario de Streamlit
- `agents.py`: Implementación de varios agentes LLM
- `utilities.py`: Funciones de utilidad para procesamiento de texto y búsqueda web
- `config.yaml`: Configuración del proyecto

## Configuración
Ajusta la configuración en `config.yaml` según tus necesidades. Puedes modificar los modelos disponibles, las prioridades de procesamiento y otras opciones.

## Contribuir
Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios mayores antes de hacer un pull request.

## Licencia
[MIT License](https://opensource.org/licenses/MIT)

## Contacto
[Alexander Oviedo Fadul] - [alexander.oviedo.fadul@gmail.com]

Enlace del proyecto: [https://github.com/bladealex1844/MALLO](https://github.com/bladealex1844/MALLO)
```