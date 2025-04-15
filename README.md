![Logo de MALLO](https://raw.githubusercontent.com/bladealex9848/MALLO/main/assets/logo.jpg)

# MALLO: MultiAgent LLM Orchestrator

## Tabla de Contenidos
1. [Descripción](#descripción)
2. [Características Principales](#características-principales)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Requisitos Previos](#requisitos-previos)
5. [Instalación](#instalación)
6. [Configuración](#configuración)
7. [Uso](#uso)
8. [Evaluación Ética y de Cumplimiento](#evaluación-ética-y-de-cumplimiento)
9. [TIPS y Trucos](#tips-y-trucos)
10. [Arquitectura del Sistema](#arquitectura-del-sistema)
11. [Componentes Principales](#componentes-principales)
12. [Flujo de Trabajo](#flujo-de-trabajo)
13. [APIs y Servicios Integrados](#apis-y-servicios-integrados)
14. [Manejo de Errores y Logging](#manejo-de-errores-y-logging)
15. [Optimización y Caché](#optimización-y-caché)
16. [Pruebas](#pruebas)
17. [Contribución](#contribución)
18. [Registro de Cambios](#registro-de-cambios)
19. [ExperimentaLABs](#experimentalabs)
20. [Roadmap](#roadmap)
21. [Licencia](#licencia)
22. [Contacto](#contacto)

## Descripción

MALLO (MultiAgent LLM Orchestrator) es una plataforma avanzada de orquestación de modelos de lenguaje que implementa una arquitectura distribuida basada en agentes. Diseñada específicamente para el procesamiento de consultas complejas, MALLO utiliza un sistema de selección dinámica de agentes que combina:

- Modelos de lenguaje locales para procesamiento de baja latencia
- APIs de modelos en la nube para tareas complejas
- Asistentes especializados para dominios específicos
- Sistemas de búsqueda y enriquecimiento de contexto

La arquitectura del sistema se basa en principios de diseño modular y extensible, permitiendo la integración seamless de nuevos modelos y servicios. El núcleo del sistema implementa algoritmos avanzados de evaluación de complejidad y selección de agentes, optimizando la relación entre precisión, latencia y consumo de recursos.

## Características Principales

### 1. Arquitectura Multi-Agente Avanzada
- **Procesamiento Distribuido**:
  - Pipeline de procesamiento paralelo para múltiples agentes
  - Sistema de votación y consenso para respuestas múltiples
  - Mecanismos de fallback y recuperación automática

- **Integración de Modelos**:
  - Modelos locales vía Ollama para baja latencia (offline-capable)
  - APIs cloud premium (OpenAI, Anthropic, Groq)
  - Proveedores especializados (Together, DeepInfra, DeepSeek)
  - Modelos open-source optimizados (Mistral, Cohere)

### 2. Sistema de Procesamiento Inteligente
- **Análisis de Complejidad**:
  - Evaluación heurística de consultas
  - Detección automática de dominio y contexto
  - Selección dinámica de agentes basada en métricas

- **Optimización de Recursos**:
  - Sistema de caché multinivel con validación
  - Balanceo dinámico de carga entre agentes
  - Gestión inteligente de cuotas y límites de API

### 3. Capacidades Avanzadas de Búsqueda y Contextualización
- **Motor de Búsqueda Multi-Fuente**:
  - Integración primaria con API de YOU
  - Fallback a Tavily para búsquedas especializadas
  - Sistema de respaldo con DuckDuckGo
  - Agregación y deduplicación de resultados

- **Procesamiento de Contexto**:
  - Análisis semántico de consultas
  - Extracción de entidades y relaciones
  - Generación de embeddings para búsqueda contextual

### 4. Framework de Evaluación y Mejora
- **Sistema de Evaluación Ética**:
  - Detección automática de sesgos
  - Validación de privacidad y seguridad
  - Alineación con principios éticos configurables

- **Mecanismos de Feedback**:
  - Evaluación continua de calidad de respuestas
  - Sistema de aprendizaje basado en retroalimentación
  - Métricas de rendimiento en tiempo real

### 5. Interfaz y Monitoreo
- **UI Moderna y Responsive**:
  - Framework Streamlit con diseño optimizado
  - Componentes dinámicos de visualización
  - Sistema de progreso en tiempo real

- **Logging y Observabilidad**:
  - Logging estructurado con niveles configurables
  - Métricas Prometheus para monitoreo
  - Sistema de alertas para eventos críticos

### 6. Configuración y Extensibilidad
- **Sistema de Configuración Robusto**:
  - Configuración centralizada via YAML
  - Soporte para múltiples entornos
  - Hot-reload de configuraciones

- **Arquitectura Extensible**:
  - APIs bien documentadas para nuevos agentes
  - Sistema de plugins para funcionalidades adicionales
  - Hooks para personalización de comportamiento

### 7. Seguridad y Cumplimiento
- **Protección de Datos**:
  - Encriptación en tránsito y en reposo
  - Sanitización de entradas y salidas
  - Gestión segura de credenciales

- **Auditoría y Compliance**:
  - Registro detallado de operaciones
  - Trazabilidad de decisiones del sistema
  - Reportes de cumplimiento normativo

### 8. Optimización de Rendimiento
- **Sistema de Caché Avanzado**:
  - Caché multinivel (memoria, disco, distribuido)
  - Políticas de invalidación inteligentes
  - Compresión y optimización de almacenamiento

- **Gestión de Recursos**:
  - Límites de consumo configurables
  - Balanceo de carga automático
  - Recuperación graceful ante fallos

## Estructura del Proyecto

```
MALLO/
│
├── agents.py                   # Implementación de la clase AgentManager y lógica de agentes
├── config.yaml                 # Configuración general del sistema
├── main.py                     # Punto de entrada principal y UI de Streamlit
├── README.md                   # Documentación del proyecto (este archivo)
├── utilities.py                # Funciones de utilidad y helpers
├── model_speeds.json           # Índice de velocidad de modelos locales y en la nube
├── CHANGELOG.md                # Registro de cambios y versiones
├── mallo_enhancer.py           # Lógica experimental para mejorar y evaluar respuestas
│
├── .streamlit/
│   └── secrets.toml            # Almacenamiento seguro de claves API (no incluido en el repositorio)
│
├── tests/                      # Directorio para pruebas unitarias y de integración
│
├── logs/                       # Directorio para archivos de log
│
├── assets/                     # Directorio para recursos estáticos (imágenes, estilos, etc.)
|
├── tools/                      # Directorio utilidades externas
│   └── test_model_speeds.py    # Pequeño script para medir la velocidad de los modelos
│
├── docs/
│   ├── informes/               # Informes técnicos, de investigación y otros documentos relacionados con el proyecto
│   │
│   └── ejemplos_consultas/     # Ejemplos de consultas y respuestas para demostración y prueba
|
└── requirements.txt            # Dependencias del proyecto
```

## Requisitos Previos

- Python 3.8+
- Ollama instalado localmente (para modelos locales)
- Claves API para servicios externos (OpenAI, Groq, Together, etc.)
- Conexión a Internet (para búsqueda web y APIs externas)

## Instalación

1. Clona el repositorio:
   ```
   git clone https://github.com/bladealex9848/MALLO.git
   cd MALLO
   ```

2. Crea y activa un entorno virtual:
   ```
   python3 -m venv venv
   source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
   ```

3. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

4. Configura tus claves API:
   Crea un archivo `.streamlit/secrets.toml` y añade tus claves API:
   ```toml
   OPENAI_API_KEY = "tu_clave_openai_aqui"
   GROQ_API_KEY = "tu_clave_groq_aqui"
   TOGETHER_API_KEY = "tu_clave_together_aqui"   
   DEEPINFRA_API_KEY="tu_clave_deepinfra_aqui"
   ANTHROPIC_API_KEY="tu_clave_anthropic_aqui"
   DEEPSEEK_API_KEY="tu_clave_deepseek_aqui"
   MISTRAL_API_KEY="tu_clave_mistral_aqui"
   COHERE_API_KEY="tu_clave_cohere_aqui"
   REPLICATE_API_TOKEN= "tu_clave_replicate_aqui"
   OPENROUTER_API_KEY= "tu_clave_openrouter_aqui"
   # Añade otras claves API según sea necesario
   ```

## Configuración

El archivo `config.yaml` contiene la configuración principal del sistema. Aquí puedes ajustar:

- Modelos disponibles para cada proveedor de LLM
- Prioridades de procesamiento
- Umbrales de complejidad para la selección de agentes
- Configuración de asistentes especializados
- Parámetros de utilidades como búsqueda web y análisis de imágenes

Ejemplo de configuración:

```yaml
ollama:
  base_url: "http://localhost:11434"
  default_model: "gemma2:2b"
  models:
    - "gemma2:2b"
    - "llava:latest"

openai:
  default_model: "gpt-4.1-nano"
  models:  
    - "gpt-4.1-nano"

# ... (otras configuraciones)

thresholds:
  moa_complexity: 0.8
  local_complexity: 0.5
  web_search_complexity: 0.4
```

## Uso

Para iniciar la aplicación, ejecuta:
```
streamlit run main.py
```

Luego, abre tu navegador y ve a `http://localhost:8501`.

La interfaz de usuario te permitirá:
1. Ingresar consultas en lenguaje natural.
2. Ver las respuestas generadas por el sistema.
3. Observar el progreso del procesamiento en tiempo real.
4. Explorar detalles sobre el procesamiento de cada consulta, incluyendo la evaluación ética.
5. Acceder a información sobre el estado del sistema y rendimiento de los agentes.
6. Ver la evaluación ética y de cumplimiento de cada respuesta generada.
7. Ver el contexto de la conversación actual, incluyendo el historial de preguntas y respuestas.

La interfaz ahora incluye:
- Un chat limpio que muestra solo las preguntas y respuestas.
- Una sección expandible de "Detalles del proceso" que muestra información sobre el procesamiento interno.
- Una sección expandible de "Evaluación Ética y de Cumplimiento" que muestra los resultados de la evaluación ética de cada respuesta.
- Una sección expandible de "Contexto de la conversación" que muestra el contexto de la conversación actual, incluyendo el historial de preguntas y respuestas.

## Evaluación Ética y de Cumplimiento en MALLO

MALLO implementa un sistema integral de evaluación ética y cumplimiento normativo fundamentado en el Acuerdo PCSJA24-12243 del 16 de diciembre de 2024 y los principios del documento CONPES 4144 del 14 de febrero de 2025, que establecen:

### Sistema de Evaluación Automática

La evaluación se realiza sobre los siguientes criterios fundamentales:

1.  **Primacía de Derechos Fundamentales**
    * Respeto y protección de derechos fundamentales.
    * Promoción de garantías constitucionales.
    * Alineación con mandatos superiores.
2.  **Regulación Ética**
    * Cumplimiento de mandatos constitucionales y legales.
    * Adherencia al Código Iberoamericano de Ética.
    * Uso razonable de sistemas de IA.
3.  **Transparencia y Responsabilidad**
    * Identificación clara del uso de IA.
    * Explicación detallada de fuentes y metodologías.
    * Documentación de procesos y decisiones.
4.  **Seguridad y Privacidad**
    * Protección de datos personales.
    * Prevención de sesgos y discriminación.
    * Mantenimiento de la confidencialidad de la información.
### Proceso de Mejora y Verificación

1.  **Control y Verificación Humana**
    * Revisión detallada de resultados.
    * Validación de fuentes y referencias.
    * Supervisión de impactos y consecuencias.
2.  **Proceso de Mejora**
    * Intervención del asistente especializado de transformación digital (ID: 'asst\_F33bnQzBVqQLcjveUTC14GaM').
    * Refinamiento de respuestas según criterios éticos.
    * Validación de cumplimiento normativo.
### Cumplimiento Normativo

1.  **Marco Legal Aplicable**
    * Sentencia T-323 de 2024 de la Corte Constitucional.
    * Directrices internacionales de gobernanza de IA.
    * Normativas específicas del sector judicial.
    * Lineamientos del documento CONPES 4144 de 2025.
2.  **Salvaguardas Específicas**
    * Protección de datos personales y privacidad.
    * Prevención de sesgos y discriminación.
    * Garantía de acceso a la justicia.
### Documentación y Transparencia

1.  **Registro de Evaluaciones**
    * Documentación detallada del proceso evaluativo.
    * Identificación de herramientas y modelos utilizados.
    * Registro de modificaciones y mejoras realizadas.
2.  **Acceso a la Información**
    * Disponibilidad de resultados de evaluación.
    * Explicación de criterios aplicados.
    * Trazabilidad de decisiones y procesos.
Esta implementación asegura que MALLO opere bajo los más altos estándares éticos y legales, proporcionando un servicio que no solo es técnicamente competente, sino también responsable y transparente en su funcionamiento.

## TIPS y Trucos

### Palabras Clave para Prompts Especializados

MALLO está diseñado para detectar automáticamente el tipo de consulta y aplicar prompts especializados. Sin embargo, puedes ayudar al sistema a seleccionar el prompt más adecuado incluyendo ciertas palabras clave en tu consulta. Aquí te presentamos algunas sugerencias:

1. **Matemáticas**: 
   Palabras clave: "cálculo", "ecuación", "álgebra", "geometría", "estadística", "probabilidad"

2. **Programación**: 
   Palabras clave: "código", "algoritmo", "función", "debug", "software", "desarrollo"

3. **Legal**: 
   Palabras clave: "ley", "legislación", "jurídico", "contrato", "demanda", "jurisprudencia"

4. **Científico**: 
   Palabras clave: "experimento", "hipótesis", "teoría", "investigación", "método científico"

5. **Histórico**: 
   Palabras clave: "época", "siglo", "período", "civilización", "evento histórico"

6. **Filosófico**: 
   Palabras clave: "ética", "metafísica", "epistemología", "lógica", "existencialismo"

7. **Contexto Colombiano**: 
   Palabras clave: "Colombia", "Bogotá", "Medellín", "Andes", "Caribe", "cultura colombiana"

8. **Cultural**: 
   Palabras clave: "arte", "literatura", "música", "tradición", "costumbres"

9. **Político**: 
   Palabras clave: "gobierno", "elecciones", "política pública", "partidos políticos", "constitución"

10. **Económico**: 
    Palabras clave: "mercado", "inflación", "PIB", "finanzas", "economía"

Incluir estas palabras clave en tu consulta puede ayudar a MALLO a aplicar el prompt especializado más adecuado, lo que potencialmente mejorará la precisión y relevancia de las respuestas.

### Mejores Prácticas

1. **Sé Específico**: Cuanto más específica sea tu consulta, mejor podrá MALLO seleccionar los agentes y prompts adecuados.

2. **Contexto**: Proporciona contexto adicional cuando sea necesario. Por ejemplo, si estás haciendo una pregunta sobre historia colombiana, menciona explícitamente "en el contexto de la historia de Colombia".

3. **Usa Terminología Relevante**: Incluir términos técnicos o específicos del campo puede ayudar a MALLO a identificar mejor el dominio de conocimiento.

4. **Solicita Análisis**: Si necesitas un análisis más profundo, puedes incluir frases como "analiza en profundidad" o "proporciona un análisis detallado".

5. **Pide Múltiples Perspectivas**: Si deseas que MALLO utilice múltiples agentes, puedes solicitar explícitamente "diferentes perspectivas" o "análisis desde varios ángulos".

Recuerda que MALLO está diseñado para ser intuitivo y adaptativo, pero proporcionar estas pistas adicionales puede ayudar a obtener respuestas más precisas y relevantes.

## Arquitectura del Sistema

MALLO utiliza una arquitectura de microservicios basada en agentes, donde cada agente (modelo de lenguaje o servicio especializado) puede procesar consultas de manera independiente. El componente central, `AgentManager`, orquesta estos agentes basándose en la complejidad de la consulta y la disponibilidad de recursos.

### Componentes Clave:
1. **Orquestador de Agentes**: Selecciona y gestiona los agentes apropiados para cada consulta.
2. **Evaluador de Complejidad**: Analiza las consultas para determinar su complejidad y requisitos.
3. **Sistema de Caché**: Almacena y recupera respuestas para consultas frecuentes.
4. **Módulo de Búsqueda Web**: Enriquece las respuestas con información actual de la web.
5. **Interfaz de Usuario**: Proporciona una experiencia interactiva para los usuarios.

## Componentes Principales

### agents.py
- **Clase AgentManager**: 
  - Gestiona la inicialización y selección de agentes.
  - Implementa la lógica de procesamiento de consultas.
  - Maneja la integración con diferentes APIs de LLM.

### utilities.py
- Funciones para evaluación de complejidad de consultas.
- Implementación de búsqueda web.
- Sistema de caché para respuestas.
- Funciones de logging y manejo de errores.

### main.py
- Implementa la interfaz de usuario con Streamlit.
- Gestiona el flujo principal de la aplicación.
- Procesa las entradas del usuario y muestra las respuestas.

## Flujo de Trabajo

1. El usuario ingresa una consulta a través de la interfaz de Streamlit.
2. La consulta se evalúa para determinar su complejidad y requisitos.
3. Se selecciona el agente o conjunto de agentes más apropiados.
4. Si es necesario, se realiza una búsqueda web para enriquecer el contexto.
5. Los agentes seleccionados procesan la consulta.
6. Se evalúa y refina la respuesta generada.
7. La respuesta final se presenta al usuario junto con detalles del proceso.

## APIs y Servicios Integrados

MALLO integra varios servicios de LLM y APIs, incluyendo:
- OpenAI
- Groq
- Together
- DeepInfra
- Anthropic
- DeepSeek
- Mistral
- Cohere
- Ollama
- OpenRouter

Cada servicio se inicializa y gestiona a través de la clase `AgentManager`, permitiendo una fácil expansión a nuevos proveedores en el futuro.

## Manejo de Errores y Logging

El sistema implementa un robusto sistema de logging y manejo de errores:
- Los errores se registran en el archivo `mallo.log`.
- Se utilizan diferentes niveles de logging (INFO, WARNING, ERROR) para categorizar los eventos.
- Los errores críticos se muestran al usuario a través de la interfaz de Streamlit.

## Optimización y Caché

Para mejorar el rendimiento y reducir la carga en las APIs externas, MALLO implementa:
- Un sistema de caché para almacenar respuestas frecuentes.
- Evaluación de complejidad para evitar el uso innecesario de recursos en consultas simples.
- Selección inteligente de agentes basada en la disponibilidad y rendimiento histórico.

## Pruebas

El proyecto incluye un conjunto de pruebas unitarias y de integración en el directorio `tests/`. Para ejecutar las pruebas:

```
python -m unittest discover tests
```

## Contribución

Las contribuciones son bienvenidas. Por favor, sigue estos pasos:

1. Haz fork del repositorio.
2. Crea una nueva rama (`git checkout -b feature/AmazingFeature`).
3. Realiza tus cambios y haz commit (`git commit -m 'Add some AmazingFeature'`).
4. Push a la rama (`git push origin feature/AmazingFeature`).
5. Abre un Pull Request.

## Registro de Cambios

Consulta [CHANGELOG.md](CHANGELOG.md) para ver el historial detallado de cambios del proyecto.

## ExperimentaLABs

MALLO incluye ahora características experimentales diseñadas para mejorar la calidad de las instrucciones y respuestas mediante un proceso de reflexión iterativa. Estas características están disponibles a través de la función `process_user_input_experimental`.

### MALLOEnhancer

MALLOEnhancer es una nueva clase que implementa un enfoque de reflexión y mejora utilizando múltiples modelos de lenguaje. Esta característica está diseñada para:

- Mejorar iterativamente las instrucciones y respuestas.
- Utilizar múltiples APIs de LLM para obtener diversas perspectivas.
- Evaluar la calidad de las instrucciones y respuestas mediante métricas como IFD y r-IFD.

Para habilitar estas características experimentales, use la bandera `--experimental` al iniciar la aplicación:

```
streamlit run main.py -- --experimental
```

## Roadmap

Nuestros planes futuros para MALLO incluyen:

1. Refinar y optimizar el proceso de reflexión iterativa en MALLOEnhancer.
2. Mejorar la integración y el uso eficiente de múltiples APIs de LLM.
3. Desarrollar métricas más avanzadas para la evaluación de la calidad de instrucciones y respuestas.
4. Implementar un sistema de aprendizaje continuo basado en el feedback de los usuarios.
5. Explorar la posibilidad de fine-tuning de modelos basados en los resultados de MALLOEnhancer.

Estas características experimentales representan nuestro compromiso con la mejora continua de MALLO y la exploración de nuevas formas de optimizar la interacción entre humanos y LLMs.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - vea el archivo [LICENSE](LICENSE) para más detalles.

## Contacto

Alexander Oviedo Fadul

[GitHub](https://github.com/bladealex9848) | [Website](https://alexanderoviedofadul.dev/) | [Instagram](https://www.instagram.com/alexander.oviedo.fadul) | [Twitter](https://twitter.com/alexanderofadul) | [Facebook](https://www.facebook.com/alexanderof/) | [WhatsApp](https://api.whatsapp.com/send?phone=573015930519&text=Hola%20!Quiero%20conversar%20contigo!) | [LinkedIn](https://www.linkedin.com/in/alexander-oviedo-fadul/)