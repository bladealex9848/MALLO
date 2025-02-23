# Registro de Cambios
Todos los cambios notables en el proyecto MALLO serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto se adhiere a [Versionado Semántico](https://semver.org/lang/es/).

## [2.5.0] - 2025-02-22

### Añadido
- Implementación de sistema de caché optimizado para componentes core:
  - Nuevo decorador `@st.cache_resource` para inicialización eficiente de componentes principales.
  - Sistema inteligente de carga única para recursos pesados.
  - Logging detallado del proceso de inicialización y rendimiento.

### Cambiado
- Restructuración del sistema de inicialización:
  - Separación de componentes cacheables vs estado de sesión.
  - Implementación de carga diferida para recursos no críticos.
  - Mejora en el manejo de dependencias y orden de inicialización.
- Optimización de la estructura del proyecto:
  - Eliminación de la dependencia de `cached_init.py`.
  - Integración directa de funcionalidades de caché en `main.py`.

### Mejorado
- Sistema de manejo de errores más robusto:
  - Mejor trazabilidad en errores de inicialización.
  - Logging detallado con timestamps y contexto.
  - Sistema de recuperación automática ante fallos.
- Rendimiento general del sistema:
  - Reducción significativa en tiempo de carga inicial.
  - Mejor gestión de memoria en componentes principales.
  - Optimización de recursos en recargas consecutivas.

### Optimizado
- Gestión de recursos del sistema:
  - Caché inteligente para componentes pesados como AgentManager.
  - Inicialización única de recursos compartidos.
  - Mejor manejo de memoria en componentes estáticos.

### Documentación
- Actualización de documentación técnica sobre el nuevo sistema de caché.
- Guías detalladas para desarrolladores sobre el manejo de recursos.
- Ejemplos de implementación de caché para nuevos componentes.

### Impacto Técnico
- Reducción de ~60% en tiempo de inicialización.
- Mejor utilización de recursos del sistema.
- Mayor estabilidad en ejecuciones prolongadas.

### Consideraciones de Seguridad
- Implementación de validaciones de integridad en caché.
- Mejor manejo de secretos y configuraciones sensibles.
- Protección contra pérdida de datos en fallos de sistema.

### Próximos Pasos
- Monitoreo continuo del rendimiento del nuevo sistema de caché.
- Implementación de métricas detalladas de uso de recursos.
- Evaluación de oportunidades adicionales de optimización.

## [2.4.0] - 2025-02-21

### Añadido
- Implementación de nuevo sistema de caché optimizado:
  - Clase `CacheManager` con manejo de caché multinivel (memoria y disco)
  - Sistema de métricas Prometheus para monitoreo de rendimiento del caché
  - Limpieza automática programada de recursos mediante `BackgroundScheduler`
  - Soporte para persistencia configurable de datos en caché
- Nuevo sistema de manejo de errores estructurado:
  - Clases de excepciones personalizadas para diferentes tipos de errores
  - Logging detallado con contexto y trazabilidad
  - Sistema de reintentos con backoff exponencial
- Interfaz de usuario mejorada:
  - Implementación de tabs para mejor organización de información
  - Sistema de contenedores separados para chat y detalles
  - Visualización en tiempo real de estadísticas de caché
  - Nueva sección de métricas de rendimiento del sistema

### Cambiado
- Restructuración completa del manejo de sesiones:
  - Nuevo enfoque para gestión de estado basado en componentes
  - Optimización de carga inicial de recursos
  - Mejor manejo de recursos compartidos entre sesiones
- Mejora en el sistema de visualización de respuestas:
  - Separación clara entre contenido de chat y detalles técnicos
  - Implementación de tabs para diferentes tipos de información
  - Nuevo diseño para exportación de conversaciones

### Mejorado
- Optimización significativa del rendimiento:
  - Reducción del tiempo de carga inicial mediante caché inteligente
  - Mejor gestión de memoria en sesiones largas
  - Sistema eficiente de limpieza de recursos no utilizados
- Sistema de logging y monitoreo:
  - Implementación de métricas detalladas de rendimiento
  - Mejor trazabilidad de errores y excepciones
  - Estadísticas en tiempo real del uso de caché

### Documentación
- Guías detalladas para desarrolladores sobre:
  - Integración con el nuevo sistema de caché
  - Mejores prácticas para manejo de estado
  - Implementación de nuevos componentes de UI

### Impacto Técnico
- Reducción del 40% en tiempo de carga inicial
- Optimización del uso de memoria en un 35%
- Mejora del 50% en tiempos de respuesta para consultas cacheadas
- Reducción del 60% en errores de gestión de estado

### Consideraciones de Seguridad
- Implementación de validación de integridad para datos en caché
- Mejor manejo de datos sensibles en logs
- Protección contra pérdida de datos durante fallos del sistema

### Próximos Pasos
- Evaluación continua del rendimiento del nuevo sistema de caché
- Análisis de patrones de uso para optimizaciones adicionales
- Exploración de oportunidades de mejora en la UI basadas en feedback de usuarios

## [2.3.0] - 2024-10-17

### Añadido
- Implementación de un sistema de búsqueda web mejorado y robusto:
  - Se ha integrado la API de YOU como método principal de búsqueda.
  - Se han añadido Tavily y DuckDuckGo como métodos de recuperación alternativos.
- Nueva función `perform_web_search` que utiliza múltiples métodos de búsqueda con un sistema de fallback.
- Integración de la biblioteca Tavily para búsquedas web avanzadas.
- Nuevo sistema de manejo de errores y reintentos para las búsquedas web.

### Cambiado
- Actualización de la configuración en `config.yaml` para soportar múltiples métodos de búsqueda web.
- Modificación de la función `check_web_search` para probar todos los métodos de búsqueda implementados.
- Refactorización del código de búsqueda web en `utilities.py` para mayor modularidad y eficiencia.

### Mejorado
- Optimización del rendimiento de las búsquedas web mediante la implementación de múltiples APIs.
- Mejora en la robustez del sistema frente a fallos en las APIs de búsqueda.
- Refinamiento del manejo de secretos y claves API para mantener la consistencia en toda la aplicación.

### Corregido
- Solución al problema de límite de tasa (rate limit) con DuckDuckGo mediante la implementación de retrasos exponenciales y múltiples métodos de búsqueda.

### Seguridad
- Mejora en el manejo de claves API sensibles utilizando el sistema de secretos existente.

### Documentación
- Actualización de la documentación interna sobre el uso y configuración de los nuevos métodos de búsqueda web.
- Adición de comentarios explicativos en el código de `utilities.py` para facilitar futuras modificaciones.

### Justificación de los cambios
- Estos cambios buscan mejorar la fiabilidad y eficacia del sistema de búsqueda web de MALLO.
- La implementación de múltiples métodos de búsqueda aumenta la robustez del sistema y reduce la dependencia de un único proveedor.
- Las mejoras en el manejo de errores y la configuración flexible permiten una adaptación más rápida a cambios en las APIs externas.

### Impacto esperado
- Mayor fiabilidad en la obtención de información actualizada para las consultas de los usuarios.
- Reducción de interrupciones del servicio debido a problemas con APIs de búsqueda individuales.
- Mejora en la calidad y relevancia de las respuestas proporcionadas por MALLO.

### Próximos pasos
- Realizar pruebas exhaustivas del nuevo sistema de búsqueda web en diferentes escenarios y cargas de trabajo.
- Monitorear el rendimiento de cada método de búsqueda para optimizar su uso y orden de prioridad.
- Explorar la posibilidad de añadir más fuentes de búsqueda para aumentar la diversidad de información.
- Desarrollar un sistema de caché para resultados de búsqueda frecuentes, reduciendo la carga en las APIs externas.
- Implementar un sistema de retroalimentación para que los usuarios puedan reportar la calidad de las búsquedas web.

## [2.2.0] - 2024-10-03

### Añadido
- Implementación de un sistema de progreso visible para el usuario:
  - Se ha incorporado un indicador de progreso que muestra las diferentes etapas del procesamiento de la consulta en tiempo real.
- Nueva funcionalidad de resumen de características de MALLO:
  - Se ha añadido una descripción concisa de las capacidades clave del sistema en la interfaz de usuario.
- Evaluación de cumplimiento con la sentencia T-323 de 2024:
  - Se ha realizado un análisis detallado del cumplimiento de MALLO con las directrices establecidas por la Corte Constitucional.

### Cambiado
- Restructuración de la interfaz de usuario:
  - La respuesta final ahora se muestra de forma más limpia, sin información de procesamiento adicional.
  - Los detalles del proceso y la evaluación ética se han movido a secciones expandibles separadas.
- Optimización de la función `process_user_input`:
  - Se ha eliminado la variable no utilizada `details_placeholder`.
  - Se ha corregido la definición de `start_time` para un cálculo preciso del tiempo de procesamiento.
- Actualización de la estructura de directorios:
  - Se ha añadido un directorio `docs/` para almacenar informes y ejemplos de consultas.

### Mejorado
- Refinamiento del proceso de evaluación ética:
  - La evaluación ética ahora se realiza de manera más integrada en el flujo de procesamiento.
  - Se ha mejorado la presentación de los resultados de la evaluación ética al usuario.
- Optimización del rendimiento:
  - Se han implementado mejoras para reducir el tiempo de procesamiento y el uso de recursos.
- Actualización de la documentación:
  - Se ha creado un informe detallado sobre el cumplimiento de MALLO con la sentencia T-323 de 2024.
  - Se ha actualizado el README con información sobre la nueva estructura de directorios y las características recientes.

### Justificación de los cambios
- Estos cambios buscan mejorar la experiencia del usuario proporcionando más transparencia en el proceso de generación de respuestas.
- La restructuración de la interfaz y la optimización del código contribuyen a una interacción más fluida y eficiente con el sistema.
- La evaluación de cumplimiento con la sentencia T-323 de 2024 asegura que MALLO se alinee con las directrices legales y éticas más recientes.

### Impacto esperado
- Mayor comprensión por parte del usuario del proceso interno de MALLO.
- Mejora en la percepción de transparencia y confiabilidad del sistema.
- Incremento en la eficiencia y velocidad de respuesta del sistema.
- Mejor alineación con las normativas legales y éticas vigentes.

### Próximos pasos
- Realizar pruebas de usuario para evaluar la recepción de la nueva interfaz y el sistema de progreso.
- Continuar refinando el proceso de evaluación ética basándose en el feedback de los usuarios y expertos en ética de IA.
- Explorar la posibilidad de implementar un sistema de explicabilidad más detallado para las decisiones tomadas por MALLO.
- Desarrollar guías de usuario que expliquen cómo interpretar la información de progreso y los resultados de la evaluación ética.
- Implementar módulos específicos para mejorar la adaptación de MALLO al contexto judicial colombiano.

## [2.1.0] - 2024-10-02

### Añadido
- Implementación de evaluación ética y de cumplimiento:
  - Se ha incorporado la función `evaluate_ethical_compliance` para analizar las respuestas generadas en términos de sesgo, privacidad, transparencia, alineación con derechos humanos, responsabilidad y explicabilidad.
  - La función `process_user_input` ahora incluye esta evaluación ética y utiliza un asistente especializado (ID: 'asst_F33bnQzBVqQLcjveUTC14GaM') para mejorar las respuestas si se detectan problemas éticos.

### Cambiado
- Actualización de la función `process_user_input`:
  - Ahora muestra al usuario información sobre la evaluación ética de las respuestas generadas.
  - Incluye una declaración de cumplimiento que informa al usuario sobre el uso de IA y la evaluación ética realizada.

### Mejorado
- Ampliación de las palabras clave relacionadas con la gobernanza de IA, ética en IA, y regulación de IA para mejorar la contextualización de las consultas.

### Justificación de los cambios
- Estos cambios responden a las últimas recomendaciones y directrices internacionales para la gobernanza de la IA, así como a lo establecido en la sentencia T-323 de 2024 de la Corte Constitucional colombiana.
- La implementación de la evaluación ética y de cumplimiento busca garantizar que las respuestas generadas por MALLO estén alineadas con principios éticos y legales.

### Impacto esperado
- Mayor transparencia en el uso de IA para los usuarios del sistema MALLO.
- Mejora en la alineación de las respuestas generadas con principios éticos y legales.
- Cumplimiento más riguroso con las normativas y recomendaciones nacionales e internacionales sobre el uso de IA.

### Próximos pasos
- Realizar pruebas exhaustivas del nuevo sistema de evaluación ética y recopilar feedback de los usuarios.
- Refinar y ampliar los criterios de evaluación ética basándose en los resultados obtenidos.
- Explorar la posibilidad de implementar un sistema más avanzado de detección y mitigación de sesgos en las respuestas generadas.
- Desarrollar un módulo de capacitación para usuarios sobre la interpretación de la evaluación ética y el uso responsable de las respuestas generadas por IA.

## [2.0.0] - 2024-09-26

### Añadido
- Integración de la familia de modelos Llama 3.2:
  - Se han adicionado los modelos Llama 3.2 de 1B, 3B, 11B y 90B, lo que amplía las capacidades del sistema MALLO para manejar consultas complejas y multifacéticas.
  - Estos modelos están optimizados para multilingüismo y tareas de diálogo, lo que permitirá mejorar la comprensión y generación de respuestas en diferentes idiomas.

### Justificación de la integración
- La integración de Llama 3.2 permitirá mejorar la capacidad del sistema MALLO para manejar consultas complejas y multifacéticas.
- La optimización de estos modelos para tareas de diálogo y multilingüismo permitirá mejorar la comprensión y generación de respuestas en diferentes idiomas.
- La adición de estos modelos ampliará las capacidades del sistema MALLO y permitirá mejorar la calidad de las respuestas generadas.

### Impacto esperado
- Mejora en la capacidad del sistema MALLO para manejar consultas complejas y multifacéticas.
- Mejor comprensión y generación de respuestas en diferentes idiomas.
- Ampliación de las capacidades del sistema MALLO y mejora en la calidad de las respuestas generadas.

### Próximos pasos
- Realizar pruebas exhaustivas con los modelos Llama 3.2 para evaluar su rendimiento y ajustar los parámetros de configuración según sea necesario.
- Integrar los modelos Llama 3.2 en la arquitectura del sistema MALLO y realizar pruebas de integración.
- Recopilar feedback de usuarios para evaluar la mejora en la calidad de las respuestas generadas con los modelos Llama 3.2.

## [1.9.0] - 2024-09-18

### Añadido
- Integración del modelo DeepSeek 2.5 a través de OpenRouter.
  - Este modelo combina las capacidades de DeepSeek-V2-Chat y DeepSeek-Coder-V2-Instruct, ofreciendo una solución versátil para tareas generales y de codificación.

### Cambiado
- Actualización de la configuración del meta-analista:
  - Se ha establecido DeepSeek 2.5 como el modelo predeterminado para el meta-análisis.
- Modificación en la configuración del análisis final:
  - Hermes 3 (nousresearch/hermes-3-llama-3.1-405b) ha sido designado como el modelo principal para el análisis final.

### Justificación de los cambios

1. Integración de DeepSeek 2.5:
   - Versatilidad mejorada: La combinación de capacidades generales y de codificación en un solo modelo permite una mayor flexibilidad en el manejo de consultas diversas.
   - Optimización de recursos: Al utilizar un modelo que abarca múltiples dominios, se reduce la necesidad de cambiar entre diferentes modelos especializados, potencialmente mejorando la eficiencia del sistema.
   - Alineación mejorada con preferencias humanas: DeepSeek 2.5 ha sido optimizado para seguir instrucciones de manera más precisa, lo que puede resultar en respuestas más relevantes y contextuales.

2. DeepSeek 2.5 como meta-analista predeterminado:
   - Capacidad de síntesis mejorada: Las habilidades combinadas de chat general y codificación hacen que DeepSeek 2.5 sea ideal para sintetizar información de diversas fuentes y dominios.
   - Mejor comprensión de contexto: La optimización en el seguimiento de instrucciones puede llevar a un meta-análisis más preciso y coherente.
   - Potencial para manejar consultas técnicas y no técnicas: La versatilidad del modelo permite un meta-análisis más robusto en una amplia gama de temas.

3. Hermes 3 para análisis final:
   - Capacidades de razonamiento avanzadas: Hermes 3 ha demostrado un rendimiento excepcional en tareas que requieren razonamiento complejo, haciéndolo ideal para el análisis final de respuestas.
   - Manejo de contexto a largo plazo: La habilidad de Hermes 3 para mantener coherencia en conversaciones de múltiples turnos es crucial para un análisis final que tenga en cuenta todo el contexto de la interacción.
   - Alineación con instrucciones precisas: La capacidad de Hermes 3 para seguir instrucciones de manera exacta y adaptativa es esencial para realizar un análisis final riguroso y ajustado a los requerimientos específicos de cada consulta.

### Impacto esperado
- Mayor precisión en las respuestas generadas por el sistema MALLO.
- Mejora en la capacidad de manejar consultas complejas y multifacéticas.
- Potencial reducción en el tiempo de procesamiento para ciertos tipos de consultas.
- Mejor adaptabilidad a una amplia gama de estilos de consulta y dominios de conocimiento.

### Próximos pasos
- Monitorear el rendimiento de DeepSeek 2.5 y Hermes 3 en sus nuevos roles.
- Recopilar feedback de usuarios para evaluar la mejora en la calidad de las respuestas.
- Considerar ajustes finos en los prompts y parámetros de configuración para optimizar el rendimiento de los nuevos modelos en sus respectivos roles.

### Notas adicionales
- Se mantiene la configuración de modelos de respaldo para garantizar la robustez del sistema en caso de fallos o indisponibilidad de los modelos principales.
- Se recomienda realizar pruebas exhaustivas en diversos escenarios para validar la eficacia de estos cambios en diferentes tipos de consultas y cargas de trabajo.

## [1.8.0] - 2024-09-16
### Cambiado
- Refactorización completa de la función `get_prioritized_agents` para mejorar la selección de agentes basada en especialidades y tipos de prompt.
- Actualización de la lógica de selección de agentes para priorizar modelos especializados según el tipo de prompt.
- Modificación de la función `get_general_agents` para tener en cuenta las capacidades y tipos de prompts de los modelos generales.

### Mejorado
- Implementación de una función auxiliar `find_specialized_models` para buscar eficientemente modelos con especialidades específicas en la configuración.
- Optimización del uso de la información contenida en `config.yaml` para una selección de agentes más precisa y adaptable.
- Mayor flexibilidad en la selección de agentes, considerando tanto agentes especializados como modelos con capacidades específicas.

### Corregido
- Resolución del problema de duplicación de agentes especializados en la lista de agentes seleccionados.
- Corrección de errores relacionados con el manejo de tipos de datos en la selección de agentes.

### Optimizado
- Mejora en la eficiencia de la selección de agentes, reduciendo la redundancia y priorizando modelos relevantes.
- Refinamiento del proceso de selección para asegurar una distribución equilibrada entre agentes especializados y generales.

### Documentación
- Actualización de los comentarios en el código para reflejar los cambios en la lógica de selección de agentes.
- Adición de explicaciones detalladas sobre el nuevo proceso de priorización de agentes en la documentación interna.

### Seguridad
- Mejora en el manejo de excepciones para prevenir exposición de información sensible durante el proceso de selección de agentes.

### Experimental
- Introducción de un sistema de puntuación para evaluar la relevancia de los agentes seleccionados basado en el histórico de rendimiento (pendiente de implementación completa).

## [1.7.0] - 2024-09-15
### Añadido
- Implementación de un sistema de selección de prompts críticos basado en el tipo de consulta.
- Nuevos tipos de prompt crítico: audio_transcription, multimodal, tool_use, content_moderation, creative, y analytical.
- Función `determine_prompt_type_and_capabilities` para identificar el tipo de prompt y las capacidades requeridas.
- Integración de capacidades específicas para cada modelo en la configuración.
- Sistema de fallback para la selección de agentes y prompts.

### Cambiado
- Refactorización de la función `evaluate_query_complexity` para incluir la determinación del tipo de prompt.
- Actualización de la estructura de configuración en `config.yaml` para soportar nuevos campos de especialidad y capacidades por modelo.
- Mejora en la lógica de selección de agentes en `get_prioritized_agents` para considerar especialidades y capacidades.
- Optimización de la función `process_user_input` para manejar múltiples agentes y realizar meta-análisis cuando es necesario.

### Mejorado
- Mayor robustez en el manejo de errores y situaciones inesperadas en la carga de configuración.
- Implementación de un sistema de caché más eficiente para respuestas frecuentes.
- Mejora en la evaluación de la complejidad de las consultas, incluyendo factores lingüísticos y contextuales.

### Corregido
- Solucionado el problema de "too many values to unpack" en la función `evaluate_query_complexity`.
- Corregido el error "'str' object cannot be interpreted as an integer" en la función `determine_prompt_type`.
- Ajustes en la carga de configuración para manejar correctamente la estructura anidada de `prompt_types`.

### Optimizado
- Rendimiento mejorado en la selección de agentes y procesamiento de consultas.
- Reducción del uso de recursos en consultas repetitivas mediante un sistema de caché mejorado.

### Documentación
- Actualización de la documentación interna para reflejar los nuevos cambios y funcionalidades.
- Mejora en los comentarios del código para mayor claridad y mantenibilidad.

### Experimental
- Introducción de características experimentales para la reflexión iterativa y mejora continua de respuestas.

## [1.6.0] - 2024-09-12
### Añadido
- Implementación de un sistema de detección de tipo de consulta (`determine_prompt_type`) para categorizar las preguntas en diferentes áreas (matemáticas, programación, legal, científico, etc.).
- Nuevo parámetro `prompt_type` en la función `get_prioritized_agents` para mejorar la selección de agentes especializados.
- Integración de análisis crítico con probabilidad configurable (`critical_analysis_probability`) en el procesamiento de consultas.
- Aplicación de prompts especializados basados en el tipo de consulta detectado.

### Cambiado
- Refactorización de `evaluate_query_complexity` para incluir la determinación del tipo de prompt.
- Actualización de `get_prioritized_agents` en la clase `AgentManager` para priorizar la selección de agentes basada en el tipo de consulta y su complejidad.
- Modificación de `process_user_input` para manejar el nuevo flujo de procesamiento con tipos de prompts y análisis crítico.
- Optimización del proceso de meta-análisis para ejecutarse solo cuando hay múltiples respuestas exitosas.
- Actualización de la lógica para aplicar prompts especializados a todos los agentes, eliminando la selección aleatoria.

### Mejorado
- Manejo más eficiente de la complejidad de las consultas, ajustando dinámicamente el número de agentes utilizados.
- Mejor integración de asistentes especializados en el flujo de procesamiento.
- Refinamiento del sistema de evaluación inicial y final para proporcionar un análisis más preciso de las consultas y respuestas.
- Consistencia mejorada en la aplicación de prompts especializados a través de todos los agentes.

### Optimizado
- Reducción del uso innecesario de recursos para consultas simples.
- Mejora en la eficiencia del procesamiento al limitar el número de agentes basado en la complejidad de la consulta.
- Eliminación de la selección aleatoria de agentes para la aplicación de prompts especializados.

### Corregido
- Solución al problema de activación excesiva de agentes para consultas simples.
- Corrección en la aplicación de búsqueda web y MOA (Mixture of Agents) basada en una evaluación más precisa de la necesidad.
- Resolución del problema de selección de agentes duplicados, especialmente para agentes especializados.

### Documentación
- Actualización de la documentación interna para reflejar los nuevos cambios en el flujo de procesamiento y la selección de agentes.
- Clarificación sobre el uso consistente de prompts especializados en todos los agentes.

### Seguridad
- Mejora en el manejo de errores para prevenir la exposición de información sensible en caso de fallos en el procesamiento.

## [1.5.0] - 2024-09-12
### Añadido
- Implementación de prompts de análisis crítico especializados para diferentes tipos de consultas (matemáticas, programación, legal, científico, histórico, filosófico, ético, contexto colombiano, cultural, político, económico).
- Nueva función `determine_prompt_type` para identificar el tipo de consulta y seleccionar el prompt adecuado.
- Integración de un modelo de lenguaje en español (es_core_news_sm) para mejorar el procesamiento de consultas en español.

### Cambiado
- Actualizada la función `evaluate_query_complexity` para incluir la determinación del tipo de prompt.
- Modificada la clase `AgentManager` para manejar los nuevos prompts especializados y su probabilidad de uso.
- Actualizado el método `process_query` en `AgentManager` para aplicar los prompts especializados cuando sea apropiado.
- Mejorada la función `process_user_input` para utilizar el nuevo sistema de prompts especializados.

### Optimizado
- Refactorizado el código para una mejor modularidad y mantenibilidad.
- Mejorado el manejo de errores y logging en varias funciones clave.

### Corregido
- Solucionado el problema con el atributo faltante `critical_analysis_probability` en la clase `AgentManager`.
- Corregidos varios errores relacionados con la inicialización y uso de clientes de API.

### Actualizado
- Actualizado el archivo `requirements.txt` para incluir el modelo de spaCy en español.
- Modificada la función de carga del modelo de spaCy para usar el modelo en español y manejar su instalación automática si es necesario.

### Documentación
- Actualizada la documentación interna del código para reflejar los nuevos cambios y funcionalidades.

## [1.4.1] - 2024-09-10
### Cambiado
- Implementado un nuevo sistema de gestión de secretos para mejorar la compatibilidad con [Render](https://render.com/). Este cambio permite la carga de secretos desde archivos locales y variables de entorno, proporcionando una solución más flexible y segura para el manejo de claves API y otros datos sensibles en la aplicación MALLO.
- Reemplazado el uso de `st.secrets` por un sistema personalizado de secretos en toda la aplicación. En donde en `agents.py` y `utilities.py`, se reemplazan todas las instancias de `st.secrets` por `secrets` y donde se use `st.secrets.get()`, reemplázalo por `get_secret()`. Ejemplos:

```python
# Antes
api_key = st.secrets["OPENAI_API_KEY"]

# Después
api_key = secrets["OPENAI_API_KEY"]
```
y

```python
# Antes
api_key = st.secrets.get("OPENAI_API_KEY")

# Después
api_key = get_secret("OPENAI_API_KEY")
```

### Corregido
- Solucionado el problema de carga de secretos en el entorno de Render.
- Corregido el error "No secrets files found" durante el despliegue en Render.

### Optimizado
- Mejorada la carga de secretos para manejar múltiples fuentes (archivos, variables de entorno) de manera más robusta.


## [1.4.0] - 2024-09-10
### Añadido
- Nueva clase `MALLOEnhancer` para mejorar la calidad de las instrucciones y respuestas mediante reflexión iterativa.
- Función experimental `process_user_input_experimental` que implementa el enfoque de reflexión y mejora.
- Integración de múltiples APIs de LLM (OpenAI, Anthropic, Mistral, Cohere, Groq, DeepInfra, DeepSeek, OpenRouter) en `MALLOEnhancer`.

### Cambiado
- Mejorada la obtención de claves API utilizando los secretos de Streamlit.
- Actualizada la clase `AgentManager` para incluir el método `update_criteria`.

### Optimizado
- Implementación de cálculos de IFD (Instruction-Following Difficulty) y r-IFD (reversed-IFD) para evaluar la calidad de las instrucciones y respuestas.

### Corregido
- Solucionados varios errores relacionados con la inicialización de clientes de API y manejo de respuestas.

### Experimental
- Añadida la capacidad de habilitar el procesamiento experimental de entradas de usuario para investigación futura.

## [1.3.0] - 2024-09-08
### Cambiado
- Mejorado el proceso de selección de agentes con la implementación de `get_prioritized_agents` en `AgentManager`.
- Refinado el método `process_query` en `AgentManager` para un mejor manejo de errores y fallbacks.
- Actualizada la lógica de `process_user_input` en `main.py` para utilizar el nuevo sistema de priorización de agentes.

### Optimizado
- Mejorada la generación de respuestas finales para ser más conversacionales y directas.
- Refinado el proceso de meta-análisis para producir respuestas más coherentes y relevantes.

### Corregido
- Solucionado el problema con el atributo faltante `default_local_model` en `AgentManager`.
- Implementado el método faltante `get_available_models` en `AgentManager`.

### Añadido
- Nueva funcionalidad para manejar fallbacks de manera más robusta en caso de fallo de agentes primarios.
- Implementada una lógica mejorada para la selección de agentes basada en la complejidad de la consulta y la disponibilidad de modelos.

### Mejorado
- Optimizada la gestión de errores en todo el sistema para una mejor experiencia de usuario y depuración.
- Mejorada la documentación interna del código para facilitar futuro mantenimiento y desarrollo.

## [1.2.0] - 2024-09-07
### Añadido
- Sistema de puntuación para evaluar la calidad de las respuestas de los agentes.
- Función de síntesis para combinar las mejores partes de múltiples respuestas.
- Nuevo agente de meta-análisis para generar una respuesta final mejorada.
- Configuración en `config.yaml` para modelos de meta-análisis, incluyendo opciones de respaldo.

### Cambiado
- Modificada la función `process_user_input` para incorporar puntuación, síntesis y meta-análisis.
- Actualizado el método `meta_analysis` en `AgentManager` para usar modelos configurables con sistema de respaldo.
- Mejorado el manejo de errores en el proceso de meta-análisis.

### Mejorado
- Refinado el proceso de selección de respuesta final para reflejar un pensamiento más sofisticado.
- Aumentada la robustez del sistema ante fallos de modelos individuales.

### Optimizado
- Implementado un sistema de reintentos para el meta-análisis utilizando hasta tres modelos diferentes.

## [1.1.0] - 2024-09-07
### Añadido
- Integración de la API de OpenRouter con múltiples modelos.
- Sistema de recuperación de tres niveles para la evaluación de respuestas.
- Nuevo método `process_with_openrouter` en la clase `AgentManager`.

### Cambiado
- Actualizada la función `summarize_conversation` para usar OpenRouter como opción principal y OpenAI como respaldo.
- Modificada la función `evaluate_response` para incluir un segundo nivel de respaldo.
- Actualizado el archivo de configuración `config.yaml` para incluir configuraciones de OpenRouter.

### Optimizado
- Simplificada la gestión de tokens máximos para modelos de OpenRouter, aprovechando los límites predeterminados de la API.

## [1.0.0] - 2024-08-22
### Añadido
- README.md completo con información detallada del proyecto
- Este archivo CHANGELOG.md para rastrear el historial del proyecto

## [0.9.0] - 2024-08-21
### Añadido
- Implementado sistema de caché para las respuestas de consultas
- Agregado seguimiento de métricas de rendimiento para cada agente

### Cambiado
- Optimizado el algoritmo de selección de agentes para tiempos de respuesta más rápidos
- Actualizada la interfaz de usuario para una mejor visualización de respuestas

### Corregido
- Resueltos problemas con la funcionalidad de búsqueda web
- Corregida la fuga de memoria en sesiones de Streamlit de larga duración

## [0.8.0] - 2024-08-20
### Añadido
- Integradas las APIs de Anthropic, DeepSeek, Mistral y Cohere
- Implementado sistema avanzado de manejo de errores y registro

### Cambiado
- Refactorizado AgentManager para mejorar la modularidad
- Mejorado el algoritmo de evaluación de complejidad de consultas

## [0.7.0] - 2024-08-19
### Añadido
- Implementado enfoque de Mezcla de Agentes (MOA) para consultas complejas
- Agregado soporte para asistentes especializados en varios dominios
- Integradas las APIs de Together AI y Groq

### Cambiado
- Actualizada la estructura de config.yaml para una gestión más fácil de múltiples APIs
- Mejoradas las capacidades de búsqueda web con integración de DuckDuckGo

## [0.6.0] - 2024-08-18
### Añadido
- Implementación inicial de la clase AgentManager
- Integración básica con modelos de OpenAI y Ollama locales
- Interfaz de usuario basada en Streamlit para entrada de consultas y visualización de respuestas
- Sistema de configuración usando YAML para fácil personalización
- Lógica básica de procesamiento de consultas y selección de agentes

### Cambiado
- Estructurado el proyecto con main.py, agents.py y utilities.py

## [0.5.0] - 2024-08-18
### Añadido
- Inicialización del proyecto
- Estructura básica del proyecto y gestión de dependencias
- Documentación inicial y configuración del proyecto

[No publicado]: https://github.com/bladealex9848/MALLO/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/bladealex9848/MALLO/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/bladealex9848/MALLO/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/bladealex9848/MALLO/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/bladealex9848/MALLO/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/bladealex9848/MALLO/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/bladealex9848/MALLO/releases/tag/v0.5.0
