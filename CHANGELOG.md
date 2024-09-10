# Registro de Cambios
Todos los cambios notables en el proyecto MALLO serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto se adhiere a [Versionado Semántico](https://semver.org/lang/es/).

# Registro de Cambios
Todos los cambios notables en el proyecto MALLO serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto se adhiere a [Versionado Semántico](https://semver.org/lang/es/).

# Registro de Cambios

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