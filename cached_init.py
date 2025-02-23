import streamlit as st
import logging
import time
from typing import Dict, Any
from agents import AgentManager
from utilities import initialize_system, load_config, load_speed_test_results

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mallo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@st.cache_resource
def initialize_core_components() -> Dict[str, Any]:
    """
    Inicializa y cachea los componentes principales del sistema.
    Esta función solo se ejecutará una vez y Streamlit mantendrá
    los resultados en caché entre recargas.
    
    Returns:
        Dict[str, Any]: Diccionario con los componentes principales inicializados
    """
    try:
        start_time = time.time()
        logger.info("Iniciando inicialización de componentes principales...")

        # Cargar configuración
        config = load_config()
        logger.info("Configuración cargada exitosamente")
        
        # Inicializar sistema
        system_status = initialize_system(config)
        logger.info("Sistema inicializado con éxito")
        
        # Crear gestor de agentes
        agent_manager = AgentManager(config)
        logger.info("Gestor de agentes creado exitosamente")
        
        # Cargar resultados de pruebas de velocidad
        speed_test_results = load_speed_test_results()
        logger.info("Resultados de pruebas de velocidad cargados")

        initialization_time = time.time() - start_time
        logger.info(f"Inicialización completada en {initialization_time:.2f} segundos")
        
        return {
            'config': config,
            'system_status': system_status,
            'agent_manager': agent_manager,
            'speed_test_results': speed_test_results,
            'initialization_time': initialization_time
        }
        
    except Exception as e:
        error_msg = f"Error crítico durante la inicialización: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def initialize_session_state():
    """
    Inicializa el estado de la sesión con valores predeterminados.
    Esta función debe ejecutarse en cada recarga.
    """
    default_states = {
        "messages": [],
        "context": "",
        "show_settings": False,
        "last_details": {},
        "error_count": 0,
        "last_successful_response": None,
        "initialization_timestamp": time.time()
    }
    
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

def initialize_app() -> Dict[str, Any]:
    """
    Inicializa la aplicación utilizando componentes cacheados.
    
    Returns:
        Dict[str, Any]: Diccionario con todos los componentes necesarios
    """
    try:
        # Inicializar estado de la sesión (esto debe ejecutarse cada vez)
        initialize_session_state()
        logger.info("Estado de sesión inicializado")
        
        # Obtener componentes cacheados
        core_components = initialize_core_components()
        logger.info(f"Componentes principales recuperados (tiempo original de inicialización: {core_components.get('initialization_time', 'N/A')}s)")
        
        return core_components
        
    except Exception as e:
        error_msg = f"Error durante la inicialización de la aplicación: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)