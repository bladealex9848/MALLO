import streamlit as st
from typing import Dict, Any, Optional, List, Union
import time
import logging
import json
from datetime import datetime, timedelta
import orjson
from pathlib import Path
from prometheus_client import Counter, Histogram
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential
from pythonjsonlogger import jsonlogger
import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Configuración de logging
logger = logging.getLogger(__name__)
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Métricas de Prometheus
CACHE_HITS = Counter('mallo_cache_hits_total', 'Total number of cache hits')
CACHE_MISSES = Counter('mallo_cache_misses_total', 'Total number of cache misses')
CACHE_OPERATION_DURATION = Histogram('mallo_cache_operation_seconds', 'Duration of cache operations')

class CacheManager:
    """
    Gestiona el caché global de MALLO para componentes persistentes.
    """
    
    def __init__(self, cache_dir: str = ".cache", ttl: int = 3600):
        """
        Inicializa el gestor de caché.
        
        Args:
            cache_dir: Directorio para almacenar caché persistente
            ttl: Tiempo de vida del caché en segundos (default 1 hora)
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.memory_cache = {}
        self.cache_dir.mkdir(exist_ok=True)
        self._init_scheduler()
        logger.info(f"CacheManager initialized with cache_dir={cache_dir}, ttl={ttl}")

    def _init_scheduler(self):
        """Inicializa el planificador de tareas para limpieza de caché."""
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self.cleanup_cache,
            trigger=IntervalTrigger(hours=1),
            id='cache_cleanup',
            name='Clean expired cache entries'
        )
        self.scheduler.start()
        logger.info("Cache cleanup scheduler initialized")

    @st.cache_resource
    def get_core_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inicializa y cachea los componentes principales del sistema.
        
        Args:
            config: Configuración del sistema
            
        Returns:
            Dict con los componentes principales inicializados
        """
        from agents import AgentManager
        from utilities import initialize_system, load_speed_test_results
        
        cache_key = self._generate_cache_key("core_components", config)
        
        # Intentar obtener del caché
        cached_result = self.get(cache_key)
        if cached_result:
            logger.info("Core components retrieved from cache")
            CACHE_HITS.inc()
            return cached_result
        
        CACHE_MISSES.inc()
        with CACHE_OPERATION_DURATION.time():
            try:
                start_time = time.time()
                
                # Inicializar componentes
                system_status = initialize_system(config)
                agent_manager = AgentManager(config)
                speed_test_results = load_speed_test_results()
                
                components = {
                    'system_status': system_status,
                    'agent_manager': agent_manager,
                    'speed_test_results': speed_test_results,
                    'initialization_time': time.time() - start_time,
                    'initialized_at': time.time()
                }
                
                # Guardar en caché
                self.set(cache_key, components)
                logger.info(f"Core components initialized in {components['initialization_time']:.2f} seconds")
                
                return components
                
            except Exception as e:
                logger.error(f"Error initializing core components: {str(e)}")
                raise

    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        """
        Genera una clave de caché única basada en los datos.
        
        Args:
            prefix: Prefijo para la clave
            data: Datos para generar la clave
            
        Returns:
            Clave de caché única
        """
        if isinstance(data, dict):
            serialized = orjson.dumps(data, option=orjson.OPT_SORT_KEYS)
        else:
            serialized = str(data).encode()
        return f"{prefix}_{hash(serialized)}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene un valor del caché.
        
        Args:
            key: Clave de búsqueda
            
        Returns:
            Valor cacheado o None
        """
        # Verificar caché en memoria
        if key in self.memory_cache:
            value, timestamp = self.memory_cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.memory_cache[key]
        
        # Verificar caché persistente
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = orjson.loads(f.read())
                    if time.time() - cached_data['timestamp'] < self.ttl:
                        return cached_data['value']
                    else:
                        cache_file.unlink()
            except Exception as e:
                logger.error(f"Error reading cache file {cache_file}: {str(e)}")
        
        return None

    def set(self, key: str, value: Any, persist: bool = True):
        """
        Almacena un valor en el caché.
        
        Args:
            key: Clave para almacenar
            value: Valor a almacenar
            persist: Si debe persistir a disco
        """
        try:
            timestamp = time.time()
            
            # Guardar en caché de memoria
            self.memory_cache[key] = (value, timestamp)
            
            # Persistir si es necesario
            if persist:
                cache_data = {
                    'value': value,
                    'timestamp': timestamp
                }
                cache_file = self.cache_dir / f"{key}.json"
                with open(cache_file, 'wb') as f:
                    f.write(orjson.dumps(cache_data))
            
            logger.info(f"Cache entry set for key: {key}")
            
        except Exception as e:
            logger.error(f"Error setting cache entry for key {key}: {str(e)}")
            raise

    def cleanup_cache(self):
        """Limpia entradas expiradas del caché."""
        current_time = time.time()
        
        # Limpiar caché en memoria
        expired_keys = [
            key for key, (_, timestamp) in self.memory_cache.items()
            if current_time - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self.memory_cache[key]
        
        # Limpiar caché persistente
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = orjson.loads(f.read())
                    if current_time - cached_data['timestamp'] >= self.ttl:
                        cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Error cleaning up cache file {cache_file}: {str(e)}")
                    cache_file.unlink()  # Eliminar archivo corrupto
            
            logger.info(f"Cache cleanup completed. Removed {len(expired_keys)} memory entries")
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {str(e)}")

    def clear(self):
        """Limpia todo el caché."""
        try:
            # Limpiar caché en memoria
            self.memory_cache.clear()
            
            # Limpiar caché persistente
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            
            logger.info("Cache cleared completely")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del caché.
        
        Returns:
            Diccionario con estadísticas
        """
        try:
            memory_entries = len(self.memory_cache)
            persistent_entries = len(list(self.cache_dir.glob("*.json")))
            memory_size = sum(len(str(v)) for v, _ in self.memory_cache.values())
            persistent_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
            
            return {
                'memory_entries': memory_entries,
                'persistent_entries': persistent_entries,
                'memory_size_kb': memory_size / 1024,
                'persistent_size_kb': persistent_size / 1024,
                'ttl_seconds': self.ttl,
                'cache_dir': str(self.cache_dir)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {}

# Instancia global del CacheManager
cache_manager = CacheManager()

def get_cache_manager() -> CacheManager:
    """
    Obtiene la instancia global del CacheManager.
    
    Returns:
        Instancia de CacheManager
    """
    return cache_manager