import numpy as np
import logging
import hashlib

logger = logging.getLogger("WAG-Narrator")

# Intentamos cargar librerías de ML, si fallan usamos modo simulado (Hashing)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("⚠️ sentence-transformers no encontrado. Usando modo SIMULACIÓN matemática (Hash determinista).")

class TensorNarrator:
    """
    El Cerebro Matemático del Grid.
    Transforma texto -> tensores y calcula 'Cohesión' en lugar de colisiones físicas.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.use_ml = ML_AVAILABLE
        self.vector_dim = 384 # Dimensión estándar de MiniLM
        self.model = None
        
        if self.use_ml:
            logger.info(f"🧠 Cargando modelo tensorial: {model_name}...")
            try:
                self.model = SentenceTransformer(model_name)
                logger.info("✅ Modelo tensorial cargado en RAM.")
            except Exception as e:
                logger.error(f"Error cargando modelo: {e}")
                self.use_ml = False

    def embed(self, text: str) -> np.ndarray:
        """Convierte texto en un vector (tensor 1D)."""
        if not text:
            return np.zeros(self.vector_dim)
            
        if self.use_ml and self.model:
            return self.model.encode(text)
        else:
            # MODO SIMULACIÓN: Hashing determinista
            # Esto asegura que el mismo texto siempre dé el mismo vector "falso"
            # pero matemáticamente consistente para pruebas.
            seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (2**32)
            np.random.seed(seed)
            vec = np.random.rand(self.vector_dim)
            return vec / np.linalg.norm(vec) # Normalizar

    def calculate_cohesion(self, vec_a, vec_b) -> float:
        """
        Mide la alineación entre dos pensamientos.
        1.0 = Idénticos
        0.0 = Ortogonales (Nada que ver)
        -1.0 = Opuestos
        """
        if vec_a is None or vec_b is None: return 0.0
        
        # Producto punto normalizado (Similitud Coseno)
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0: return 0.0
        
        return float(dot_product / (norm_a * norm_b))

    def tensor_to_rgb(self, vector: np.ndarray) -> tuple:
        """
        Proyecta el vector de 384 dimensiones a 3 dimensiones (Color).
        Esto permite 'ver' el significado del paquete en la UI.
        """
        if vector is None: return (200, 200, 200)
        
        # Reducción de dimensionalidad muy simple (promedio por tercios)
        # En producción usaríamos PCA o t-SNE, esto es una aproximación rápida.
        chunk_size = len(vector) // 3
        r = np.abs(np.mean(vector[0:chunk_size]))
        g = np.abs(np.mean(vector[chunk_size:chunk_size*2]))
        b = np.abs(np.mean(vector[chunk_size*2:]))
        
        # Normalizar a 0-255 con saturación para que se vea bonito en UI
        def norm(v): return int(min(255, v * 2000)) # Multiplicador alto para saturación
        
        return (norm(r), norm(g), norm(b))

# Singleton para uso fácil en todo el sistema
_narrator_instance = None

def get_narrator():
    global _narrator_instance
    if _narrator_instance is None:
        _narrator_instance = TensorNarrator()
    return _narrator_instance
