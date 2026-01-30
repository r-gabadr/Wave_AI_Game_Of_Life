import os
import time
import uuid
import hashlib
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import lancedb
import duckdb
import pyarrow as pa
import logging

# --- CONFIGURACIÓN DE LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WAG-Memory-Quantum")

# --- DETECCIÓN DE BACKEND TAICHI ---
try:
    import taichi as ti
    # Inicialización con soporte GPU si está disponible
    ti.init(arch=ti.gpu if ti._lib.core.with_cuda() else ti.cpu, log_level=ti.ERROR)
    TAICHI_AVAILABLE = True
    logger.info("🚀 NPE Activated: TAICHI CORE (GPU/CPU)")
except Exception as e:
    TAICHI_AVAILABLE = False
    logger.warning(f"⚠️ NPE Mode: Fallback to NUMPY PHYSICS ({e}).")

# ============================================================================
# CAPA 0: FÍSICA Y TENSOR (VOXEL GRID & ENTROPY)
# ============================================================================

if TAICHI_AVAILABLE:
    @ti.data_oriented
    class TaichiTensorField:
        """
        El 'Lienzo Holográfico' del sistema (Capa 4D).
        Almacena el estado volumétrico del Grid para Raymarching futuro.
        """
        def __init__(self, resolution=64):
            self.res = resolution
            # Campo escalar de densidad (Energía/Prioridad)
            self.density = ti.field(dtype=ti.f32)
            # Campo de identificación (Hash del Ciclo ID)
            self.cycle_id = ti.field(dtype=ti.i32)
            
            # Estructura SVO (Sparse Voxel Octree) para eficiencia de memoria
            # Bloques grandes -> Bloques pequeños -> Vóxeles
            self.block = ti.root.pointer(ti.ijk, (resolution // 8, resolution // 8, resolution // 8))
            self.pixel = self.block.dense(ti.ijk, (8, 8, 8))
            self.pixel.place(self.density, self.cycle_id)

        @ti.kernel
        def clear_grid(self):
            """Limpia el tensor para el siguiente frame."""
            for I in ti.grouped(self.block):
                self.density[I] = 0.0
                self.cycle_id[I] = -1

        @ti.kernel
        def inject_voxel(self, x: int, y: int, z: int, energy: float, c_id: int):
            """Inyecta un 'paquete' en el espacio volumétrico."""
            if 0 <= x < self.res and 0 <= y < self.res and 0 <= z < self.res:
                self.density[x, y, z] = energy
                self.cycle_id[x, y, z] = c_id

    @ti.data_oriented
    class EntropicVRAMIndex:
        """Motor de Física Semántica (Cálculo de S_real y S_phase)."""
        def __init__(self, max_elements=4096, dims=128):
            self.max_elements = max_elements
            self.dims = dims
            
            # Campos Taichi
            self.vectors = ti.field(dtype=ti.f32, shape=(max_elements, dims))
            self.densities = ti.field(dtype=ti.f32, shape=(max_elements))
            self.scores = ti.field(dtype=ti.f32, shape=(max_elements))
            self.last_query = ti.field(dtype=ti.f32, shape=(dims))
            self.has_history = ti.field(dtype=ti.i32, shape=())
            
            # Punteros para buffer circular
            self.write_ptr = 0
            self.current_count = 0
            
            # Output vectorizado: [Entropy_Real, Entropy_Phase]
            self.physics_output = ti.Vector.field(2, dtype=ti.f32, shape=())
            
            # Constantes físicas
            self.T_ENT = 0.15 # Temperatura entrópica

        @ti.kernel
        def _compute_physics(self, query_vec: ti.types.ndarray()):
            s_real = 0.0
            s_phase = 0.0
            sum_rho = 0.0
            
            # 1. Fase de Activación (Dot Product + Density Weighting)
            for i in range(self.current_count):
                dot = 0.0
                for d in range(self.dims): 
                    dot += self.vectors[i, d] * query_vec[d]
                
                # Modulación por densidad (Gravedad Semántica)
                force = dot
                if self.densities[i] < 0.1:
                    force = ti.sqrt(ti.abs(dot)) * ti.math.sign(dot) # MOND regime
                
                self.scores[i] = ti.min(force * 10.0, 20.0)
                sum_rho += ti.exp(self.scores[i])

            # 2. Fase de Entropía (Cálculo Termodinámico)
            for i in range(self.current_count):
                rho = ti.exp(self.scores[i]) / (sum_rho + 1e-9)
                
                # Entropía Real (Shannon)
                s_real += -rho * ti.log(rho + 1e-9)
                
                # Entropía de Fase (Coherencia Temporal)
                if self.has_history[None] == 1:
                    delta_dot = 0.0
                    for d in range(self.dims):
                        # Cambio respecto a la query anterior (Derivada temporal)
                        delta_dot += (query_vec[d] - self.last_query[d]) * self.vectors[i, d]
                    
                    v_ent = self.T_ENT * ti.log(rho + 1e-6)
                    phase = ti.atan2(delta_dot + v_ent, 1.0)
                    s_phase += ti.abs(phase) * rho
                    
            self.physics_output[None] = ti.Vector([s_real, s_phase])

        @ti.kernel
        def _update_history(self, query_vec: ti.types.ndarray()):
            for d in range(self.dims): self.last_query[d] = query_vec[d]
            self.has_history[None] = 1
        @ti.kernel
        def _set_soliton_kernel(self, idx: int, vec: ti.types.ndarray(), density: float):
            self.densities[idx] = density
            for d in range(self.dims):
                self.vectors[idx, d] = vec[d]
        def add_soliton(self, vec: np.ndarray, density: float = 1.0):
            """Añade un solitón al buffer circular usando el kernel de Taichi."""
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            # Usamos el kernel para mover los datos a la memoria de Taichi
            self._set_soliton_kernel(self.write_ptr, vec, density)
            
            self.write_ptr = (self.write_ptr + 1) % self.max_elements
            self.current_count = min(self.current_count + 1, self.max_elements)

        def calculate_entropy(self, query_vec: np.ndarray) -> Tuple[float, float]:
            if self.current_count == 0: return 0.0, 0.0
            # Normalizar query también
            query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
            self._compute_physics(query_vec)
            out = self.physics_output[None]
            return float(out[0]), float(out[1])
        
        def reset_phase(self):
            self.has_history[None] = 0
            
        def snapshot(self):
            """Captura el estado actual de los punteros."""
            return self.write_ptr, self.current_count

        def restore(self, snap):
            """Restaura los punteros (Rollback)."""
            self.write_ptr, self.current_count = snap
            
        def get_physics_state(self) -> Dict:
            if self.current_count == 0:
                return {"soliton_count": 0, "density_mean": 0.0}
            densities = self.densities.to_numpy()[:self.current_count]
            return {
                "soliton_count": self.current_count,
                "density_mean": float(np.mean(densities)),
                "density_std": float(np.std(densities))
            }

else:
    # --- FALLBACKS (SIN GPU) ---
    class TaichiTensorField:
        def __init__(self, resolution=64): pass
        def clear_grid(self): pass
        def inject_voxel(self, x, y, z, e, c): pass
    
    class EntropicVRAMIndex:
        def __init__(self, **kwargs): self.current_count = 0
        def add_soliton(self, *args): pass
        def calculate_entropy(self, *args): return 0.0, 0.0
        def _update_history(self, *args): pass
        def reset_phase(self): pass
        def snapshot(self): return None
        def restore(self, snap): pass
        def get_physics_state(self): return {}


# ============================================================================
# CAPA 1 y 2: MEMORIA TRIPLE (SINGLETON)
# ============================================================================

class TripleMemory:
    """
    SINGLETON: Interfaz unificada de memoria.
    - Capa 0: VRAM (Física/Tensor)
    - Capa 1: LanceDB (Vectores)
    - Capa 2: DuckDB (Estructura/SQL)
    """
    _instance = None

    @classmethod
    def get(cls):
        """Método estático para acceder a la memoria desde cualquier módulo (Grid/Types)."""
        if cls._instance is None:
            # Si se llama antes de que Core lo inicie, creamos uno por defecto
            cls._instance = cls()
        return cls._instance

    def __init__(self, base_dir="wag_storage"):
        # Asegurar Singleton
        if TripleMemory._instance is not None:
            # Si ya existe, copiamos el estado del existente a este nuevo 'self'
            # para mantener compatibilidad si alguien instancia directamente.
            self.__dict__ = TripleMemory._instance.__dict__
            return

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. LanceDB (Semántica)
        self.ldb = lancedb.connect(self.base_dir / "lancedb")
        self._init_lancedb()
        
        # 2. DuckDB (Estructural)
        self.duck = duckdb.connect(str(self.base_dir / "analytics.duckdb"))
        self._init_duckdb()
        
        # 3. VRAM (Física)
        self.vram = EntropicVRAMIndex()
        
        # 4. Tensor Grid (Holográfico)
        self.grid_tensor = TaichiTensorField(resolution=64)
        
        self.active_conversation_id = None
        
        # Registrar instancia global
        TripleMemory._instance = self
        logger.info("🧠 WAG TripleMemory: ONLINE (Singleton)")

    # --- INICIALIZACIÓN DE BASES DE DATOS ---
    def _init_lancedb(self):
        table_names = self.ldb.table_names()

        # Initialize 'knowledge' table
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), 128)),
            pa.field("text", pa.string()),
            pa.field("ts", pa.float64()),
            pa.field("density", pa.float32()),
            pa.field("conversation_id", pa.string())
        ])
        if "knowledge" not in table_names:
            self.table = self.ldb.create_table("knowledge", schema=schema)
        else:
            self.table = self.ldb.open_table("knowledge")

        # Initialize 'summaries' table
        summary_schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), 128)),
            pa.field("summary_text", pa.string()),
            pa.field("ts", pa.float64()),
            pa.field("conversation_id", pa.string())
        ])
        if "summaries" not in table_names:
            self.summary_table = self.ldb.create_table("summaries", schema=summary_schema)
        else:
            self.summary_table = self.ldb.open_table("summaries")

    def _init_duckdb(self):
        # Tablas base
        self.duck.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id VARCHAR PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                summary TEXT
            )
        """)
        self.duck.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id VARCHAR PRIMARY KEY,
                conversation_id VARCHAR,
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                role TEXT,
                content TEXT,
                entropy_real FLOAT DEFAULT 0.0,
                entropy_phase FLOAT DEFAULT 0.0
            )
        """)
        
        # Migraciones de esquema seguras (v1 -> v2)
        def safe_add(table, col, type_def):
            cols = [r[1] for r in self.duck.execute(f"PRAGMA table_info('{table}')").fetchall()]
            if col not in cols:
                self.duck.execute(f"ALTER TABLE {table} ADD COLUMN {col} {type_def}")
        
        safe_add('conversations', 'summary', 'TEXT')
        safe_add('messages', 'entropy_real', 'FLOAT DEFAULT 0.0')
        safe_add('messages', 'entropy_phase', 'FLOAT DEFAULT 0.0')
        safe_add('messages', 'cycle_info', 'TEXT')

    # =========================================================================
    # API PÚBLICA (INTERFAZ DE USUARIO Y SISTEMA)
    # =========================================================================

    def sync_grid_tensor(self, x_norm: float, y_norm: float, z_norm: float, cycle_hash: int):
        """
        Permite al Grid inyectar voxels en el campo Taichi directamente.
        Se usa desde wag_grid.py.
        """
        if TAICHI_AVAILABLE:
            res = self.grid_tensor.res
            ix = int(x_norm * res)
            iy = int(y_norm * res)
            iz = int(z_norm * res)
            self.grid_tensor.inject_voxel(ix, iy, iz, 1.0, cycle_hash)

    def retrieve(self, query: str, conv_id: Optional[str] = None, limit: int = 5) -> Tuple[str, float, float]:
        """
        Recuperación híbrida: Busca semántica y calcula física.
        Retorna: (Contexto Texto, Entropía Real, Entropía Fase)
        """
        vec = self._text_to_vec(query)
        
        # 1. Física (Taichi)
        s_real, s_phase = self.vram.calculate_entropy(vec)
        
        # Solo actualizamos historial si es una interacción real (no simulación)
        if conv_id is not None:
            self.vram._update_history(vec)
        
        # 2. Semántica (LanceDB)
        search = self.table.search(vec.tolist()).limit(limit)
        if conv_id:
            try:
                res = search.where(f"conversation_id = '{conv_id}'").to_list()
            except: 
                res = search.to_list() # Fallback si falla el filtro
        else:
            res = search.to_list()
            
        context = "\n".join([f"- {r['text']}" for r in res]) if res else ""
        return context, s_real, s_phase

    def fusion_retrieve(self, query: str, conversation_ids: List[str]) -> Tuple[str, float, float]:
        """Recuperación multi-hilo para fusión de contextos (CON ROLLBACK)."""
        vec = self._text_to_vec(query)
        
        # Snapshot del estado antes de simular
        snap = self.vram.snapshot()
        self.vram.reset_phase()
        
        fused_context = []
        # Traer un poco de cada hilo
        for cid in conversation_ids:
            res = self.table.search(vec.tolist()).where(f"conversation_id = '{cid}'").limit(2).to_list()
            for r in res:
                v_soliton = np.array(r["vector"], dtype=np.float32)
                # Inyectamos en VRAM temporalmente para calcular la interferencia
                self.vram.add_soliton(v_soliton, density=0.5)
                fused_context.append(f"[{cid}]: {r['text']}")
        
        s_real, s_phase = self.vram.calculate_entropy(vec)
        
        # ROLLBACK: Limpiamos la VRAM de la simulación
        self.vram.restore(snap)
        
        return "\n".join(fused_context), s_real, s_phase

    def ingest(self, text: str, role: str, entropy: Tuple[float, float] = (0.0, 0.0), conv_id: Optional[str] = None, cycle_info: Optional[str] = None):
        """Ingesta unificada (Log + Vector + Solitón)."""
        target_conv = conv_id or self.active_conversation_id or self.create_conversation()
        msg_id = str(uuid.uuid4())[:8]
        
        try:
            # 1. Log Estructural (DuckDB)
            self.duck.execute(
                "INSERT INTO messages (id, conversation_id, role, content, entropy_real, entropy_phase, cycle_info) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (msg_id, target_conv, role, text, entropy[0], entropy[1], cycle_info)
            )
            self.duck.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (target_conv,))
            
            # 2. Memoria Semántica (LanceDB)
            vec = self._text_to_vec(text)
            density = 1.0 / (1.0 + abs(entropy[0])) # Menos entropía = Más densidad/peso
            self.table.add([{
                "vector": vec.tolist(), "text": text[:2000], 
                "ts": time.time(), "density": density, 
                "conversation_id": target_conv
            }])
            
            # 3. Física Inmediata (VRAM Soliton)
            self.vram.add_soliton(vec, density=density)
            
        except Exception as e:
            logger.error(f"Ingest Error: {e}")

    def ingest_summary(self, summary_text: str, conv_id: Optional[str] = None):
        """Guarda resúmenes consolidados (Sueño REM)."""
        target = conv_id or self.active_conversation_id
        if not target: return
        vec = self._text_to_vec(summary_text)
        self.summary_table.add([{
            "vector": vec.tolist(), "summary_text": summary_text,
            "ts": time.time(), "conversation_id": target
        }])
        self.duck.execute("UPDATE conversations SET summary = ? WHERE id = ?", (summary_text, target))

    def create_conversation(self, title="New Chat") -> str:
        cid = str(uuid.uuid4())[:8]
        self.duck.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", (cid, title))
        self.active_conversation_id = cid
        return cid

    def _text_to_vec(self, text: str) -> np.ndarray:
        # Hashing determinista ESTABLE entre procesos
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "little")
        rng = np.random.default_rng(seed)
        v = rng.random(128, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def get_vram_state(self) -> Dict[str, Any]:
        """Telemetría para el Dashboard."""
        return self.vram.get_physics_state()
