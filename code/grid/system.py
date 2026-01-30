import logging
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import asdict

# CONTRATO DE TIPOS: Aseguramos coherencia absoluta
from core.types import TaskPacket, NodeJournal, ActiveCycle, CycleStatus, CycleBlueprint, RouteBlueprint

# COMPONENTES DE INFRAESTRUCTURA
from data.memory_manager import TripleMemory
from data.narrator import get_narrator 

logger = logging.getLogger("WAG-Grid-Tensor")

class WAGGridSystem:
    """
    Sistema de Topología Tensorial.
    Gestiona la convergencia entre Nodos Físicos (RAM/VRAM) y Rutas Virtuales (Blueprints).
    Guiado por semántica (Tensores) y preparado para orquestación vía JSON/Front.
    """
    def __init__(self, persistence_path="wag_storage"):
        self.base_path = Path(persistence_path)
        self.base_path.mkdir(exist_ok=True)
        self.persistence_file = self.base_path / "grid_tensor.json"
        self.bp_file = self.base_path / "blueprints.json"
        
        # --- ESTADO FÍSICO (Recursos en RAM/VRAM) ---
        # Mapeo: (ID_Worker, Slot_Y) -> ID_Paquete
        self.physical_grid: Dict[Tuple[str, int], str] = {}
        
        # --- ESTADO LÓGICO (Capas del Tensor / Ciclos Vivos) ---
        self.active_cycles: Dict[str, ActiveCycle] = {}
        
        # --- REGISTROS ---
        self.packets: Dict[str, TaskPacket] = {}
        self.journals: Dict[str, NodeJournal] = {} 
        
        # --- BIBLIOTECA DE PLANOS (Diseños inyectados desde el Front) ---
        self.blueprints: Dict[str, CycleBlueprint] = {}
        
        self.lock = asyncio.Lock()
        
        # INICIALIZAR EL NARRADOR (Cerebro Matemático)
        self.narrator = get_narrator()
        
        self.load_blueprints()
        self.load_state()

    # =========================================================================
    # GESTIÓN DE BLUEPRINTS (Para formularios JSON del Front)
    # =========================================================================

    def save_blueprint(self, blueprint_data: Dict) -> bool:
        """
        Recibe un JSON del front y registra un nuevo diseño de flujo lógico.
        """
        try:
            bp_id = blueprint_data.get("id")
            routes = {}
            for rid, rdata in blueprint_data.get("routes", {}).items():
                routes[rid] = RouteBlueprint(
                    id=rid,
                    steps=rdata.get("steps", []),
                    role=rdata.get("role", "executor"),
                    priority_z=rdata.get("priority", 0.5)
                )

            bp = CycleBlueprint(
                id=bp_id,
                name=blueprint_data.get("name", "Untitled"),
                description=blueprint_data.get("description", ""),
                routes=routes,
                sync_rules=blueprint_data.get("sync_rules", {})
            )
            
            self.blueprints[bp_id] = bp
            self._persist_blueprints_to_disk()
            
            logger.info(f"📐 Blueprint guardado y listo para exportar: {bp.name}")
            return True
        except Exception as e:
            logger.error(f"Error guardando blueprint desde JSON: {e}")
            return False

    def _persist_blueprints_to_disk(self):
        """Serializa la biblioteca completa a JSON para persistencia."""
        dump = {
            bid: {
                "id": b.id, "name": b.name, "description": b.description,
                "routes": {rid: {"steps": r.steps, "role": r.role, "priority_z": r.priority_z} for rid, r in b.routes.items()},
                "sync_rules": b.sync_rules
            }
            for bid, b in self.blueprints.items()
        }
        with open(self.bp_file, "w") as f:
            json.dump(dump, f, indent=2)

    def load_blueprints(self):
        """Carga los diseños de flujos guardados al iniciar."""
        try:
            if not self.bp_file.exists():
                self._create_default_blueprints()
                return

            with open(self.bp_file, "r") as f:
                data = json.load(f)
                for bid, bdata in data.items():
                    routes = {
                        rid: RouteBlueprint(
                            id=rid, 
                            steps=r["steps"], 
                            role=r["role"],
                            priority_z=r.get("priority_z", 0.5)
                        )
                        for rid, r in bdata["routes"].items()
                    }
                    self.blueprints[bid] = CycleBlueprint(
                        id=bid, name=bdata["name"], description=bdata["description"],
                        routes=routes, sync_rules=bdata.get("sync_rules", {})
                    )
            logger.info(f"📜 {len(self.blueprints)} Blueprints cargados en la biblioteca.")
        except Exception as e:
            logger.error(f"Error cargando biblioteca de blueprints: {e}")

    def _create_default_blueprints(self):
        """Planos básicos de fábrica para arranque en frío."""
        self.save_blueprint({
            "id": "SIMPLE_CHAT",
            "name": "Chat Directo (R1)",
            "description": "Ciclo simple de inferencia.",
            "routes": {"main": {"steps": ["R1", "OUTPUT"], "role": "executor"}}
        })
        self.save_blueprint({
        "id": "RECURSIVE_DEV",
        "name": "Investigación y Desarrollo Recursivo",
        "description": "Busca en web, razona, aplica código y valida. Si falla, retrocede.",
        "routes": {
            "main": {
                "steps": ["WEB", "R1", "AIDER", "R1", "OUTPUT"],
                "role": "executor"
            }
        }
        })
        self.save_blueprint({
            "id": "EVOLUTIVE_CODE",
            "name": "Evolutivo con Aider",
            "description": "Edición física con validación posterior.",
            "routes": {"main": {"steps": ["AIDER", "R1", "OUTPUT"], "role": "executor"}}
        })

    # =========================================================================
    # GESTIÓN DE CICLOS VIRTUALES
    # =========================================================================

    def create_cycle(self, blueprint_id: str, context: str = "") -> Optional[ActiveCycle]:
        """Instancia un ciclo lógico basado en un Blueprint."""
        if blueprint_id not in self.blueprints:
            logger.error(f"Blueprint {blueprint_id} no encontrado. Abortando.")
            return None
            
        cycle_id = f"CYC-{str(uuid.uuid4())[:6]}"
        new_cycle = ActiveCycle(
            id=cycle_id,
            blueprint_id=blueprint_id,
            status=CycleStatus.RUNNING,
            shared_context=context
        )
        
        self.active_cycles[cycle_id] = new_cycle
        logger.info(f"🌀 Ciclo Virtual Activado: {cycle_id} [{blueprint_id}]")
        self.save_state()
        return new_cycle

    def get_cycle(self, cycle_id: str) -> Optional[ActiveCycle]:
        return self.active_cycles.get(cycle_id)

    def get_journal(self, worker_id: str) -> NodeJournal:
        if worker_id not in self.journals:
            self.journals[worker_id] = NodeJournal(worker_id)
        return self.journals[worker_id]

    # =========================================================================
    # FÍSICA SEMÁNTICA (Movimiento y Superposición)
    # =========================================================================

    async def move_packet(self, packet: TaskPacket, target_worker: str) -> Optional[TaskPacket]:
        """
        Mueve un paquete al nodo físico. 
        Si hay colisión semántica, genera evento de SÍNTESIS sin sustituir.
        """
        async with self.lock:
            # 1. Hidratación Semántica (Capa Matemática)
            if packet.tensor_state is None:
                packet.tensor_state = self.narrator.embed(packet.content)
                packet.ui_color = self.narrator.tensor_to_rgb(packet.tensor_state)

            # 2. Desmaterializar de posición previa
            old_pos = (packet.coords["x"], packet.coords["y"])
            if self.physical_grid.get(old_pos) == packet.id:
                del self.physical_grid[old_pos]

            # 3. Escaneo de colisión y superposición en el Nodo Físico
            target_y = 0
            while (target_worker, target_y) in self.physical_grid:
                resident_id = self.physical_grid[(target_worker, target_y)]
                resident_pkt = self.packets.get(resident_id)
                
                if resident_pkt and resident_pkt.tensor_state is not None:
                    cohesion = self.narrator.calculate_cohesion(packet.tensor_state, resident_pkt.tensor_state)
                    
                    # REGLA DE INTERFERENCIA (>0.85): No sustituye, genera estudio paralelo
                    if cohesion > 0.85:
                        logger.info(f"✨ Superposición Semántica ({cohesion:.2f}) en {target_worker}")
                        return self._create_synthesis_packet(packet, resident_pkt, target_worker)

                target_y += 1 # Apilamiento en la cola física (Eje Y)
                if target_y > 50: break

            # 4. Materialización en el Grid Físico
            packet.coords["x"] = target_worker
            packet.coords["y"] = target_y
            self.physical_grid[(target_worker, target_y)] = packet.id
            self.packets[packet.id] = packet
            
            # 5. Sincronización con VRAM (Taichi)
            self._sync_vram(target_worker, target_y, packet)
            
            self.save_state()
            return None

    def _create_synthesis_packet(self, p1: TaskPacket, p2: TaskPacket, worker: str) -> TaskPacket:
        """Crea un paquete de interferencia para que un nodo paralelo los estudie."""
        return TaskPacket(
            id=f"SYN-{str(uuid.uuid4())[:4]}",
            content=f"ESTUDIO DE SUPERPOSICIÓN:\n- A: {p1.content[:60]}\n- B: {p2.content[:60]}",
            cycle_id=p1.cycle_id,
            route_id="synthesis",
            tags=["fusion_event", "quantum_interference"],
            params={"parent_a": p1.id, "parent_b": p2.id, "worker": worker}
        )

    def _sync_vram(self, worker_id: str, y: int, pkt: TaskPacket):
        """Mapea la topología lógica a la física del reactor Taichi."""
        worker_map = {"ROUTER": 0.1, "AIDER": 0.3, "R1": 0.5, "CORE": 0.7, "OUTPUT": 0.9}
        x_norm = worker_map.get(worker_id, 0.5)
        y_norm = min(1.0, y / 10.0) 
        z_norm = pkt.coords.get("z", 0.5)
        
        try:
            # Enviamos color y posición a la Memoria Triple
            TripleMemory.get().sync_grid_tensor(x_norm, y_norm, z_norm, pkt.ui_color or (200,200,200))
        except Exception: pass 

    # =========================================================================
    # EXPOSICIÓN ESPECTRAL (Snapshot para el WebSocket Server)
    # =========================================================================

    def get_spectral_snapshot(self) -> Dict[str, Any]:
        """Snapshot completo del Grid para el Dashboard."""
        snapshot = {
            "physical_grid": [], 
            "active_blueprints": list(self.blueprints.keys()),
            "cycles": {}        
        }
        
        for (x, y), pid in self.physical_grid.items():
            pkt = self.packets.get(pid)
            if pkt:
                snapshot["physical_grid"].append({
                    "id": pkt.id, "x": x, "y": y, "z": pkt.coords.get("z", 0.5),
                    "color": pkt.ui_color, "cycle_id": pkt.cycle_id,
                    "snippet": pkt.content[:40],
                    "type": "fusion" if "fusion_event" in pkt.tags else "task"
                })

        for cid, cycle in self.active_cycles.items():
            if cycle.status == CycleStatus.RUNNING:
                snapshot["cycles"][cid] = {
                    "blueprint": cycle.blueprint_id,
                    "age": time.time() - cycle.start_time
                }
        return snapshot

    # =========================================================================
    # PERSISTENCIA
    # =========================================================================

    def save_state(self):
        """Persiste el estado de los ciclos y journals."""
        try:
            data = {
                "active_cycles": {
                    cid: {
                        "id": c.id, "blueprint": c.blueprint_id, 
                        "status": c.status.value, "context": c.shared_context
                    } for cid, c in self.active_cycles.items()
                },
                "journals": {
                    wid: {
                        "worker_id": j.worker_id,
                        "log": [asdict(e) for e in j.log] # Quirúrgico: Dataclass a Dict para JSON
                    } for wid, j in self.journals.items()
                }
            }
            with open(self.persistence_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error persistiendo estado del tensor: {e}")

    def load_state(self):
        """Restaura el estado del tensor tras un reinicio."""
        if not self.persistence_file.exists(): return
        try:
            with open(self.persistence_file, "r") as f:
                data = json.load(f)
            
            for cid, cdata in data.get("active_cycles", {}).items():
                self.active_cycles[cid] = ActiveCycle(
                    id=cdata["id"],
                    blueprint_id=cdata["blueprint"],
                    status=CycleStatus(cdata["status"]),
                    shared_context=cdata.get("context", "")
                )
            
            # Importación local para evitar la circularidad
            from core.types import WorkerEntry 
            for wid, jdata in data.get("journals", {}).items():
                j = NodeJournal(wid)
                j.log = [WorkerEntry(**e) for e in jdata.get("log", [])]
                self.journals[wid] = j
                
            logger.info(f"🌌 Topología restaurada: {len(self.active_cycles)} ciclos vivos.")
        except Exception as e:
            logger.error(f"Error restaurando topología: {e}")
