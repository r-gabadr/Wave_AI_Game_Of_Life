import asyncio
import logging
import uuid
import json
import time
from typing import Dict, Any, Optional, List, TYPE_CHECKING

# Imports de infraestructura - Aseguramos compatibilidad con el Grid y la Memoria
from core.agent import WAGAgentCore
from core.types import TaskPacket, ActiveCycle, CycleStatus, RouteBlueprint, PacketStep
from core.tools import SmartSearcher
from grid.system import WAGGridSystem
from data.memory_manager import TripleMemory
from data.narrator import get_narrator
from services.gemini_cli_worker import GeminiCLIWorker
from config import settings
from pathlib import Path

logger = logging.getLogger("WAG-Director")

class WAGMasterAgent:
    """
    El Director de Orquesta y Narrador (WAG Master Agent).
    Gestiona el tráfico de NPCs (Nodos Virtuales) sobre Workers Físicos.
    Nivel de Orquestación: Controla la convergencia entre el Grid Tensorial y el Pensamiento.
    """
    def __init__(self, core: WAGAgentCore):
        self.core = core
        self.grid = WAGGridSystem()
        self.narrator = get_narrator()
        self.smart_searcher = SmartSearcher(core.aider, Path(settings.STORAGE_DIR))
        self.running = False
        
        # Colas de Recursos: Diferenciamos entre el bus lógico (ROUTER) y la salida (OUTPUT)
        self.queues = {
            "ROUTER": asyncio.Queue(),
            "OUTPUT": asyncio.Queue()
        }
        
        # Registro de workers activos en memoria (Mapeo de carga en RAM/VRAM)
        self.active_workers = {}

    async def start(self):
        """Arranca el Kernel de Orquestación y activa los servicios base de la topología."""
        self.running = True
        logger.info("🚀 WAG Grid System: Kernel Activo. Iniciando servicios espectrales.")
        
        # 1. Servicios Base (Ciclos de vida del sistema)
        asyncio.create_task(self.grid_state_reporter())
        asyncio.create_task(self.worker_router())
        asyncio.create_task(self.worker_output())
        asyncio.create_task(self.memory_consolidator()) # Procesamiento REM

        # 2. Despliegue Inicial de Recursos Físicos
        await self.deploy_worker("R1", "r1", "deepseek-r1")
        await self.deploy_worker("aider", "aider", "lora")
        await self.deploy_worker("gemini_cli", "gemini_cli", "default_cli_model")

    # =========================================================================
    # EL CEREBRO: ROUTER (Kernel de Orquestación y Navegación Geodésica)
    # =========================================================================
    async def reconfigure_worker(self, worker_id: str, model_path: str, lora: str = "lora"):
        """Reconfigura un micronodo GGUF en caliente."""
        # Se accede a self.active_workers, que existe en esta clase
        if worker_id in self.active_workers:
            logger.info(f"🔄 Reconfigurando worker {worker_id} -> {model_path}")
            
            self.active_workers[worker_id]["status"] = "loading"
            
            # Aquí va la lógica real de carga si es necesaria
            await asyncio.sleep(1) 
            
            self.active_workers[worker_id]["model"] = model_path
            self.active_workers[worker_id]["status"] = "idle"
            return True
        return False
    async def worker_router(self):
        """
        Kernel que gestiona el salto entre NPCs y la hidratación semántica.
        1. Ejecuta Herramientas Lógicas (WEB) como transformaciones inline.
        2. Hidrata al NPC con contexto RAG dinámico (Top-K).
        3. Gestiona la Física del Grid (Colisiones) y el Handoff a Workers.
        """
        while self.running:
            packet = await self.queues["ROUTER"].get()
            try:
                # 1. Recuperar la configuración del NPC para el nodo actual de la ruta
                node_cfg = self._get_node_config(packet)
                target = node_cfg.get("worker_type", "R1")
                
                logger.info(f"🎯 [ROUTER] Paquete {packet.id} en paso {packet.current_step_index}. Destino: {target}")

                # --- CONDICIÓN DE SALIDA DURA (Frontera del Sistema) ---
                if target == "OUTPUT":
                    logger.info(f"🏁 [ROUTER] Ciclo {packet.cycle_id} alcanzando nodo terminal.")
                    await self.queues["OUTPUT"].put(packet)
                    continue

                # --- CAPA LÓGICA: Pasos Inline / Transformaciones Instantáneas ---
                if target == "WEB":
                    search_limit = node_cfg.get("top_k", 3)
                    logger.info(f"🌐 [ROUTER] Ejecutando búsqueda WEB para {packet.id}")
                    await self.core._emit("status_update", {"text": f"🌐 [TOOL] Consultando DuckDuckGo para: {packet.content[:30]}..."})
                    
                    search_result = await self.core._search_web(packet.content)
                    
                    # WEAVING INLINE: Inyectamos el nuevo conocimiento
                    packet.content = f"CONTEXTO WEB ENCONTRADO:\n{search_result}\n\nPREGUNTA ORIGINAL: {packet.content}"
                    packet.node_history += f"\n[SYSTEM/WEB]: Búsqueda completada e integrada.\n"
                    
                    # AVANCE DE PUNTERO: Consumimos el paso para evitar bucles infinitos
                    packet.current_step_index += 1
                    
                    # Re-evaluación inmediata en el Router para el siguiente salto
                    await self.queues["ROUTER"].put(packet)
                    continue

                # --- CAPA FÍSICA: Handoff a Worker Real (RAM/VRAM) ---
                # 2. HIDRATACIÓN SEMÁNTICA: El NPC decide su propia profundidad de RAG
                top_k = node_cfg.get("top_k", 5)
                context, _, _ = TripleMemory.get().retrieve(packet.content, limit=top_k)
                packet.context = context

                # 3. FÍSICA DE GRID: Movimiento de la partícula de información y detección de colisiones
                logger.info(f"🗺️ [ROUTER] Moviendo {packet.id} a Grid de {target}...")
                collision_event = await self.grid.move_packet(packet, target)

                if collision_event:
                    # GESTIÓN DE COLISIÓN: Superposición de capas de pensamiento
                    logger.warning(f"⚡ [ROUTER] Colisión detectada: {packet.id} en {target}")
                    await self.core._emit("status_update", {
                        "text": f"💥 Superposición semántica en {target}. Iniciando síntesis..."
                    })
                    collision_event.tags.append("fusion_event")
                    self.grid.packets[collision_event.id] = collision_event 
                    
                    # ARBITRAJE: El paquete de síntesis se envía al pool de razonamiento (R1)
                    arbiter_queue = "R1" if "R1" in self.queues else target
                    await self.queues[arbiter_queue].put(collision_event)
                else:
                    # FLUJO NORMAL: Inyectamos el 'npc_setup' para el worker físico
                    if target in self.queues:
                        packet.params["npc_setup"] = node_cfg
                        logger.info(f"📤 [ROUTER] Inyectando paquete {packet.id} en cola de {target}")
                        await self.queues[target].put(packet)
                    else:
                        logger.error(f"❌ [ROUTER] Worker {target} no tiene cola activa.")
                        await self.queues["OUTPUT"].put(packet)

            except Exception as e:
                logger.error(f"❌ [ROUTER] Error crítico procesando paquete: {e}", exc_info=True)
            finally:
                self.queues["ROUTER"].task_done()

    def _get_node_config(self, packet: TaskPacket) -> Dict[str, Any]:
        """Extrae la ficha del NPC, priorizando rutas personalizadas o planos registrados."""
        # 1. Prioridad: Geodésica Forzada (custom_steps) enviada desde la interfaz
        custom_steps = packet.params.get("custom_steps")
        if custom_steps and packet.current_step_index < len(custom_steps):
            step = custom_steps[packet.current_step_index]
            if isinstance(step, str):
                return {"worker_type": step, "role": "executor", "top_k": 5}
            return step

        # 2. Fallback: Blueprints registrados en la biblioteca de la Grid
        if packet.cycle_id:
            cycle = self.grid.get_cycle(packet.cycle_id)
            if cycle:
                blueprint = self.grid.blueprints.get(cycle.blueprint_id)
                route = blueprint.routes.get(packet.route_id, blueprint.routes.get("main"))
                if route and packet.current_step_index < len(route.steps):
                    step_data = route.steps[packet.current_step_index]
                    if isinstance(step_data, dict): return step_data
                    return {"worker_type": step_data, "role": "assistant", "top_k": 5}
        
        # 3. Fin de ruta detectado
        return {"worker_type": "OUTPUT", "role": "fallback"}

    # =========================================================================
    # WORKERS: EJECUCIÓN CON WEAVING NARRATIVO (Conciencia de NPCs)
    # =========================================================================

    async def _generic_r1_loop(self, worker_id: str, model_id: str):
        """Worker R1: Encarna al NPC y gestiona el razonamiento + Bucle recursivo."""
        logger.info(f"🟢 Worker físico {worker_id} ONLINE.")
        while self.running:
            try:
                packet = await self.queues[worker_id].get()
                self.active_workers[worker_id]["status"] = "busy"
                
                # 1. CARGAR MÁSCARA: Configuramos la identidad del NPC actual
                setup = packet.params.get("npc_setup", {})
                role = setup.get("role", "expert")
                logger.info(f"🧠 [{worker_id}] Encarnando NPC: {role} para paquete {packet.id}")
                
                # 2. RECUPERAR IDENTIDAD: Diario del Nodo
                journal = self.grid.get_journal(f"NPC_{role}") 
                node_memory = "\n".join([e.result_summary for e in journal.log[-5:]])
                
                # 3. WEAVING: Tejido narrativo inter-NPC
                prompt = f"""[SISTEMA: ROL DEL NODO]
Eres el agente: {role}. {setup.get('instructions', 'Analiza y responde coherentemente.')}

[TU MEMORIA CORTA]
{node_memory}

[CONTEXTO RAG]
{packet.context}

[HILO DEL CICLO]
{packet.node_history}

[INPUT ACTUAL]
{packet.content}

Instrucción final: Responde como {role} manteniendo la coherencia narrativa del flujo."""
                
                # 4. EJECUCIÓN: Gemelo Lingüístico R1
                response = await self.core._stream_r1_with_thinking(prompt)
                
                # --- LÓGICA DE RECURSIÓN: Bucle Inteligente ---
                if "REINTENTAR" in response.upper() or "ERROR_DETECTADO" in response.upper():
                    logger.info(f"🔄 [{worker_id}] {role} solicita retroceso en {packet.id}")
                    packet.current_step_index = max(0, packet.current_step_index - 1)
                else:
                    packet.node_history += f"\n<{role.upper()}>: {response[:150]}...\n"
                    packet.content = response 
                    packet.current_step_index += 1
                
                # 5. PERSISTENCIA
                journal.log_action(packet.id, packet.cycle_id or "ROOT", f"SPEECH_{role}", response[:100])
                
                if "fusion_event" in packet.tags:
                    packet.content = f"RESOLUCIÓN DE CONFLICTO POR {role}:\n{response}"
                
                logger.info(f"🔄 [{worker_id}] Pensamiento completado. Re-enrutando {packet.id}")
                await self.queues["ROUTER"].put(packet)
                
            except Exception as e:
                logger.error(f"❌ Error en Loop de razonamiento {worker_id}: {e}", exc_info=True)
            finally:
                self.active_workers[worker_id]["status"] = "idle"
                self.queues[worker_id].task_done()

    async def _generic_aider_loop(self, worker_id: str):
        """Worker Aider Físico: Mutación de archivos con persistencia evolutiva en Git."""
        logger.info(f"🟢 Worker físico {worker_id} ONLINE.")
        while self.running:
            try:
                packet = await self.queues[worker_id].get()
                self.active_workers[worker_id]["status"] = "busy"
                logger.info(f"🔧 [{worker_id}] Iniciando mutación física para {packet.id}")
                
                journal = self.grid.get_journal(worker_id)
                cycle_branch = f"feat/wag-{packet.id}"
                
                # ACCIÓN FÍSICA: Mutación vía Bridge
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self.core.aider.execute_one_shot,
                    packet.content, # instruction
                    packet.params.get("filename", "evolution.py"), # filename
                    packet.params.get("lora", "lora"), # adapter_family
                    True, # create_backup
                    cycle_branch, # cycle_branch
                    False # unload_after
                )
                
                if result:
                    logger.info(f"✅ [{worker_id}] Mutación exitosa en {cycle_branch}")
                    packet.node_history += f"\n[SYSTEM]: Código mutado físicamente en rama {cycle_branch}.\n"
                    packet.content = result
                else:
                    logger.warning(f"⚠️ [{worker_id}] Aider no produjo cambios en {packet.id}")

                packet.current_step_index += 1
                await self.queues["ROUTER"].put(packet)
                
            except Exception as e:
                logger.error(f"❌ Error en Loop de Aider {worker_id}: {e}", exc_info=True)
            finally:
                self.active_workers[worker_id]["status"] = "idle"
                self.queues[worker_id].task_done()

    # =========================================================================
    # INFRAESTRUCTURA DE RED Y PERSISTENCIA (Gestión de Carga)
    # =========================================================================

    async def deploy_worker(self, worker_id: str, worker_type: str, model_id: str):
        """Instancia recursos físicos y arranca sus ciclos de pensamiento."""
        if worker_id in self.queues: return
        self.queues[worker_id] = asyncio.Queue()
        self.active_workers[worker_id] = {"type": worker_type, "model": model_id, "status": "idle"}
        
        if worker_type == "r1":
            asyncio.create_task(self._generic_r1_loop(worker_id, model_id))
        elif worker_type == "aider":
            asyncio.create_task(self._generic_aider_loop(worker_id))
        elif worker_type == "gemini_cli":
            gemini_cli_worker_instance = GeminiCLIWorker(worker_id, self.core, self)
            asyncio.create_task(gemini_cli_worker_instance.loop(self.queues[worker_id]))
            
        await self.core._emit("infra_update", {"workers": self.active_workers})

    async def worker_output(self):
        """Nodo de salida final: Desmaterializa el paquete del Grid y emite el resultado."""
        while self.running:
            packet = await self.queues["OUTPUT"].get()
            try:
                async with self.grid.lock:
                    pos = (packet.coords["x"], packet.coords["y"])
                    if self.grid.physical_grid.get(pos) == packet.id:
                        del self.grid.physical_grid[pos]
                
                logger.info(f"🏁 [OUTPUT] Ciclo {packet.cycle_id} completado con éxito.")
                await self.core._emit("chat_complete", {"cycle_id": packet.cycle_id, "content": packet.content})
            except Exception as e:
                logger.error(f"❌ Error en salida de ciclo: {e}")
            finally:
                self.queues["OUTPUT"].task_done()

    async def grid_state_reporter(self):
        """Actualización espectral del Grid para el Dashboard (Taichi/UI)."""
        while self.running:
            try:
                snapshot = self.grid.get_spectral_snapshot()
                await self.core._emit("grid_update", {"grid": snapshot["physical_grid"]})
            except Exception: pass
            await asyncio.sleep(0.2)

    async def memory_consolidator(self):
        """SUEÑO REM: Cristalización de diarios en memoria a largo plazo."""
        while self.running:
            await asyncio.sleep(45)
            try:
                for wid, journal in self.grid.journals.items():
                    if journal.log:
                        recent = "\n".join([e.result_summary for e in journal.log[-5:]])
                        TripleMemory.get().ingest(
                            text=f"[SUEÑO REM - NODO {wid}]\n{recent}", 
                            role="system",
                            entropy=(0.1, 0.1)
                        )
                        logger.debug(f"💾 Memoria de {wid} cristalizada.")
            except Exception as e:
                logger.error(f"Error en consolidación REM: {e}")

    async def inject_task(self, content: str, params: Dict):
        """Inyecta una tarea. Maneja 'Fast Lane' para blueprints especiales."""
        bp_id = params.get("blueprint_id", "SIMPLE_CHAT")

        # --- FAST LANE: AIDER_CON_BUSQUEDA ---
        if bp_id == "AIDER_CON_BUSQUEDA":
            logger.info(f"🚀 [FAST LANE] Iniciando Blueprint: {bp_id} para: '{content[:30]}...'")
            await self.core._emit("status_update", {"text": "Iniciando Búsqueda Activa con SML..."})
            
            # 1, 2, 3. Orquestación de Búsqueda Inteligente
            web_context = await self.smart_searcher.research(content)
            
            final_prompt = content
            if web_context:
                final_prompt = f"{web_context}\n\nTAREA:\n{content}"
                await self.core._emit("status_update", {"text": "Contexto web inyectado. Llamando a Aider..."})
            else:
                await self.core._emit("status_update", {"text": "No se encontró contexto relevante. Usando prompt original."})

            # 4. Ejecución directa en Aider
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.core.aider.execute_one_shot,
                final_prompt,
                params.get("filename", "fast_lane_task.py"),
                "r1_vllm", # Usar el worker principal
                True, # create_backup
                f"feat/fast-lane-{str(uuid.uuid4())[:6]}" # cycle_branch
            )
            
            # 5. Emisión del resultado y finalización
            await self.core._emit("chat_complete", {"cycle_id": f"fast-lane-{str(uuid.uuid4())[:6]}", "content": result or "Aider no produjo cambios."})
            logger.info(f"🏁 [FAST LANE] {bp_id} completado.")
            return # Cortocircuita el flujo normal del Grid

        # --- FLUJO NORMAL (GRID SYSTEM) ---
        cycle = self.grid.create_cycle(bp_id, context=params.get("context", ""))
        if not cycle:
            logger.error(f"Blueprint {bp_id} inexistente. Misión abortada.")
            return

        packet = TaskPacket(
            id=str(uuid.uuid4())[:8],
            content=content,
            cycle_id=cycle.id,
            route_id="main",
            current_step_index=0,
            params=params,
            coords={"x": "ROUTER", "y": 0, "z": 0.5}
        )
        self.grid.packets[packet.id] = packet
        logger.info(f"🚀 [INJECTOR] Paquete {packet.id} inyectado en ROUTER.")
        await self.queues["ROUTER"].put(packet)
