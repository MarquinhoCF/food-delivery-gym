"""Deterministic clone of a live FoodDelivery SimPy/Gym environment.

SimPy processes are backed by Python generators, which cannot be deep-copied.
This module snapshots pending waits from the SimPy queue, copies domain state,
and relaunches equivalent processes from named resume points.
"""

from __future__ import annotations

import copy
import pickle
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from simpy.core import Environment as SimpyEnvironment
from simpy.events import Initialize, Process, Timeout

from food_delivery_gym.main.environment.food_delivery_simpy_env import FoodDeliverySimpyEnv
from food_delivery_gym.main.generator.initial_generator import InitialGenerator
from food_delivery_gym.main.generator.poisson_order_generator import PoissonOrderGenerator
from food_delivery_gym.main.utils.rng_factory import RngFactory

ENV_REF_ATTRS = ("_environment", "environment", "enviroment")


@dataclass
class WakeSpec:
    eid: int
    time: float
    priority: int
    remaining: float
    co_name: str
    owner_id: int
    chain_names: list[str]
    object_ids: dict[str, int] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def _should_skip_walk(obj: Any) -> bool:
    if obj is None or isinstance(obj, (int, float, str, bytes, bool, complex, np.generic)):
        return True
    if isinstance(obj, (np.ndarray, np.random.Generator, np.random.SeedSequence)):
        return True
    if isinstance(obj, SimpyEnvironment):
        return True
    if type(obj).__name__ == "generator":
        return True
    return False


def _strip_simpy_refs(root: Any) -> list[tuple[Any, str, Any]]:
    stripped: list[tuple[Any, str, Any]] = []
    seen: set[int] = set()

    def walk(obj: Any) -> None:
        if _should_skip_walk(obj):
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(obj, (list, tuple, deque, set, frozenset)):
            for item in obj:
                walk(item)
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(key)
                walk(value)
            return

        data = getattr(obj, "__dict__", None)
        if data is None:
            return
        for attr in ENV_REF_ATTRS:
            if attr in data and isinstance(data[attr], SimpyEnvironment):
                stripped.append((obj, attr, data[attr]))
                data[attr] = None
        for value in list(data.values()):
            walk(value)

    walk(root)
    return stripped


def _restore_stripped(stripped: list[tuple[Any, str, Any]]) -> None:
    for obj, attr, value in stripped:
        setattr(obj, attr, value)


def _bind_new_env(root: Any, new_env: FoodDeliverySimpyEnv) -> None:
    seen: set[int] = set()

    def walk(obj: Any) -> None:
        if _should_skip_walk(obj):
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(obj, (list, tuple, deque, set, frozenset)):
            for item in obj:
                walk(item)
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(key)
                walk(value)
            return

        data = getattr(obj, "__dict__", None)
        if data is None:
            return
        for attr in ENV_REF_ATTRS:
            if attr in data:
                data[attr] = new_env
        for value in list(data.values()):
            walk(value)

    walk(root)


def _callback_owner_process(event) -> Optional[Process]:
    for callback in event.callbacks or []:
        owner = getattr(callback, "__self__", None)
        if isinstance(owner, Process):
            return owner
    return None


def _process_chain(event) -> list[Process]:
    chain: list[Process] = []
    current = event
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        process = _callback_owner_process(current)
        if process is None:
            break
        chain.append(process)
        current = process
    return chain


def _innermost_generator(process: Process, *, skip_private: bool = True):
    generator = process._generator
    seen: set[int] = set()
    while generator is not None and id(generator) not in seen:
        seen.add(id(generator))
        inner = getattr(generator, "gi_yieldfrom", None)
        if inner is None:
            break
        inner_name = inner.gi_code.co_name if getattr(inner, "gi_code", None) else ""
        if skip_private and inner_name.startswith("_"):
            break
        generator = inner
    return generator


def _generator_name(process: Process) -> str:
    generator = _innermost_generator(process)
    if generator is None:
        return "<none>"
    return generator.gi_code.co_name


def _generator_self(process: Process):
    generator = _innermost_generator(process)
    if generator is None or generator.gi_frame is None:
        return None
    return generator.gi_frame.f_locals.get("self")


def _generator_locals(process: Process) -> dict:
    generator = _innermost_generator(process)
    if generator is None or generator.gi_frame is None:
        return {}
    return dict(generator.gi_frame.f_locals)


def _canonical_process_name(name: str) -> str:
    if name.startswith("resume_"):
        return name[len("resume_"):]
    return name


def _plain_extra(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _prepare_order_phase(process: Process) -> tuple[str, dict]:
    generator = process._generator
    loc = _generator_locals(process)
    inner = getattr(generator, "gi_yieldfrom", None)
    if inner is not None and getattr(inner, "gi_code", None) is not None:
        if inner.gi_code.co_name == "_handle_driver_allocation":
            if "remaining_time_to_prepare" in loc:
                return "alloc_wait", {"remaining_prep": loc["remaining_time_to_prepare"]}
            return "excess_alloc", {}
    if "remaining_time_to_prepare" in loc:
        return "remaining_prep", {}
    extras = {}
    if "time_to_allocate_driver" in loc and "time_to_prepare" in loc:
        extras["excess_alloc"] = loc["time_to_allocate_driver"] - loc["time_to_prepare"]
    return "prep_before_excess", extras

"""
Como generators Python não podem ser copiados, precisamos capturar o estado de cada processo SimPy e reconstituí-los no clone. 
O SimPy mantém uma fila de eventos futuros em env._queue: tuplas (time, priority, eid, event). Cada event pendente (tipicamente 
um Timeout) está associado, através de callbacks, a um Process que está "dormindo" esperando aquele tempo passar.
A função `capture_wakes` varre essa fila e, para cada processo dormindo, produz um WakeSpec, um retrato do que é preciso para 
recriá-lo depois.
"""
def capture_wakes(env: FoodDeliverySimpyEnv) -> list[WakeSpec]:
    wakes: list[WakeSpec] = []
    seen_outer: set[int] = set()

    for time, priority, eid, event in list(env._queue):
        if isinstance(event, Initialize):
            continue # Ignora eventos de inicialização (não representam um processo "esperando algo").

        # Sobe a cadeia de callbacks (event -> processo dono -> evento que ele espera -> processo dono desse...) 
        # até achar o processo mais "externo" (o que efetivamente está em espera de tempo, tipo um 
        # yield env.timeout(...)). Isso evita capturar o mesmo processo mais de uma vez quando vários eventos 
        # internos apontam pro mesmo processo raiz.
        chain = _process_chain(event)
        if not chain:
            continue
        outermost = chain[-1]
        if id(outermost) in seen_outer:
            continue
        seen_outer.add(id(outermost))

        owner = _generator_self(outermost) # Dono do método generator (O objeto self, ex.: uma instância de Driver, Cook, PoissonOrderGenerator...)
        raw_name = _generator_name(outermost) # Nome do método generator (ex.: "prepare_order", "picking_up", "generate", etc)
        co_name = _canonical_process_name(raw_name) # Nome do método normalizado (removendo o prefixo "resume_" se o processo já era, ele mesmo, resultado de um resume anterior).
        if owner is None:
            raise RuntimeError(f"Processo SimPy sem 'self': {raw_name}")

        # Geradores "iniciais" que só rodam uma vez (não recorrentes) são descartados 
        # Apenas o PoissonOrderGenerator precisa ser recapturado pois é recorrente
        if co_name == "generate" and isinstance(owner, InitialGenerator) and not isinstance(owner, PoissonOrderGenerator):
            continue

        # Calcula quanto tempo ainda falta para o evento disparar, em vez de guardar o tempo absoluto
        remaining = max(0.0, float(time) - float(env.now))
        object_ids: dict[str, int] = {}
        extras: dict[str, Any] = {}
        loc = _generator_locals(outermost)
        innermost_name = _canonical_process_name(_generator_name(chain[0]))

        # Usando _generator_locals (que lê gi_frame.f_locals do generator), ele pega as variáveis locais do processo 
        # pausado e guarda o id() dos objetos de domínio referenciados (não os objetos em si). Isso é essencial 
        # porque, depois do deepcopy, esses objetos originais viram cópias novas, o id() original serve de chave num 
        # dicionário memo (id(original) -> cópia) para remapear a referência certa depois.
        for key in ("order", "cook", "route_segment", "driver"):
            if key in loc and loc[key] is not None:
                object_ids[key] = id(loc[key])

        # Destination é tratado à parte: se for uma tupla de coordenadas (imutável, sem identidade problemática) é guardada 
        # direto como valor; senão guardada por id() também.
        if "destination" in loc and loc["destination"] is not None:
            dest = loc["destination"]
            if isinstance(dest, tuple):
                extras["destination"] = dest
            else:
                object_ids["destination"] = id(dest)

        # Capturando o "esatado fino" de cada processo:
        # Cada tipo de processo (co_name) precisa de informação extra específica para saber exatamente em que ponto retomar, 
        # porque um mesmo método pode ter múltiplos pontos de espera:

        if co_name == "prepare_order":
            if loc.get("phase") in {"alloc_wait", "remaining_prep", "prep_before_excess", "excess_alloc"}:
                phase = loc["phase"]
                extras["phase"] = phase
                if phase == "alloc_wait" and loc.get("remaining_prep") is not None:
                    extras["remaining_prep"] = loc["remaining_prep"]
                if phase in {"prep_before_excess", "excess_alloc"} and loc.get("excess_alloc") is not None:
                    extras["excess_alloc"] = loc["excess_alloc"]
            else:
                phase, phase_extras = _prepare_order_phase(outermost)
                extras["phase"] = phase
                extras.update(phase_extras)

        elif co_name == "picking_up":
            extras["phase"] = loc.get("phase") or ("moving" if innermost_name == "move_to" else "waiting_ready")

        elif co_name == "process_accepted_orders":
            cook = loc.get("cook")
            if cook in getattr(owner, "cooks", []):
                extras["cook_index"] = owner.cooks.index(cook)
            elif "cook_index" in loc:
                extras["cook_index"] = int(loc["cook_index"])
            else:
                extras["cook_index"] = 0

        elif co_name == "generate" and isinstance(owner, PoissonOrderGenerator):
            extras["arrival_index"] = owner.current_order_id - 1

        elif co_name == "sequential_processor":
            segment = getattr(owner, "current_route_segment", None) or loc.get("route_segment")
            if segment is not None:
                object_ids["route_segment"] = id(segment)

        extras = {key: _plain_extra(value) for key, value in extras.items()}

        # Guarda tudo: quando o evento aconteceria, prioridade (para desempate determinístico no SimPy), quanto 
        # tempo falta, qual método/processo é, quem é o dono (por id, para remapear depois), a cadeia de nomes dos 
        # processos aninhados, e os object_ids/extras específicos da fase.
        wakes.append(WakeSpec(
            eid=int(eid),
            time=float(time),
            priority=int(priority),
            remaining=remaining,
            co_name=co_name,
            owner_id=id(owner),
            chain_names=[_canonical_process_name(_generator_name(p)) for p in chain],
            object_ids=object_ids,
            extras=extras,
        ))

    # Ordena para reconstrução determinística, replicando a ordem de desempate que o próprio SimPy usaria.
    wakes.sort(key=lambda spec: (spec.time, spec.priority, spec.eid))
    return wakes


def _mapped(memo: dict, object_id: Optional[int], label: str):
    if object_id is None:
        return None
    if object_id not in memo:
        raise RuntimeError(f"Clone: objeto '{label}' (id={object_id}) não foi copiado")
    return memo[object_id]


def _make_resume_generator(spec: WakeSpec, owner, memo: dict, new_env: FoodDeliverySimpyEnv):
    remaining = spec.remaining
    name = spec.co_name
    extras = spec.extras
    object_ids = spec.object_ids

    order = _mapped(memo, object_ids.get("order"), "order")
    cook = _mapped(memo, object_ids.get("cook"), "cook")
    route_segment = _mapped(memo, object_ids.get("route_segment"), "route_segment")

    if name == "process_order_requests":
        return owner.resume_process_order_requests(remaining)
    
    if name == "process_order_request":
        return owner.resume_process_order_request(order, remaining)
    
    if name == "process_accepted_orders":
        return owner.resume_process_accepted_orders(remaining, extras.get("cook_index", 0))
    
    if name == "prepare_order":
        return owner.resume_prepare_order(
            cook,
            order,
            extras["phase"],
            remaining,
            remaining_prep=extras.get("remaining_prep"),
            excess_alloc=extras.get("excess_alloc"),
        )
    
    if name == "process_route_requests":
        return owner.resume_process_route_requests(remaining)
    
    if name == "sequential_processor":
        if route_segment is None:
            route_segment = owner.current_route_segment
        return owner.resume_sequential_processor(remaining, route_segment)
    
    if name == "picking_up":
        if order is None and owner.current_route_segment is not None:
            order = owner.current_route_segment.order
        return owner.resume_picking_up(order, extras["phase"], remaining)
    
    if name == "delivering":
        if order is None and owner.current_route_segment is not None:
            order = owner.current_route_segment.order
        return owner.resume_delivering(order, remaining)
    
    if name == "move_to":
        destination = extras.get("destination") or _mapped(memo, object_ids.get("destination"), "destination")
        if destination is None and owner.current_route_segment is not None:
            destination = owner.current_route_segment.coordinate
        return owner.move_to(destination, resume_remaining=remaining)
    
    if name == "wait_customer_pick_up_order":
        if order is None and owner.current_route_segment is not None:
            order = owner.current_route_segment.order
        return owner.resume_wait_customer_pick_up_order(order, remaining)
    
    if name == "receive_order":
        driver = _mapped(memo, object_ids.get("driver"), "driver")
        return owner.resume_receive_order(order, driver, remaining)
    
    if name == "generate" and isinstance(owner, PoissonOrderGenerator):
        return owner.resume_generate(new_env, remaining, extras["arrival_index"])

    raise RuntimeError(
        f"Processo SimPy sem resume implementado: {name} (cadeia={spec.chain_names}, "
        f"tipo={type(owner).__name__})"
    )


def _clone_numpy_rng(state_blob: bytes) -> np.random.Generator:
    bit_generator = np.random.PCG64()
    bit_generator.state = pickle.loads(state_blob)
    return np.random.Generator(bit_generator)


def _collect_rng_owners(src: FoodDeliverySimpyEnv):
    owners = [src.map, *src.generators, *src.state.customers, *src.state.establishments, *src.state.drivers]
    return [owner for owner in owners if owner is not None and getattr(owner, "rng", None) is not None]


def _snapshot_rng_states(src: FoodDeliverySimpyEnv) -> dict[int, bytes]:
    return {id(owner): pickle.dumps(owner.rng.bit_generator.state) for owner in _collect_rng_owners(src)}


def _apply_rng_states(owners_and_clones, states: dict[int, dict]) -> None:
    for source_id, cloned in owners_and_clones:
        if source_id not in states:
            continue
        cloned.rng = _clone_numpy_rng(states[source_id])


def clone_simpy_env(src: FoodDeliverySimpyEnv) -> tuple[dict, FoodDeliverySimpyEnv]:
    # Captura um "retrato" de todos os processos SimPy pendentes (quem são, em que fase, quanto tempo falta)
    wakes = capture_wakes(src)

    rng_states = _snapshot_rng_states(src) # Salva o estado interno dos RNGs numpy de cada "dono"
    factory_seq = copy.deepcopy(src.rng_factory._seed_sequence) if src.rng_factory is not None else None # Copia a seed sequence da fábrica de RNGs

    # Agrupa em um dict só as partes do estado do ambiente que precisam ser deep-copiadas
    bundle = {
        "map": src.map,
        "generators": src.generators,
        "state": src._state,
        "core_events": src.core_events,
    }
    # Generators/objetos SimPy dentro desse bundle guardam referências de volta para o SimpyEnvironment original.
    # Essas referências não são deep-copiáveis, então são removidas
    stripped = _strip_simpy_refs(bundle)

    memo: dict = {} # dicionário de deepcopy: mapeia id(objeto original) -> objeto copiado.
    try:
        copied = copy.deepcopy(bundle, memo) # Faz a cópia profunda de fato do estado do domínio
    finally:
        # Restaura as referências ao ambiente SimPy no objeto ORIGINAL (`src`), já que elas foram zeradas antes do deepcopy 
        # e precisam voltar ao normal.
        _restore_stripped(stripped)
        # O deepcopy pode ter consumido o estado interno dos RNGs do original; aqui eles são recriados, garantindo que `src` 
        # continue intacto.
        for owner in _collect_rng_owners(src):
            owner.rng = _clone_numpy_rng(rng_states[id(owner)])

    # Cria um novo ambiente SimPy "vazio" (initialize=False evita já disparar a lógica normal de inicialização/agendamento, 
    # pois vamos montar o estado manualmente a partir do clone).
    rng_factory = RngFactory(seed_sequence=factory_seq) if factory_seq is not None else None
    new_env = FoodDeliverySimpyEnv(
        map=copied["map"],
        generators=copied["generators"],
        optimizer=None,
        view=None,
        rng_factory=rng_factory,
        initialize=False,
    )
    new_env._now = src.now
    new_env._state = copied["state"]
    new_env.core_events = copied["core_events"]
    new_env.env_mode = src.env_mode
    new_env.last_time_step = src.last_time_step

    # Agora que `new_env` existe, percorre toda a estrutura copiada e substitui as referências de ambiente (que foram zeradas 
    # antes do deepcopy) apontando para o NOVO ambiente clonado, assim os objetos do domínio "sabem" que agora pertencem a 
    # `new_env`, não a `src`.
    _bind_new_env(copied, new_env)

    # Restaura, nos objetos JÁ CLONADOS, o estado dos RNGs salvo no início, assim o clone tem RNGs com o mesmo estado que o 
    # original tinha no momento da captura, garantindo geração de números aleatórios determinística e independente
    # a partir daqui.
    _apply_rng_states(
        ((id(owner), memo[id(owner)]) for owner in _collect_rng_owners(src) if id(owner) in memo),
        rng_states,
    )

    # Para cada processo capturado em `wakes`, recria um generator equivalente a partir do ponto de execução salvo (fase, tempo 
    # restante, objetos referenciados) e o agenda no novo ambiente.
    for spec in wakes:
        owner = _mapped(memo, spec.owner_id, spec.co_name)
        resume_gen = _make_resume_generator(spec, owner, memo, new_env)
        new_env.process(resume_gen)

    # `env.process()` só agenda um evento `Initialize` urgente para cada processo; aqui esses eventos são "descarregados" sem avançar 
    # o relógio da simulação, fazendo os processos recém-criados chegarem até o mesmo ponto de espera (ex.: aguardando um timeout) 
    # que os processos originais estavam.
    _flush_initialize_events(new_env)
    return memo, new_env


def _flush_initialize_events(env: FoodDeliverySimpyEnv) -> None:
    """
    Inicia processos retomados sem avançar o tempo de simulação.

    `Environment.process` apenas agenda um evento urgente do tipo `Initialize`.
    O processamento desses eventos faz com que o clone aguarde pelos mesmos timeouts
    que a origem; assim, um clone de um clone pode reproduzir esperas equivalentes.
    """
    while env._queue:
        time, _priority, _eid, event = env._queue[0]
        if time != env.now or not isinstance(event, Initialize):
            break
        env.step()


def clone_gym_env(env):
    from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv

    if env.simpy_env is None:
        raise RuntimeError("clone() requer um ambiente após reset()")

    cloned = FoodDeliveryGymEnv.__new__(FoodDeliveryGymEnv)
    for key, value in env.__dict__.items():
        # Cópia shallow: Cópia rasa do dicionário de atributos. 
        # Cada atributo é copiado por referência, não por valor
        setattr(cloned, key, value)

    # A partir daqui, o clone precisa de um SimPy Environment separado
    # Para não compartilhar a mesma referência
    memo, new_simpy = clone_simpy_env(env.simpy_env)
    cloned.simpy_env = new_simpy
    cloned.last_simpy_env = None
    cloned.render_mode = None
    cloned.order_generator_config = copy.deepcopy(env.order_generator_config)

    if env.current_order is not None:
        cloned.current_order = memo[id(env.current_order)]
    else:
        cloned.current_order = None

    if env._cached_busy_times is not None:
        cloned._cached_busy_times = np.copy(env._cached_busy_times)

    return cloned
