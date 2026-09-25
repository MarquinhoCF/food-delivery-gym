from abc import ABC, abstractmethod
from collections import defaultdict
import math
import os
import traceback
from typing import Any, List, Union

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer.optimizer import Optimizer
from food_delivery_gym.main.order.order import Order
from food_delivery_gym.main.route.delivery_route_segment import DeliveryRouteSegment
from food_delivery_gym.main.route.pickup_route_segment import PickupRouteSegment
from food_delivery_gym.main.route.route import Route
from food_delivery_gym.main.environment.state_log import format_step_result
from food_delivery_gym.main.statistics.simulation_stats import SimulationStats
from food_delivery_gym.main.statistics.boards.board import Board


def jsonable_hyperparameter(value: Any) -> Any:
    """Converte um hiperparâmetro recebido em valor serializável (JSON/NPZ)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, dict):
        return {str(key): jsonable_hyperparameter(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_hyperparameter(item) for item in value]

    described: dict[str, Any] = {"class": type(value).__name__}
    label = getattr(value, "label", None)
    if isinstance(label, str) and label:
        described["label"] = label
    for name, attr in vars(value).items():
        if name.startswith("_") or callable(attr):
            continue
        described[name] = jsonable_hyperparameter(attr)
    return described


def _write_hyperparameters(results_file, params: dict, indent: int = 0) -> None:
    prefix = "  " * indent
    for key, value in params.items():
        if isinstance(value, dict):
            results_file.write(f"{prefix}* {key}:\n")
            _write_hyperparameters(results_file, value, indent + 1)
        else:
            results_file.write(f"{prefix}* {key}: {value}\n")


class OptimizerGym(Optimizer, ABC):

    def __init__(self, environment: Union[FoodDeliveryGymEnv, VecEnv]):
        self.wrapped_env = environment
        self.gym_env = self._unwrap_environment(environment)
        self.state = None
        self.done = False
        self.truncated = False
        self.is_vectorized = isinstance(environment, VecEnv)

    # ========================================================
    #     Funções para suporte de ambientes vectorizados
    # ========================================================

    # Desembrulha o ambiente para acessar o FoodDeliveryGymEnv original.
    def _unwrap_environment(self, env) -> FoodDeliveryGymEnv:
        current_env = env
        
        # Se for um ambiente vectorizado
        if isinstance(current_env, VecEnv):
            # Para VecEnv, precisamos acessar o ambiente base
            if hasattr(current_env, 'venv'):
                current_env = current_env.venv
            
            # Se ainda for VecEnv, tenta acessar os envs individuais
            if isinstance(current_env, VecEnv):
                if hasattr(current_env, 'envs') and len(current_env.envs) > 0:
                    current_env = current_env.envs[0]
                elif hasattr(current_env, 'env'):
                    current_env = current_env.env
        
        # Desembrulha wrappers normais
        while hasattr(current_env, 'env') and not isinstance(current_env, FoodDeliveryGymEnv):
            current_env = current_env.env
        
        # Verifica se conseguiu encontrar o ambiente original
        if not isinstance(current_env, FoodDeliveryGymEnv):
            raise ValueError(
                f"Não foi possível encontrar o FoodDeliveryGymEnv original. "
                f"Ambiente encontrado: {type(current_env)}"
            )
        
        return current_env

    """
        Chama um método no ambiente, lidando com ambientes vectorizados.
        
        Args:
            method_name: Nome do método a ser chamado
            *args: Argumentos posicionais
            **kwargs: Argumentos nomeados
            
        Returns:
            Resultado do método
    """
    def _call_env_method(self, method_name: str, *args, **kwargs):
        if self.is_vectorized:
            # Para ambientes vectorizados, usa env_method se disponível
            if hasattr(self.wrapped_env, 'env_method'):
                try:
                    results = self.wrapped_env.env_method(method_name, *args, **kwargs)
                    return results[0] if isinstance(results, list) and len(results) > 0 else results
                except AttributeError:
                    # Se o método não existir no wrapper, tenta no ambiente original
                    pass
        
        # Fallback para o ambiente original
        return getattr(self.gym_env, method_name)(*args, **kwargs)
    
    # ========================================================
    #     Funções abstratas para implementação do otimizador
    # ========================================================
    
    @abstractmethod
    def select_driver(self, obs: dict, drivers: List[Driver], route: Route):
        pass

    @abstractmethod
    def get_title(self):
        pass

    def get_hyperparameters(self) -> dict:
        """Hiperparâmetros recebidos na construção, em formato serializável."""
        return {}
    
    # =======================================================================
    #     Funções para execução do otimizador e coleta de estatísticas
    # =======================================================================

    def reset_env(self, seed: int | None = None):
        if self.is_vectorized:
            if seed is not None and hasattr(self.wrapped_env, 'seed'):
                self.wrapped_env.seed(seed)

            # wrapped_env.reset() percorre toda a pilha de wrappers (incluindo
            # VecNormalize). clear_simpy_env descarta o SimPy env
            # do episódio anterior sem disparar um segundo reset.
            self.state = self.wrapped_env.reset()
            self._call_env_method("clear_simpy_env")
        else:
            self.state, _ = self.wrapped_env.reset(seed=seed)
            self.gym_env.clear_simpy_env()

        self.done = False
        self.truncated = False

    def prepare_episode(self, seed: int | None = None):
        """
        Prepara o otimizador para um novo episódio com semente explícita.

        Subclasses (ex.: rollout) podem resetar RNGs próprios aqui.
        """
        self.reset_env(seed=seed)
        self._call_env_method("set_mode", EnvMode.EVALUATING)

    def assign_driver_to_order(self, obs: dict, order: Order):
        segment_pickup = PickupRouteSegment(order)
        segment_delivery = DeliveryRouteSegment(order)
        route = Route(self.gym_env.get_simpy_env(), [segment_pickup, segment_delivery])
        drivers = self.gym_env.get_drivers()
        return self.select_driver(obs, drivers, route)
    
    def run(self):
        sum_reward = 0
        step_count = 0
        
        while not (self.done or self.truncated):
            try:
                order = self.gym_env.get_current_order()
                action = self.assign_driver_to_order(self.state, order)
                
                if self.is_vectorized:
                    # Para ambientes vectorizados
                    action_array = np.array([action]) if not isinstance(action, np.ndarray) else action
                    obs, reward, done, info = self.wrapped_env.step(action_array)
                    
                    # Extrai valores para ambientes vectorizados
                    self.state = obs
                    reward = reward[0] if isinstance(reward, np.ndarray) else reward
                    self.done = done[0] if isinstance(done, np.ndarray) else done
                    self.truncated = info[0].get('TimeLimit.truncated', False) if isinstance(info, list) else info.get('TimeLimit.truncated', False)
                else:
                    # Para ambientes normais
                    self.state, reward, self.done, self.truncated, info = self.wrapped_env.step(action)
                
                sum_reward += reward
                step_count += 1
                
            except Exception as e:
                print(f"Erro durante execução: {e}")
                break

        return {
            "final_state": self.state,
            "final_reward": reward if 'reward' in locals() else 0,
            "done": self.done,
            "truncated": self.truncated,
            "sum_reward": sum_reward,
            "info": info if 'info' in locals() else {},
            "steps": step_count
        }
    
    # ========================================================
    #     Simulações em lote
    # ========================================================

    def run_simulations(
        self,
        num_runs: int,
        dir_path: str,
        seed: int | None = None,
        save_individual_plots: bool = True,
        save_mean_plots: bool = True,
        metrics_fmt: str = "npz",
        num_workers: int = 1,
        eval_spec=None,
    ):
        import time

        from food_delivery_gym.main.eval.eval_parallel import (
            EpisodeJob,
            derive_episode_seeds,
            run_episodes_parallel,
        )

        if num_workers < 1:
            raise ValueError(f"num_workers deve ser >= 1; recebido {num_workers}")
        if num_workers > 1 and eval_spec is None:
            raise ValueError(
                "num_workers > 1 exige eval_spec (EvalJobSpec) para reconstruir "
                "o otimizador nos processos filhos"
            )

        os.makedirs(dir_path, exist_ok=True)
        episode_seeds = derive_episode_seeds(seed, num_runs)
        wall_t0 = time.perf_counter()

        stats = SimulationStats()
        # Hyperparâmetros: no caminho paralelo o self do pai ainda existe e
        # reflete a mesma configuração que os workers vão reconstruir.
        if num_workers <= 1:
            self.prepare_episode(episode_seeds[0] if episode_seeds else seed)
        else:
            self._call_env_method("set_mode", EnvMode.EVALUATING)
        stats.hyperparameters = self.get_hyperparameters()

        print(f"=== {self.get_title()} | runs={num_runs} | seed={seed} ===")

        if num_workers <= 1:
            self._run_simulations_serial(
                num_runs=num_runs,
                episode_seeds=episode_seeds,
                stats=stats,
                dir_path=dir_path,
                save_individual_plots=save_individual_plots,
            )
        else:
            jobs = [
                EpisodeJob(spec=eval_spec, episode_idx=i, seed=episode_seeds[i])
                for i in range(num_runs)
            ]
            results = run_episodes_parallel(jobs, num_workers=num_workers)
            self._ingest_parallel_results(
                results=results,
                stats=stats,
                dir_path=dir_path,
                save_individual_plots=save_individual_plots,
            )

        stats.finalize()
        stats.duration_seconds = time.perf_counter() - wall_t0

        if save_mean_plots:
            try:
                board: Board = stats.get_batch_board()
                board.save(dir_path)
            except Exception as e:
                print(f"⚠  Erro ao salvar board de médias: {e}")

        stats.save(dir_path=dir_path, fmt=metrics_fmt)
        print(
            f"Resultados salvos em {dir_path} "
            f"(duração={stats.duration_seconds:.2f}s)"
        )
        return stats

    def _run_simulations_serial(
        self,
        num_runs: int,
        episode_seeds: list[int],
        stats: SimulationStats,
        dir_path: str,
        save_individual_plots: bool,
    ):
        import time

        for i in range(num_runs):
            print(f"-> Execução {i + 1} de {num_runs}...")
            t0 = time.perf_counter()
            self.prepare_episode(episode_seeds[i])

            sum_reward = 0.0
            ep_length = 0
            was_truncated = True
            run_ok = False

            try:
                resultado = self.run()
                sum_reward = resultado["sum_reward"]
                ep_length = resultado["steps"]
                was_truncated = resultado["truncated"]
                run_ok = True
            except Exception as e:
                print(f"  ✗ Erro na execução {i + 1}: {e}")
                traceback.print_exc()

            eval_seconds = time.perf_counter() - t0

            if run_ok:
                simpy_env = self.gym_env.get_simpy_env()
                orders_generated = self._call_env_method("get_num_orders_generated")
                stats.register_episode(
                    simpy_env=simpy_env,
                    reward=sum_reward,
                    length=ep_length,
                    truncated=was_truncated,
                    orders_generated=orders_generated,
                    seed=episode_seeds[i],
                    eval_seconds=eval_seconds,
                )
                episode_idx = len(stats._raw_episodes) - 1
                print(
                    f"  Retorno = {sum_reward:.4f} | Passos = {ep_length} | "
                    f"SimPy t = {simpy_env.now} | Truncada = {was_truncated} | "
                    f"eval = {eval_seconds:.2f}s"
                )
                if save_individual_plots:
                    try:
                        board: Board = stats.get_episode_board(episode_idx=episode_idx)
                        board.save(dir_path)
                    except Exception as e:
                        print(f"  ⚠  generate_episode_stats_board falhou: {e}")

    def _ingest_parallel_results(
        self,
        results: list[dict],
        stats: SimulationStats,
        dir_path: str,
        save_individual_plots: bool,
    ):
        # Progresso já foi impresso em tempo real por run_episodes_parallel.
        for result in results:
            idx = result["episode_idx"]
            if not result["ok"]:
                print(f"  ✗ Detalhe do erro na execução {idx + 1}:\n{result['error']}")
                continue

            episode = result["episode"]
            stats.register_episode_dict(episode)
            episode_idx = len(stats._raw_episodes) - 1
            eval_s = episode.get("eval_seconds")
            eval_str = f" | eval = {eval_s:.2f}s" if eval_s is not None else ""
            print(
                f"  Execução {idx + 1}: Retorno = {episode['reward']:.4f} | "
                f"Passos = {episode['length']} | SimPy t = {episode['simpy_time']} | "
                f"Truncada = {episode['truncated']}{eval_str}"
            )
            if save_individual_plots:
                try:
                    board: Board = stats.get_episode_board(episode_idx=episode_idx)
                    board.save(dir_path)
                except Exception as e:
                    print(f"  ⚠  generate_episode_stats_board falhou: {e}")
    
    # ========================================================
    #     Escrita do cabeçalho do relatório
    # ========================================================

    def _write_run_header(self, results_file, num_runs: int, seed: int | None) -> None:
        results_file.write("-------------------> " + self.get_title() + " <-------------------\n\n")
        results_file.write("---> Configurações Gerais:\n")
        results_file.write(f"Número de execuções: {num_runs}\n")
        results_file.write(f"Seed de números aleatórios: {seed}\n")
        results_file.write(f"Ambiente vectorizado: {self.is_vectorized}\n")
        results_file.write(f"Tipo do ambiente wrapper: {type(self.wrapped_env).__name__}\n")
        results_file.write("\n---> Configurações do Cenário do Ambiente: ")
        try:
            results_file.write(self._call_env_method('get_description'))
        except Exception as e:
            results_file.write(f"Erro ao obter descrição: {e}")

        hyperparameters = self.get_hyperparameters()
        if hyperparameters:
            results_file.write("\n\n---> Hiperparâmetros do Otimizador:\n")
            _write_hyperparameters(results_file, hyperparameters)

        results_file.write("\n\n---> Registro de execuções:\n")

    # ========================================================
    #     Helper de controle do ambiente
    # ========================================================

    def set_gym_env_mode(self, mode: EnvMode):
        self._call_env_method('set_mode', mode)
    
    # =====================================================================
    #     Execução interativa / automática
    # =====================================================================

    """
        Interpreta a entrada do usuário e retorna uma ação válida.
        
        Args:
            text: Texto digitado pelo usuário
            
        Returns:
            Ação válida para o action_space
    """
    def _parse_action_input(self, text: str):
        text = text.strip()
        action_space = self.gym_env.action_space
        
        if text.lower() in ("", "rand", "random"):
            return action_space.sample()

        # Discrete
        if hasattr(action_space, 'n'):  # gym.spaces.Discrete
            try:
                action = int(text)
            except ValueError:
                raise ValueError(f"Ação inválida para espaço discreto '{text}'")
            if not action_space.contains(action):
                raise ValueError(f"Ação {action} fora do espaço válido [0, {action_space.n - 1}]")
            return action

        # Fallback: tentar converter para número
        try:
            return int(text)
        except Exception:
            try:
                return float(text)
            except Exception as e:
                raise ValueError("Formato de ação desconhecido para o action_space") from e
            
    def _step_environment(self, action):
        if self.is_vectorized:
            # Para ambientes vectorizados
            action_array = np.array([action]) if not isinstance(action, np.ndarray) else action
            obs, reward, done, info = self.wrapped_env.step(action_array)
            
            # Extrai valores
            obs = obs
            reward = reward[0] if isinstance(reward, np.ndarray) else reward
            terminated = done[0] if isinstance(done, np.ndarray) else done
            truncated = info[0].get('TimeLimit.truncated', False) if isinstance(info, list) else info.get('TimeLimit.truncated', False)
            info = info[0] if isinstance(info, list) else info
        else:
            # Para ambientes normais (gymnasium format)
            step_res = self.wrapped_env.step(action)
            
            if len(step_res) == 5:
                obs, reward, terminated, truncated, info = step_res
            elif len(step_res) == 4:
                obs, reward, done, info = step_res
                terminated = done
                truncated = False
            else:
                raise RuntimeError("Formato de retorno de env.step() inesperado")
        
        return obs, reward, terminated, truncated, info
    
    def run_auto(self, max_steps: int = 10000) -> Board:
        """
        Executa o ambiente automaticamente com feedback visual.
        
        Args:
            max_steps: Número máximo de passos
        """
        step = 0
        sum_reward = 0.0
        
        print("=== Modo Automático ===")
        print("Executando até o fim...\n")
        
        while step < max_steps and not (self.done or self.truncated):
            step += 1
            
            try:
                order = self.gym_env.get_current_order()
                action = self.assign_driver_to_order(self.state, order)
                
                obs, reward, terminated, truncated, info = self._step_environment(action)
                
                self.state = obs
                self.done = terminated
                self.truncated = truncated
                sum_reward += reward
                
                # Feedback a cada 10 passos
                if step % 10 == 0:
                    print(f"Step {step}: Recompensa acumulada = {sum_reward:.2f}")
                
            except Exception as e:
                print(f"Erro no passo {step}: {e}")
                break
        
        print(f"\nExecução finalizada em {step} passos")
        print(f"Recompensa total: {sum_reward:.2f}")

        return self._generate_episode_board(sum_reward=sum_reward, step=step)

    def run_interactive(self, max_steps: int = 10000) -> Board:
        """
        Executa o ambiente em modo interativo.
        O usuário controla quando executar cada passo.
        
        Args:
            max_steps: Número máximo de passos
        """
        step = 0
        sum_reward = 0.0
        
        action_space = self.gym_env.action_space
        action_space_n = action_space.n if hasattr(action_space, 'n') else "N/A"
        
        print("=== Modo Interativo ===")
        print(f"Action space: {action_space}")
        print("Controles:")
        print("  - Enter: ação automática (usa o otimizador)")
        print("  - <número>: força uma ação específica")
        print("  - 'run': executa automaticamente até o fim")
        print("  - 'run <n>': executa N passos automaticamente")
        print("  - 'quit': encerra\n")
        
        mode = "interactive"
        steps_to_run = 0
        self.state = self.gym_env.get_observation()
        
        while step < max_steps and not (self.done or self.truncated):
            step += 1
            print(f"\n{'─'*76}")
            print(f" STEP {step}")
            
            # Reduz contador se estiver em modo limitado
            if mode == "auto_limited":
                if steps_to_run > 0:
                    steps_to_run -= 1
                if steps_to_run == 0:
                    print("\nExecução automática limitada finalizada. Voltando ao modo interativo.")
                    mode = "interactive"
            
            # Determina a ação
            if mode in ("auto", "auto_limited"):
                # Modo automático: usa o otimizador
                try:
                    order = self.gym_env.get_current_order()
                    action = self.assign_driver_to_order(self.state, order)
                    print(f"Ação automática ({self.get_title()}): {action}")
                except Exception as e:
                    print(f"Erro ao obter ação do otimizador: {e}")
                    action = action_space.sample()
                    print(f"Usando ação aleatória: {action}")
            else:
                # Modo interativo: pede input do usuário
                invalid_action = True
                action = None
                
                while invalid_action:
                    prompt = (
                        f"\n---> Enter para ação automática;"
                        f" Número [0-{action_space_n-1}] para ação manual;"
                        f" 'run' para executar até o fim;"
                        f" 'run <n>' para N passos;"
                        f" 'quit' para sair\n> "
                    )
                    user_in = input(prompt).strip()
                    
                    # Executar até o fim
                    if user_in.lower() == "run":
                        mode = "auto"
                        try:
                            order = self.gym_env.get_current_order()
                            action = self.assign_driver_to_order(self.state, order)
                        except Exception:
                            action = action_space.sample()
                        invalid_action = False
                    
                    # Executar N passos
                    elif user_in.lower().startswith("run "):
                        try:
                            steps_to_run = int(user_in.split()[1])
                            mode = "auto_limited"
                            order = self.gym_env.get_current_order()
                            action = self.assign_driver_to_order(self.state, order)
                            invalid_action = False
                            print(f"Executando automaticamente por {steps_to_run} steps...")
                        except (IndexError, ValueError):
                            print("Uso inválido: digite 'run <n>' com um número inteiro positivo.")
                        except Exception as e:
                            print(f"Erro: {e}")
                    
                    # Encerrar
                    elif user_in.lower() in ("q", "quit", "exit"):
                        print("Saindo por solicitação do usuário.")
                        return
                    
                    # Ação automática (Enter ou vazio)
                    elif user_in == "":
                        try:
                            order = self.gym_env.get_current_order()
                            action = self.assign_driver_to_order(self.state, order)
                            print(f"Ação do otimizador: {action}")
                        except Exception as e:
                            print(f"Erro ao obter ação do otimizador: {e}")
                            action = action_space.sample()
                            print(f"Usando ação aleatória: {action}")
                        invalid_action = False
                    
                    # Ação manual
                    else:
                        try:
                            action = self._parse_action_input(user_in)
                            print(f"Ação manual: {action}")
                            invalid_action = False
                        except Exception as e:
                            print(f"Erro ao interpretar ação: {e}")
            
            # Executa o passo
            try:
                obs, reward, terminated, truncated, info = self._step_environment(action)
                
                self.state = obs
                self.done = terminated
                self.truncated = truncated
                sum_reward += reward
                
                # Mostra feedback
                self.gym_env.print_environment_state()
                print(format_step_result(action, reward, sum_reward, info))
                
            except Exception as e:
                print(f"Erro ao executar passo: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print(f"\n{'='*60}")
        print("=== Execução Finalizada ===")
        print(f"Total de passos: {step}")
        print(f"Recompensa total: {sum_reward:.2f}")
        print(f"Terminado: {self.done}, Truncado: {self.truncated}")

        return self._generate_episode_board(sum_reward=sum_reward, length=step)
    
    # evite a replicação de código entre run_auto e run_interactive, mas mantenha a estrutura clara para cada modo
    def _generate_episode_board(self, sum_reward: float, length: int):
        simpy_env        = self.gym_env.get_simpy_env()
        orders_generated = self._call_env_method("get_num_orders_generated")

        stats = SimulationStats()

        stats.register_episode(
            simpy_env=simpy_env,
            reward=sum_reward,
            length=length,
            truncated=self.truncated,
            orders_generated=orders_generated,
        )

        board: Board = None
        try:
            board = stats.get_episode_board(episode_idx=0)
        except Exception as e:
            print(f"  ⚠ Geração de gráfico falhou: {e}")
        
        return board