from abc import ABC, abstractmethod
from collections import defaultdict
import os
import traceback
from typing import List, Union

import numpy as np
import statistics as stt
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.environment.env_mode import EnvMode
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer.optimizer import Optimizer
from food_delivery_gym.main.order.order import Order
from food_delivery_gym.main.route.delivery_route_segment import DeliveryRouteSegment
from food_delivery_gym.main.route.pickup_route_segment import PickupRouteSegment
from food_delivery_gym.main.route.route import Route
from food_delivery_gym.main.statistic.summarized_data_board import SummarizedDataBoard


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
    
    # =======================================================================
    #     Funções para execução do otimizador e coleta de estatísticas
    # =======================================================================

    def initialize(self, seed: int | None = None):
        self.reset_env(seed=seed)
        self._call_env_method('reset_statistics')
        SummarizedDataBoard.reset_image_counter()

    def reset_env(self, seed: int | None = None):
        if self.is_vectorized:
            self.state = self.wrapped_env.reset()
            if hasattr(self.wrapped_env, 'seed'):
                self.wrapped_env.seed(seed)
        else:
            self.state, info = self.wrapped_env.reset(seed=seed)
        
        self.done = False
        self.truncated = False

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
    
    def show_statistics_board(self):
        self._call_env_method('show_statistics_board')
    
    def show_mean_statistic_board(self):
        self._call_env_method('show_total_mean_statistics_board')

    def set_gym_env_mode(self, mode: EnvMode):
        self._call_env_method('set_mode', mode)

    def get_description(self, results_file, num_runs: int, seed: int | None = None):
        results_file.write("-------------------> " + self.get_title() + " <-------------------\n\n")

        results_file.write("---> Configurações Gerais:\n")
        results_file.write(f"Número de execuções: {num_runs}\n")
        results_file.write(f"Seed de números aleatórios: {seed}\n")
        results_file.write(f"Ambiente vectorizado: {self.is_vectorized}\n")
        results_file.write(f"Tipo do ambiente wrapper: {type(self.wrapped_env).__name__}\n")
        results_file.write("\n---> Configurações do Cenário do Ambiente: ")
        
        try:
            description = self._call_env_method('get_description')
            results_file.write(description)
        except Exception as e:
            results_file.write(f"Erro ao obter descrição: {e}")
        
        results_file.write("\n\n")
    
    def format_statistics(self, statistics: dict):
        result = ""
        
        # Formata os dados para estabelecimentos
        if 'establishments' in statistics:
            result += "Establishments:\n"
            for establishment_id, stats in statistics['establishments'].items():
                result += f"  Establishment {establishment_id}:\n"
                for key, value in stats.items():
                    result += f"    {key.replace('_', ' ').title()}:\n"
                    for stat, stat_value in value.items():
                        result += f"      {stat.title()}: {stat_value}\n"
                result += "\n"
        
        # Formata os dados para motoristas
        if 'drivers' in statistics:
            result += "Drivers:\n"
            for driver_id, stats in statistics['drivers'].items():
                result += f"  Driver {driver_id}:\n"
                for key, value in stats.items():
                    result += f"    {key.replace('_', ' ').title()}:\n"
                    for stat, stat_value in value.items():
                        result += f"      {stat.title()}: {stat_value}\n"
                result += "\n"
        
        return result

    def run_simulations(
            self, 
            num_runs: int, 
            dir_path: str, 
            seed: int | None = None,
            save_individual_plots: bool = True,
            save_mean_plots: bool = True,
        ):
        self.initialize(seed=seed)
        self.set_gym_env_mode(EnvMode.EVALUATING)

        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, "results.txt")

        self._current_dir_path = dir_path

        # ── Listas de coleta por episódio ─────────────────────────────────────
        total_rewards:         list = []
        episode_lengths:       list = []
        simpy_last_time_steps: list = []
        num_orders_generated:  list = []
        truncated_runs:        list = []

        with open(file_path, "w", encoding="utf-8") as results_file:
            self.get_description(results_file, num_runs, seed)
            results_file.write("---> Registro de execuções:\n")

            for i in range(num_runs):
                print(f"-> Execução {i + 1} de {num_runs}...")

                # ── Executa o episódio ────────────────────────────────────
                run_ok     = False
                sum_reward = 0
                ep_length  = 0
                simpy_t    = None
                was_truncated = True   # pessimista: se falhar, trata como truncada

                try:
                    resultado     = self.run()
                    sum_reward    = resultado["sum_reward"]
                    ep_length     = resultado["steps"]
                    was_truncated = resultado["truncated"]
                    info          = resultado.get("info", {})

                    if isinstance(info, list):
                        info = info[0] if info else {}

                    simpy_t = info.get("simpy_time_step", None)
                    run_ok  = True

                except Exception as e:
                    print(f"  ✗ Erro na execução {i + 1}: {e}")
                    traceback.print_exc()
                    results_file.write(f"Execução {i + 1}: ERRO - {e}\n")

                # ── Coleta métricas do episódio ───────────────────────────
                total_rewards.append(sum_reward)
                episode_lengths.append(ep_length)
                simpy_last_time_steps.append(simpy_t)
                truncated_runs.append(was_truncated)

                # Número de pedidos gerados — só faz sentido se o episódio rodou
                try:
                    orders = self._call_env_method('get_num_orders_generated') if run_ok else 0
                except Exception:
                    orders = 0
                num_orders_generated.append(orders)

                # ── Board individual de estatísticas ──────────────────────
                if run_ok:
                    results_file.write(
                        f"Execução {i + 1}: Retorno = {sum_reward:.4f} | "
                        f"Passos = {ep_length} | "
                        f"SimPy t = {simpy_t} | "
                        f"Truncada = {was_truncated}\n"
                    )
                    if save_individual_plots:
                        try:
                            self._call_env_method('show_statistics_board',
                                                sum_reward=sum_reward, dir_path=dir_path)
                        except Exception as e:
                            print(f"  ⚠  show_statistics_board falhou: {e}")

                self.reset_env()

            # ── Calcula estatísticas e escreve no arquivo ─────────────────
            results_file.write("\n" + "=" * 60 + "\n")
            results_file.write("RESUMO ESTATÍSTICO\n")
            results_file.write("=" * 60 + "\n")

            stats_data = self._compute_simulation_stats(
                num_runs, total_rewards, episode_lengths,
                simpy_last_time_steps, num_orders_generated,
                truncated_runs, results_file,
                save_mean_plots=save_mean_plots,
            )

        # ── Persiste o .npz  ─
        self.save_metrics_to_file(stats_data, dir_path=dir_path)
        print(f"Resultados salvos em {file_path}")

    # ═════════════════════════════════════════════════════════════════════════════
    #  Helpers privados de estatística
    # ═════════════════════════════════════════════════════════════════════════════

    def _safe_stats(self, values: list) -> dict:
        """
        Calcula avg / std_dev / median / mode de forma segura.
        Aceita listas com None — os Nones são ignorados.
        Retorna None se não houver valores válidos.
        """
        clean = [float(v) for v in values if v is not None]
        if not clean:
            return None

        result = {
            "avg":     sum(clean) / len(clean),
            "std_dev": stt.stdev(clean) if len(clean) > 1 else 0.0,
            "median":  stt.median(clean),
            "mode":    None,
            "n":       len(clean),
        }
        try:
            result["mode"] = stt.mode(clean)
        except stt.StatisticsError:
            result["mode"] = "Sem moda única"
        return result


    def _write_stats_block(self, results_file, title: str, stats: dict | None):
        """Escreve um bloco de estatísticas no arquivo de resultados."""
        results_file.write(f"\n---> {title}:\n")
        if stats is None:
            results_file.write("* NULO - Sem dados válidos\n")
            return
        results_file.write(f"* N amostras:    {stats['n']}\n")
        results_file.write(f"* Média:         {stats['avg']:.4f}\n")
        results_file.write(f"* Desvio Padrão: {stats['std_dev']:.4f}\n")
        results_file.write(f"* Mediana:       {stats['median']:.4f}\n")
        results_file.write(f"* Moda:          {stats['mode']}\n")


    # ═════════════════════════════════════════════════════════════════════════════
    #  _compute_simulation_stats — separado para facilitar testes e reuso
    # ═════════════════════════════════════════════════════════════════════════════

    def _compute_simulation_stats(
        self,
        num_runs: int,
        total_rewards: list,
        episode_lengths: list,
        simpy_last_time_steps: list,
        num_orders_generated: list,
        truncated_runs: list,
        results_file,
        save_mean_plots: bool = True,
    ) -> dict:
        """
        Calcula todas as estatísticas das simulações e escreve no results_file.

        Retorna um dicionário com todos os dados calculados prontos para o .npz.
        Nunca lança exceção — erros são capturados, logados e representados como None.
        """
        out = {
            # listas brutas
            "total_rewards":              total_rewards,
            "episode_lengths":            episode_lengths,
            "simpy_last_time_steps":      simpy_last_time_steps,
            "num_orders_generated":       num_orders_generated,
            "truncated_runs":             truncated_runs,
            # estatísticas (preenchidas abaixo)
            "total_rewards_statistics":             None,
            "episode_lengths_statistics":           None,
            "simpy_last_time_steps_statistics":     None,
            "time_spent_on_delivery_list":          None,
            "time_spent_on_delivery_statistics":    None,
            "total_distance_traveled_list":         None,
            "total_distance_traveled_statistics":   None,
            "total_orders_generated_statistics":    None,
            "establishment_metrics":                None,
            "driver_metrics":                       None,
            "geral_statistics":                     None,
        }

        # ── 1. Estatísticas das recompensas ───────────────────────────────────
        # Usa todas as recompensas, inclusive 0 — runs com erro já foram marcadas
        # como truncated e serão separadas nas métricas de distância.
        reward_stats = self._safe_stats(total_rewards)
        out["total_rewards_statistics"] = reward_stats
        self._write_stats_block(results_file, "Estatísticas das Recompensas", reward_stats)

        # ── 2. Estatísticas do tamanho dos episódios ──────────────────────────
        length_stats = self._safe_stats(episode_lengths)
        out["episode_lengths_statistics"] = length_stats
        self._write_stats_block(results_file, "Estatísticas do Tamanho dos Episódios (passos)", length_stats)

        # ── 3. Estatísticas do simpy_time_step final ──────────────────────────
        valid_simpy = [v for v in simpy_last_time_steps if v is not None]
        simpy_stats = self._safe_stats(valid_simpy) if valid_simpy else None
        out["simpy_last_time_steps_statistics"] = simpy_stats
        self._write_stats_block(results_file, "Estatísticas do Tempo de Simulação SimPy (último passo)", simpy_stats)

        # ── 4. Métricas de drivers/estabelecimentos ───────────────────────────
        try:
            establishment_metrics, driver_metrics = self._call_env_method('get_statistics_data')
            out["establishment_metrics"] = establishment_metrics
            out["driver_metrics"]        = driver_metrics
        except Exception as e:
            results_file.write(f"\n⚠  Não foi possível obter get_statistics_data: {e}\n")
            traceback.print_exc()
            # sem dados de driver → não calcula tempo/distância
            self._write_stats_block(results_file, "Estatísticas do Tempo Gasto com Entregas", None)
            self._write_stats_block(results_file, "Estatísticas da Distância Percorrida", None)
            self._write_stats_block(results_file, "Estatísticas do Número de Pedidos Gerados",
                            self._safe_stats(num_orders_generated))
            return out

        # ── 5. Tempo gasto em entregas ────────────────────────────────────────
        try:
            time_spent_list = []
            for i in range(num_runs):
                total_i = sum(
                    driver_metrics[drv]["time_spent_on_delivery"][i]
                    for drv in driver_metrics
                    if i < len(driver_metrics[drv].get("time_spent_on_delivery", []))
                )
                time_spent_list.append(total_i)

            out["time_spent_on_delivery_list"]       = time_spent_list
            out["time_spent_on_delivery_statistics"] = self._safe_stats(time_spent_list)
            self._write_stats_block(results_file,
                            "Estatísticas do Tempo Gasto com Entregas",
                            out["time_spent_on_delivery_statistics"])
        except Exception as e:
            results_file.write(f"\n⚠  Erro ao calcular tempo de entrega: {e}\n")
            traceback.print_exc()

        # ── 6. Distância total percorrida (exclui runs truncadas) ─────────────
        try:
            num_truncated = sum(truncated_runs)
            num_valid     = num_runs - num_truncated
            results_file.write(f"\n---> Execuções válidas para métricas de distância: {num_valid}/{num_runs}\n")
            if num_truncated > 0:
                results_file.write(f"* {num_truncated} execução(ões) truncada(s) excluída(s) das estatísticas de distância\n")

            distance_list = []
            for i in range(num_runs):
                if truncated_runs[i]:
                    continue
                total_i = sum(
                    driver_metrics[drv]["total_distance"][i]
                    for drv in driver_metrics
                    if i < len(driver_metrics[drv].get("total_distance", []))
                )
                distance_list.append(total_i)

            out["total_distance_traveled_list"]       = distance_list if distance_list else None
            out["total_distance_traveled_statistics"] = self._safe_stats(distance_list) if distance_list else None
            self._write_stats_block(results_file,
                            "Estatísticas da Distância Percorrida",
                            out["total_distance_traveled_statistics"])
        except Exception as e:
            results_file.write(f"\n⚠  Erro ao calcular distância percorrida: {e}\n")
            traceback.print_exc()

        # ── 7. Pedidos gerados ────────────────────────────────────────────────
        try:
            orders_stats = self._safe_stats(num_orders_generated)
            out["total_orders_generated_statistics"] = orders_stats
            self._write_stats_block(results_file, "Estatísticas do Número de Pedidos Gerados", orders_stats)
        except Exception as e:
            results_file.write(f"\n⚠  Erro ao calcular pedidos gerados: {e}\n")

        # ── 8. Estatísticas gerais do ambiente ────────────────────────────────
        try:
            geral_statistics = self._call_env_method('get_statistics')
            out["geral_statistics"] = geral_statistics
            results_file.write(f"\n---> Estatísticas Finais:\n")
            results_file.write(self.format_statistics(geral_statistics))
        except Exception as e:
            results_file.write(f"\n⚠  Erro ao obter estatísticas gerais: {e}\n")
            traceback.print_exc()

        # ── 9. Board de médias ────────────────────────────────────────────────
        if save_mean_plots:
            try:
                avg_reward = reward_stats["avg"] if reward_stats else 0.0
                self._call_env_method(
                    'show_total_mean_statistics_board',
                    sum_rewards_mean=avg_reward,
                    dir_path=self._current_dir_path,   # definido em run_simulations
                )
            except Exception as e:
                results_file.write(f"\n⚠  Erro ao mostrar board de médias: {e}\n")

        return out


    # ═════════════════════════════════════════════════════════════════════════════
    #  save_metrics_to_file — robusto contra None em qualquer campo
    # ═════════════════════════════════════════════════════════════════════════════

    def save_metrics_to_file(
        self,
        stats_data: dict,
        dir_path: str = "./",
        file_name: str = "metrics_data.npz",
    ) -> None:
        """
        Persiste todos os dados de stats_data em um arquivo .npz.

        Campos None são salvos como arrays vazios para não quebrar o npz.
        O arquivo é sempre gerado, mesmo que parte dos dados esteja ausente.
        """
        file_path = os.path.join(dir_path, file_name)

        def to_array(value):
            """Converte qualquer valor para array numpy serializável."""
            if value is None:
                return np.array([])
            if isinstance(value, np.ndarray):
                return value
            if isinstance(value, list):
                # Listas com None → float, substituindo None por nan
                try:
                    clean = [float('nan') if v is None else v for v in value]
                    return np.array(clean, dtype=float)
                except (TypeError, ValueError):
                    return np.array(value, dtype=object)
            # dicts e outros objetos → array de objeto
            return np.array(value, dtype=object)

        def safe_metrics(metrics):
            """Serializa defaultdict/dict de métricas de driver/establishment."""
            if metrics is None:
                return np.array({}, dtype=object)
            try:
                return np.array({k: dict(v) for k, v in metrics.items()}, dtype=object)
            except Exception:
                return np.array({}, dtype=object)

        try:
            np.savez_compressed(
                file_path,
                # ── listas brutas ──────────────────────────────────────────
                total_rewards             = to_array(stats_data.get("total_rewards")),
                episode_lengths           = to_array(stats_data.get("episode_lengths")),
                simpy_last_time_steps     = to_array(stats_data.get("simpy_last_time_steps")),
                truncated_runs            = to_array(stats_data.get("truncated_runs")),
                num_orders_generated      = to_array(stats_data.get("num_orders_generated")),
                time_spent_on_delivery_list     = to_array(stats_data.get("time_spent_on_delivery_list")),
                total_distance_traveled_list    = to_array(stats_data.get("total_distance_traveled_list")),
                # ── estatísticas (dicts salvos como array de objeto) ───────
                total_rewards_statistics            = np.array(stats_data.get("total_rewards_statistics"),  dtype=object),
                episode_lengths_statistics          = np.array(stats_data.get("episode_lengths_statistics"), dtype=object),
                simpy_last_time_steps_statistics    = np.array(stats_data.get("simpy_last_time_steps_statistics"), dtype=object),
                time_spent_on_delivery_statistics   = np.array(stats_data.get("time_spent_on_delivery_statistics"), dtype=object),
                total_distance_traveled_statistics  = np.array(stats_data.get("total_distance_traveled_statistics"), dtype=object),
                total_orders_generated_statistics   = np.array(stats_data.get("total_orders_generated_statistics"), dtype=object),
                # ── métricas de agentes ────────────────────────────────────
                establishment_metrics = safe_metrics(stats_data.get("establishment_metrics")),
                driver_metrics        = safe_metrics(stats_data.get("driver_metrics")),
                geral_statistics      = np.array(stats_data.get("geral_statistics"), dtype=object),
            )
            print(f"Métricas salvas em {file_path}")

        except Exception as e:
            print(f"Erro ao salvar métricas em {file_path}: {e}")
            traceback.print_exc()

            # Tentativa de emergência: salva pelo menos as listas brutas
            try:
                emergency_path = file_path.replace(".npz", "_emergency.npz")
                np.savez_compressed(
                    emergency_path,
                    total_rewards    = to_array(stats_data.get("total_rewards")),
                    episode_lengths  = to_array(stats_data.get("episode_lengths")),
                    truncated_runs   = to_array(stats_data.get("truncated_runs")),
                )
                print(f"⚠  Arquivo de emergência salvo em {emergency_path}")
            except Exception as e2:
                print(f"Falha total ao salvar métricas: {e2}")

    # =====================================================================
    #     Funções usadas para teste e execução interativa do otimizador
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
    
    def run_auto(self, max_steps: int = 10000):
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

    def run_interactive(self, max_steps: int = 10000):
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
            print(f"\n{'='*60}")
            print(f"--- Step {step} ---")
            print(f"Observação atual: {np.array(self.state)}")
            
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
                self.gym_env.print_enviroment_state()
                print(f"\nAção aplicada: {action}")
                print(f"Recompensa do passo: {reward}")
                print(f"Recompensa acumulada: {sum_reward:.2f}")
                print(f"Info: {info}")
                
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