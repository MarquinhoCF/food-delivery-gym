from abc import ABC, abstractmethod
from collections import defaultdict
import os
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

    def run_simulations(self, num_runs: int, dir_path: str, seed: int | None = None):
        self.initialize(seed=seed)
        self.set_gym_env_mode(EnvMode.EVALUATING)

        # Garantir que o diretório existe
        os.makedirs(dir_path, exist_ok=True)
        file_path = dir_path + "results.txt"

        total_rewards = []
        truncated_runs = []
        with open(file_path, "w", encoding="utf-8") as results_file:
            self.get_description(results_file, num_runs, seed)
            results_file.write("---> Registro de execuções:\n")
            
            for i in range(num_runs):
                print(f"-> Execução {i + 1} de {num_runs}...")

                try:
                    resultado = self.run()

                    sum_reward = resultado["sum_reward"]
                    was_truncated = resultado["truncated"]
                    
                    # Tenta mostrar o board de estatísticas
                    try:
                        self._call_env_method('show_statistics_board', sum_reward=sum_reward, dir_path=dir_path)
                    except Exception as e:
                        print(f"Aviso: Não foi possível mostrar o board de estatísticas: {e}")
                    
                    total_rewards.append(sum_reward)
                    truncated_runs.append(was_truncated)
                    results_file.write(f"Execução {i + 1}: Soma das Recompensas = {sum_reward} (Passos: {resultado.get('steps', 'N/A')})\n")

                except Exception as e:
                    print(f"Erro na execução {i + 1}: {e}")
                    results_file.write(f"Execução {i + 1}: ERRO - {str(e)}\n")
                    total_rewards.append(0)  # Adiciona 0 para manter o count

                self.reset_env()
            
            # Calcula estatísticas apenas se houver recompensas válidas
            if total_rewards:
                total_rewards_statistics = {}
                valid_rewards = [r for r in total_rewards if r != 0]  # Remove erros
                
                if valid_rewards:
                    total_rewards_statistics['avg'] = sum(valid_rewards) / len(valid_rewards)
                    total_rewards_statistics['std_dev'] = stt.stdev(valid_rewards) if len(valid_rewards) > 1 else 0
                    total_rewards_statistics['median'] = stt.median(valid_rewards)
                    total_rewards_statistics['mode'] = stt.mode(valid_rewards) if len(set(valid_rewards)) > 1 else "Nenhuma moda única"
                else:
                    total_rewards_statistics = {'avg': 0, 'std_dev': 0, 'median': 0, 'mode': 'N/A'}

                results_file.write("\n---> Estatísticas das Recompensas:\n")
                results_file.write(f"* Média: {total_rewards_statistics['avg']:.2f}\n")
                results_file.write(f"* Desvio Padrão: {total_rewards_statistics['std_dev']:.2f}\n")
                results_file.write(f"* Mediana: {total_rewards_statistics['median']:.2f}\n")
                results_file.write(f"* Moda: {total_rewards_statistics['mode']}\n")

                try:
                    establishment_metrics, driver_metrics = self._call_env_method('get_statistics_data')

                    time_spent_on_delivery_list = []
                    total_distance_traveled_list = []

                    for i in range(num_runs):
                        sum_time_spent_on_delivery_i = 0
                        for driver_id, metrics in driver_metrics.items():
                            sum_time_spent_on_delivery_i += metrics["time_spent_on_delivery"][i]
                        time_spent_on_delivery_list.append(sum_time_spent_on_delivery_i)

                        if not truncated_runs[i]:
                            sum_total_distance_traveled_i = 0
                            for driver_id, metrics in driver_metrics.items():
                                sum_total_distance_traveled_i += metrics["total_distance"][i]
                            total_distance_traveled_list.append(sum_total_distance_traveled_i)

                    time_spent_on_delivery_statistics = {}
                    time_spent_on_delivery_statistics["avg"] = stt.mean(time_spent_on_delivery_list)
                    time_spent_on_delivery_statistics["std_dev"] = stt.stdev(time_spent_on_delivery_list)
                    time_spent_on_delivery_statistics["median"] = stt.median(time_spent_on_delivery_list)
                    time_spent_on_delivery_statistics["mode"] = stt.mode(time_spent_on_delivery_list)

                    results_file.write("\n---> Estatísticas do Tempo Gasto com entregas:\n")
                    results_file.write(f"* Média: {time_spent_on_delivery_statistics['avg']:.2f}\n")
                    results_file.write(f"* Desvio Padrão: {time_spent_on_delivery_statistics['std_dev']:.2f}\n")
                    results_file.write(f"* Mediana: {time_spent_on_delivery_statistics['median']:.2f}\n")
                    results_file.write(f"* Moda: {time_spent_on_delivery_statistics['mode']}\n")

                    # Verifica se há dados confiáveis de distância
                    num_truncated = sum(truncated_runs)
                    num_valid = num_runs - num_truncated
                    
                    results_file.write(f"\n---> Execuções válidas para métricas de distância: {num_valid}/{num_runs}\n")
                    if num_truncated > 0:
                        results_file.write(f"* {num_truncated} execução(ões) truncada(s) foram excluídas das estatísticas de distância\n")

                    # Calcula estatísticas apenas se houver dados válidos
                    if total_distance_traveled_list and len(total_distance_traveled_list) > 0:
                        total_distance_traveled_statistics = {}
                        total_distance_traveled_statistics["avg"] = stt.mean(total_distance_traveled_list)
                        total_distance_traveled_statistics["std_dev"] = stt.stdev(total_distance_traveled_list)
                        total_distance_traveled_statistics["median"] = stt.median(total_distance_traveled_list)
                        total_distance_traveled_statistics["mode"] = stt.mode(total_distance_traveled_list)
                        
                        results_file.write("\n---> Estatísticas da Distância Percorrida:\n")
                        results_file.write(f"* Média: {total_distance_traveled_statistics['avg']:.2f}\n")
                        results_file.write(f"* Desvio Padrão: {total_distance_traveled_statistics['std_dev']:.2f}\n")
                        results_file.write(f"* Mediana: {total_distance_traveled_statistics['median']:.2f}\n")
                        results_file.write(f"* Moda: {total_distance_traveled_statistics['mode']}\n")
                    else:
                        total_distance_traveled_statistics = None
                        results_file.write("\n---> Estatísticas da Distância Percorrida:\n")
                        results_file.write("* NULO - Todas as execuções foram truncadas\n")

                    geral_statistics = self._call_env_method('get_statistics')
                    results_file.write(f"\n---> Estatísticas Finais:\n")
                    results_file.write(f"{self.format_statistics(geral_statistics)}")

                    self._call_env_method('show_total_mean_statistics_board', 
                                        sum_rewards_mean=total_rewards_statistics['avg'], 
                                        dir_path=dir_path)

                    self.save_metrics_to_file(total_rewards, total_rewards_statistics, 
                                            time_spent_on_delivery_list, time_spent_on_delivery_statistics, 
                                            total_distance_traveled_list, total_distance_traveled_statistics,
                                            establishment_metrics, driver_metrics, 
                                            geral_statistics, dir_path)
                except Exception as e:
                    print(f"Aviso: Erro ao processar estatísticas finais: {e}")
                    results_file.write(f"\n---> Erro ao processar estatísticas finais: {e}\n")
        
        print(f"Resultados salvos em " + file_path)

    def save_metrics_to_file(
        self,
        total_rewards: list[float], 
        total_rewards_statistics: dict[str, float],
        time_spent_on_delivery_list: list[float],
        time_spent_on_delivery_statistics: dict[str, float],
        total_distance_traveled_list: list[float],
        total_distance_traveled_statistics: dict[str, float],
        establishment_metrics: defaultdict[str, defaultdict[str, list[float]]], 
        driver_metrics: defaultdict[str, defaultdict[str, list[float]]], 
        geral_statistics: dict[str, dict[str, dict[str, float]]],
        dir_path: str = "./", 
        file_name: str = "metrics_data.npz"
    ) -> None:
        try:
            file_path = dir_path + file_name

            # Converte defaultdict para dict para evitar problemas de serialização
            establishment_metrics = {k: dict(v) for k, v in establishment_metrics.items()}
            driver_metrics = {k: dict(v) for k, v in driver_metrics.items()}

            np.savez_compressed(file_path, 
                                total_rewards=np.array(total_rewards), 
                                total_rewards_statistics=total_rewards_statistics,
                                time_spent_on_delivery_list=np.array(time_spent_on_delivery_list),
                                time_spent_on_delivery_statistics=time_spent_on_delivery_statistics,
                                total_distance_traveled_list=np.array(total_distance_traveled_list),
                                total_distance_traveled_statistics=total_distance_traveled_statistics,
                                establishment_metrics=establishment_metrics, 
                                driver_metrics=driver_metrics,
                                geral_statistics=geral_statistics)
            
            print(f"Métricas salvas em {file_path}")
            
        except Exception as e:
            print(f"Erro ao salvar métricas: {e}")

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
                    print(f"Ação automática (otimizador): {action}")
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