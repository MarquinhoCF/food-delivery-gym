from importlib.resources import files
import argparse
import sys
import textwrap
from typing import Any, Tuple

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv

# --- Config padrão ---
DEFAULT_SEED = 101010

"""
    Interpreta a string do usuário de acordo com o tipo do action_space.

    Suporta:
    - Discrete: número inteiro
    - Aceita também a palavra especial `rand` para amostrar aleatoriamente.
"""
def parse_action_input(action_space: gym.Space, text: str):
    text = text.strip()
    if text.lower() in ("", "rand", "random"):
        return action_space.sample()

    try:
        # Discrete
        if isinstance(action_space, gym.spaces.Discrete):
            try:
                action = int(text)
            except ValueError:
                raise ValueError(f"ação inválida para espaço discreto '{text}'")
            if not action_space.contains(action):
                raise ValueError(f"ação {action} fora do espaço válido '{action_space}'")
            return action

    except Exception as e:
        raise ValueError(f"Não foi possível interpretar a ação, {e}")

    # fallback: tentar converter para número
    try:
        return int(text)
    except Exception:
        try:
            return float(text)
        except Exception as e:
            raise ValueError("Formato de ação desconhecido para o action_space") from e


def prepare_env(scenario_filename: str, seed: int, render: bool) -> Tuple[FoodDeliveryGymEnv, Any]:
    scenario_path = str(files("food_delivery_gym.main.scenarios").joinpath(scenario_filename))
    env: FoodDeliveryGymEnv = FoodDeliveryGymEnv(scenario_json_file_path=scenario_path)

    # Reset com compatibilidade (alguns ambientes retornam apenas obs, outros (obs, info))
    reset_options = {"render_mode": "human"} if render else None
    if reset_options:
        reset_res = env.reset(seed=seed, options=reset_options)
    else:
        reset_res = env.reset(seed=seed)

    if isinstance(reset_res, tuple):
        obs = reset_res[0]
    else:
        obs = reset_res

    return env, obs

# Normaliza o retorno de env.step para (obs, reward, terminated, truncated, info).
def step_result_to_tuple(step_res: Any):
    if len(step_res) == 5:
        obs, reward, terminated, truncated, info = step_res
    elif len(step_res) == 4:
        obs, reward, done, info = step_res
        # gym (antigo) retornou done em vez de terminated+truncated
        terminated = done
        truncated = False
    else:
        raise RuntimeError("Formato de retorno de env.step() inesperado")
    return obs, reward, terminated, truncated, info


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Runner para FoodDeliveryGymEnv.

            Modos:
             - auto: executa até o fim com ações aleatórias (ou com modelo, se --model-path for informado)
             - interactive: passo-a-passo; o usuário pode pressionar Enter para ação aleatória,
               digitar uma ação customizada ou digitar 'run' para executar automaticamente até o fim.
             - agent: usa um modelo PPO salvo para decidir ações (necessita de --model-path)
            """
        ),
    )

    parser.add_argument("--scenario", default="complex_obj1.json", help="Arquivo de cenário dentro de food_delivery_gym.main.scenarios")
    parser.add_argument("--mode", choices=("auto", "interactive", "agent"), default="interactive")
    parser.add_argument("--model-path", default=None, help="Caminho para um modelo PPO (apenas para --mode agent)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--render", action="store_true", help="Passar render_mode='human' no reset (se suportado pelo ambiente)")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--save-log", action="store_true", help="Redirecionar stdout/stderr para log.txt")

    args = parser.parse_args()

    if args.save_log:
        log_file = open("log.txt", "w", encoding="utf-8")
        sys.stdout = log_file
        sys.stderr = log_file
    else:
        log_file = None

    try:
        env, obs = prepare_env(args.scenario, seed=args.seed, render=args.render)

        # carregar agente, se solicitado
        model = None
        if args.mode == "agent":
            if PPO is None:
                raise RuntimeError("stable-baselines3 não está disponível. Instale para usar --mode agent.")
            if not args.model_path:
                raise RuntimeError("--mode agent requer --model-path com o arquivo do modelo PPO salvo.")
            model = PPO.load(args.model_path)
            print(f"Modelo PPO carregado de: {args.model_path}")

        step = 0
        soma_recompensa = 0.0

        action_space = env.action_space
        print("=== Ambiente pronto ===")
        print(f"Action space: {action_space}")
        print("Iniciando...\n")

        mode = args.mode

        done = False
        truncated = False

        while step < args.max_steps and not (done or truncated):
            step += 1
            print(f"--- Step {step} ---")
            print(f"Observação atual: {np.array(obs)}")

            if mode == "agent":
                # usar modelo para escolher ação
                action, _ = model.predict(obs, deterministic=True)
            elif mode == "auto":
                action = action_space.sample()
            else:  # interactive
                invalid_action = False
                while not invalid_action:
                    prompt = (
                        f"\n---> Pressione 'Enter' para ação aleatória; Digite uma ação manualmente (Número inteiro [min: 0, max: {action_space.n - 1}]); Digite 'run' para executar até o fim; Digite 'quit' para sair\n> "
                    )
                    user_in = input(prompt).strip()
                    if user_in.lower() == "run":
                        mode = "auto"
                        action = action_space.sample()
                    elif user_in.lower() in ("q", "quit", "exit"):
                        print("Saindo por solicitação do usuário.")
                        break
                    elif user_in == "":
                        action = action_space.sample()
                    else:
                        try:
                            action = parse_action_input(action_space, user_in)
                        except Exception as e:
                            invalid_action = True
                            print(f"Erro ao interpretar ação: {e}")

            try:
                step_res = env.step(action)
            except Exception as e:
                print(f"Erro ao executar env.step(action): {e}")
                break

            obs, reward, terminated, truncated, info = step_result_to_tuple(step_res)
            soma_recompensa += reward

            env.print_enviroment_state()

            print(f"Ação aplicada: {action}")
            print(f"Recompensa do passo: {reward}")
            print(f"Info: {info}\n")

            # continuar loop — a condição de parada é verified no topo do while
            done = bool(terminated)

        print("\n== FIM DA EXECUÇÃO ==")
        # imprimir estatísticas finais (se o env tiver métodos conhecidos)
        env.print_enviroment_state()
        try:
            print(f"Observação final: {env.get_observation()}")
        except Exception:
            pass
        try:
            print(f"Soma das recompensas = {soma_recompensa}")
        except Exception:
            pass
        try:
            print(f"quantidade de rotas criadas = {env.simpy_env.state.get_length_orders()}")
            print(f"quantidade de rotas entregues = {env.simpy_env.state.orders_delivered}")
        except Exception:
            pass

    except Exception as e:
        print(f"Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
