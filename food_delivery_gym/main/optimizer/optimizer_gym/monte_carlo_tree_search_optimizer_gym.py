from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Type

from food_delivery_gym.main.driver.driver import Driver
from food_delivery_gym.main.environment.food_delivery_gym_env import FoodDeliveryGymEnv
from food_delivery_gym.main.optimizer.optimizer_gym.nearest_driver_optimizer_gym import (
    NearestDriverOptimizerGym,
)
from food_delivery_gym.main.optimizer.optimizer_gym.optmizer_gym import (
    OptimizerGym,
    jsonable_hyperparameter,
)
from food_delivery_gym.main.optimizer.optimizer_gym.rollout_optimizer_gym import (
    RolloutOptimizerGym,
    TerminalCostMode,
)
from food_delivery_gym.main.route.route import Route


@dataclass
class _OutcomeChild:
    """Futuro reamostrado após uma ação."""

    scenario_seed: int
    env: FoodDeliveryGymEnv
    obs: dict
    done: bool
    truncated: bool
    node: "_TreeNode | None" = None


@dataclass
class _ActionStats:
    """Estatísticas de uma ação."""
    
    action: int
    immediate_reward: float = 0.0
    continuation_value: float = 0.0
    action_visits: int = 0
    outcomes: list[_OutcomeChild] = field(default_factory=list)
    _alpha: float = 1.0

    @property
    def action_value(self) -> float:
        return self.immediate_reward + self._alpha * self.continuation_value


@dataclass
class _TreeNode:
    """
    Nó pré-decisão.

    `env is None` = raiz lógica em St (ambiente real). Nunca se dá step nem
    rollout nesse caso sem antes clonar com future='resample'.
    """

    env: FoodDeliveryGymEnv | None
    obs: dict | None
    done: bool
    truncated: bool
    depth: int
    visit_count: int = 0
    actions: dict[int, _ActionStats] = field(default_factory=dict)
    untried_actions: list[int] | None = None
    pending: dict[int, tuple[_OutcomeChild, float]] = field(default_factory=dict) # cache de desfechos expandidos

    @property
    def is_live_root(self) -> bool:
        return self.env is None


class MonteCarloTreeSearchOptimizerGym(RolloutOptimizerGym):
    """
    Monte Carlo tree search com política de base do rollout nas folhas.

    Em cada decisão (Powell: raiz = St; amostra W após a ação):
      1. Raiz lógica no estado atual.
      2. Expansões só via clone(future='resample') + step (futuro hipotético).
      3. `iterations` trajetórias: seleção / expansão / simulação / atualização.
      4. Devolve a ação da raiz com maior action_value (sem bônus de exploração).
    """

    def __init__(
        self,
        environment: FoodDeliveryGymEnv,
        base_optimizer_cls: Type[OptimizerGym] = NearestDriverOptimizerGym,
        base_optimizer_kwargs: Optional[dict] = None,
        alpha: float = 1.0,
        horizon: Optional[int] = None,
        record_decisions: bool = True,
        scenario_seed: int = 0,
        terminal_cost_mode: TerminalCostMode = "0",
        iterations: int = 8,
        exploration_weight: float = 1.0,
        depth: Optional[int] = None,
        max_outcomes: int = 1,
        max_expanded_actions: Optional[int] = None,
    ):
        super().__init__(
            environment,
            base_optimizer_cls=base_optimizer_cls,
            base_optimizer_kwargs=base_optimizer_kwargs,
            alpha=alpha,
            horizon=horizon,
            record_decisions=record_decisions,
            scenario_seed=scenario_seed,
            terminal_cost_mode=terminal_cost_mode,
        )
        if iterations < 1:
            raise ValueError(f"iterations deve ser >= 1; recebido {iterations}")
        if exploration_weight < 0:
            raise ValueError(
                f"exploration_weight deve ser >= 0; recebido {exploration_weight}"
            )
        if max_outcomes < 1:
            raise ValueError(f"max_outcomes deve ser >= 1; recebido {max_outcomes}")
        if depth is not None and depth < 0:
            raise ValueError(f"depth deve ser >= 0 ou None; recebido {depth}")
        if max_expanded_actions is not None and max_expanded_actions < 1:
            raise ValueError(
                f"max_expanded_actions deve ser >= 1 ou None; "
                f"recebido {max_expanded_actions}"
            )

        self.iterations = int(iterations)
        self.exploration_weight = float(exploration_weight)
        self.max_outcomes = int(max_outcomes)
        self.depth = depth if depth is not None else self._default_depth()
        self.max_expanded_actions = (
            None if max_expanded_actions is None else int(max_expanded_actions)
        )

    def _default_depth(self) -> int:
        if self.horizon is None:
            return 5
        return int(self.horizon)

    def get_title(self):
        base_name = self.base_optimizer_cls.__name__
        horizon_str = f"H={self.horizon}" if self.horizon is not None else "H=inf"
        dthr_str = (
            "all" if self.max_expanded_actions is None else str(self.max_expanded_actions)
        )
        return (
            f"MCTS({base_name}, alpha={self.alpha}, {horizon_str}, "
            f"TC={self.terminal_cost_mode}, i={self.iterations}, "
            f"ew={self.exploration_weight:g}, d={self.depth}, dthr={dthr_str})"
        )

    def get_hyperparameters(self):
        params = super().get_hyperparameters()
        params.update(
            {
                "iterations": jsonable_hyperparameter(self.iterations),
                "exploration_weight": jsonable_hyperparameter(self.exploration_weight),
                "depth": jsonable_hyperparameter(self.depth),
                "max_outcomes": jsonable_hyperparameter(self.max_outcomes),
                "max_expanded_actions": jsonable_hyperparameter(
                    self.max_expanded_actions
                ),
            }
        )
        return params

    def _parent_env(self, node: _TreeNode) -> FoodDeliveryGymEnv:
        """Estado St do nó: filhos têm env próprio; raiz usa o ambiente real só como fonte."""
        if node.env is not None:
            return node.env
        return self.gym_env

    def _selection_score(self, node: _TreeNode, action_stats: _ActionStats) -> float:
        if action_stats.action_visits == 0:
            return float("inf")
        bonus = self.exploration_weight * math.sqrt(
            math.log(max(node.visit_count, 1)) / action_stats.action_visits
        )
        return action_stats.action_value + bonus

    def _pick_uct_action(self, node: _TreeNode) -> int:
        best_action = None
        best_score = float("-inf")
        for action, action_stats in node.actions.items():
            score = self._selection_score(node, action_stats)
            if score > best_score:
                best_score = score
                best_action = action
        if best_action is None:
            raise RuntimeError("nó sem ações para seleção UCT")
        return best_action

    def _expand_outcome(
        self,
        parent_env: FoodDeliveryGymEnv,
        action: int,
        child_depth: int,
    ) -> tuple[_OutcomeChild, float]:
        """Ação + amostra Monte Carlo de W (nunca consome o RNG do ambiente real)."""
        scenario_seed = self._next_scenario_seed()
        cloned = parent_env.clone(future="resample", scenario_seed=scenario_seed)
        obs_after, reward, terminated, truncated, _info = cloned.step(action)
        sampled_future = _OutcomeChild(
            scenario_seed=scenario_seed,
            env=cloned,
            obs=obs_after,
            done=bool(terminated),
            truncated=bool(truncated),
            node=_TreeNode(
                env=cloned,
                obs=obs_after,
                done=bool(terminated),
                truncated=bool(truncated),
                depth=child_depth,
            ),
        )
        return sampled_future, float(reward)

    def _order_untried_actions(self, node: _TreeNode, num_drivers: int) -> None:
        """
        Ordena as ações ainda não tentadas pela contribuição imediata C(S, x).

        Avalia cada ação com clone+step (amostra de W) e guarda o desfecho em
        `pending` para reaproveitar na expansão.
        """
        parent_env = self._parent_env(node)
        child_depth = node.depth + 1
        for action in range(num_drivers):
            node.pending[action] = self._expand_outcome(
                parent_env, action, child_depth=child_depth
            )
        node.untried_actions = sorted(
            range(num_drivers),
            key=lambda a: (-node.pending[a][1], a),
        )

    def _simulate_leaf(self, node: _TreeNode) -> float:
        """
        SimPolicy na folha: sempre num clone resample descartável.

        Nunca roda rollout no env do nó (nem no ambiente real): o estado
        guardado na árvore precisa permanecer no ponto pós-ação para
        expansões futuras.
        """
        if node.done or node.truncated:
            return 0.0

        scenario_seed = self._next_scenario_seed()
        if node.is_live_root:
            hyp = self.gym_env.clone(future="resample", scenario_seed=scenario_seed)
            obs = hyp.get_observation()
            done = False
            truncated = False
        else:
            assert node.env is not None and node.obs is not None
            hyp = node.env.clone(future="resample", scenario_seed=scenario_seed)
            obs = hyp.get_observation()
            done = node.done
            truncated = node.truncated

        rollout_value, _trajectory, _terminal = self._rollout_from(
            hyp, obs, done, truncated
        )
        return float(rollout_value)

    def _run_iteration(self, root: _TreeNode, num_drivers: int) -> None:
        """
        Uma trajetória MCTS a partir da raiz.

        1. Expansão míope (Powell TreePolicy): enquanto |A(S)| < d_thr, abre a
           ação não tentada de maior recompensa imediata (já ordenada em
           untried_actions). O primeiro desfecho vem do cache `pending`.
        2. Seleção: com o limiar atingido (ou sem ações restantes), UCT escolhe
           o ramo; se a ação já tem max_outcomes futuros, desce a um deles.
        3. Expansão de W: ação UCT com menos futuros que max_outcomes.
        4. Simulação: política de base (_simulate_leaf / rollout) na folha.
        5. Atualização: _backup sobe o retorno pelas arestas visitadas.
        """
        backup_path: list[tuple[_TreeNode, _ActionStats]] = []
        node = root
        if self.max_expanded_actions is None:
            expansion_limit = num_drivers
        else:
            expansion_limit = min(num_drivers, self.max_expanded_actions)

        while True:
            # Folha: episódio acabou ou profundidade máxima -> simula e atualiza.
            if node.done or node.truncated or node.depth >= self.depth:
                leaf_value = self._simulate_leaf(node)
                self._backup(backup_path, leaf_value)
                return

            # Expansão míope: |A(S)| < d_thr e ainda há ações não tentadas.
            if node.untried_actions is None:
                self._order_untried_actions(node, num_drivers)
            if node.untried_actions and len(node.actions) < expansion_limit:
                action = node.untried_actions.pop(0)
                sampled_future, immediate_reward = node.pending.pop(action)
                action_stats = _ActionStats(
                    action=action,
                    immediate_reward=immediate_reward,
                    _alpha=self.alpha,
                )
                action_stats.outcomes.append(sampled_future)
                node.actions[action] = action_stats
                backup_path.append((node, action_stats))
                leaf_value = self._simulate_leaf(sampled_future.node)  # type: ignore[arg-type]
                self._backup(backup_path, leaf_value)
                return

            # Fallback:
            # - Nenhuma ação tentada -> fallback para a política de base.
            if not node.actions:
                leaf_value = self._simulate_leaf(node)
                self._backup(backup_path, leaf_value)
                return

            # Seleção: UCT escolhe a ação entre as já expandídas.
            action = self._pick_uct_action(node)
            action_stats = node.actions[action]

            # Expansão (outra amostra de W): ação ainda abaixo de max_outcomes.
            if len(action_stats.outcomes) < self.max_outcomes:
                sampled_future, immediate_reward = self._expand_outcome(
                    self._parent_env(node), action, child_depth=node.depth + 1
                )
                if action_stats.action_visits == 0:
                    action_stats.immediate_reward = immediate_reward
                action_stats.outcomes.append(sampled_future)
                backup_path.append((node, action_stats))
                leaf_value = self._simulate_leaf(sampled_future.node)  # type: ignore[arg-type]
                self._backup(backup_path, leaf_value)
                return

            # Descida: já tem max_outcomes futuros -> sorteia um e continua o laço.
            sampled_future = action_stats.outcomes[
                int(self._scenario_rng.integers(0, len(action_stats.outcomes)))
            ]
            backup_path.append((node, action_stats))
            node = sampled_future.node  # type: ignore[assignment]

    def _backup(
        self,
        backup_path: list[tuple[_TreeNode, _ActionStats]],
        leaf_value: float,
    ) -> None:
        """Atualiza continuation_value / action_visits de baixo para cima."""
        downstream_return = leaf_value
        for node, action_stats in reversed(backup_path):
            node.visit_count += 1
            action_stats.action_visits += 1
            backed_up_value = (
                action_stats.immediate_reward + self.alpha * downstream_return
            )
            old_value = action_stats.action_value
            new_value = (
                old_value + (backed_up_value - old_value) / action_stats.action_visits
            )
            if abs(self.alpha) < 1e-15:
                action_stats.continuation_value = 0.0
            else:
                action_stats.continuation_value = (
                    new_value - action_stats.immediate_reward
                ) / self.alpha
            downstream_return = backed_up_value

    def _serialize_tree(self, node: _TreeNode) -> dict:
        """Árvore explorada em dict JSON-serializável (sem envs)."""
        drivers = self._parent_env(node).simpy_env.state.drivers
        actions = []
        for action_stats in sorted(node.actions.values(), key=lambda s: s.action):
            actions.append(
                {
                    "action": int(action_stats.action),
                    "driver_id": int(drivers[action_stats.action].driver_id),
                    "immediate_reward": float(action_stats.immediate_reward),
                    "continuation_value": float(action_stats.continuation_value),
                    "action_value": float(action_stats.action_value),
                    "action_visits": int(action_stats.action_visits),
                    "outcomes": [
                        {
                            "scenario_seed": int(outcome.scenario_seed),
                            "node": (
                                self._serialize_tree(outcome.node)
                                if outcome.node is not None
                                else None
                            ),
                        }
                        for outcome in action_stats.outcomes
                    ],
                }
            )
        return {
            "depth": int(node.depth),
            "done": bool(node.done),
            "truncated": bool(node.truncated),
            "visit_count": int(node.visit_count),
            "actions": actions,
        }

    def _record_decision(
        self,
        root: _TreeNode,
        drivers: List[Driver],
        best_action: int,
    ) -> None:
        order = self.gym_env.get_current_order()
        simpy_env = self.gym_env.get_simpy_env()
        candidates = [
            {
                "action": action_stats.action,
                "driver_id": int(drivers[action_stats.action].driver_id),
                "immediate_reward": float(action_stats.immediate_reward),
                "continuation_value": float(action_stats.continuation_value),
                "action_value": float(action_stats.action_value),
                "action_visits": int(action_stats.action_visits),
                "num_outcomes": len(action_stats.outcomes),
            }
            for action_stats in sorted(
                root.actions.values(), key=lambda s: s.action
            )
        ]
        self.decision_log.append(
            {
                "decision_idx": len(self.decision_log),
                "sim_time": float(simpy_env.now),
                "order_id": int(order.order_id) if order is not None else None,
                "chosen_action": best_action,
                "chosen_driver_id": int(drivers[best_action].driver_id),
                "best_action_value": float(root.actions[best_action].action_value),
                "root_visits": int(root.visit_count),
                "iterations": int(self.iterations),
                "exploration_weight": float(self.exploration_weight),
                "depth": int(self.depth),
                "max_outcomes": int(self.max_outcomes),
                "max_expanded_actions": self.max_expanded_actions,
                "alpha": float(self.alpha),
                "horizon": self.horizon,
                "candidates": candidates,
                "tree": self._serialize_tree(root),
            }
        )

    def select_driver(self, obs: dict, drivers: List[Driver], route: Route):
        num_drivers = len(drivers)
        if num_drivers == 0:
            return None

        # depth=0: sem árvore
        if self.depth == 0:
            return super().select_driver(obs, drivers, route)

        # Raiz lógica no estado atual.
        root = _TreeNode(
            env=None,
            obs=None,
            done=False,
            truncated=False,
            depth=0,
        )

        # Iterações MCTS.
        for _ in range(self.iterations):
            self._run_iteration(root, num_drivers)

        # Nenhuma ação tentada -> fallback para a política de base.
        if not root.actions:
            return super().select_driver(obs, drivers, route)

        # Ação com maior action_value
        best_action = max(
            root.actions.values(),
            key=lambda action_stats: action_stats.action_value,
        ).action

        if self.record_decisions:
            self._record_decision(root, drivers, best_action)

        return best_action
