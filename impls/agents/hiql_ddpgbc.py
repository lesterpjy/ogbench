import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCValue, Identity, LengthNormalize


class HIQLDDPGBCAgent(flax.struct.PyTreeNode):
    """
    Hierarchical Implicit Q-Learning (HIQL) with DDPG+BC policy extraction.

    This agent implements the three-phase training proposed by the user:
    1.  Learn a goal-conditioned state-value function V(s, g) using an IQL-style expectile loss.
    2.  Learn action-value functions (critics) Q_h and Q_l by treating the learned V-function as a target.
        - Q_h(s, g, z): High-level critic for choosing a subgoal representation 'z'.
        - Q_l(s, z, a): Low-level critic for choosing a primitive action 'a' to reach subgoal 'z'.
    3.  Extract deterministic policies mu_h and mu_l using a DDPG+BC objective, which uses the learned Q-functions.

    All phases are trained end-to-end in a single update step, with gradients carefully controlled
    using jax.lax.stop_gradient to maintain the logic of the phased approach.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # This is the original IQL value loss from the base HIQL agent.
    # It learns the state-value function V(s, g).
    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """
        Computes the expectile loss, which is an asymmetric squared loss.
        This is the core of Implicit Q-Learning (IQL).

        Args:
            adv: The advantage (Q - V). Used to determine the weighting.
            diff: The difference to be penalized (e.g., Q - V).
            expectile: The expectile level (τ). A value of 0.5 is standard MSE. A value > 0.5
                       weights positive errors more heavily, pushing the learned value up.

        Why:
        By setting τ > 0.5, we asymmetrically penalize the difference between the Q-value and V-value.
        This forces the learned V-value to be an upper expectile of the Q-values for actions in the
        dataset, effectively approximating the 'max' operator over actions without ever needing to
        query actions outside the dataset.
        """
        # Determine the weight: 'expectile' for positive advantages, '1 - expectile' for negative.
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        # return weighted squared difference
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """
        Phase 1: Compute the IQL state-value function loss for V(s, g).
        Identical to the original HIQL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        # --- Clipped Double Q-Learning for Target Value Calculation ---
        # To stabilize training, we use two target value networks (ensemble) and take the minimum.
        # This is the "Clipped Double" technique, which helps prevent overestimation of values.
        # (next_v1_t, next_v2_t) are the values of the *next* state from the two *target* networks.
        (next_v1_t, next_v2_t) = self.network.select("target_value")(
            batch["next_observations"], batch["value_goals"]
        )
        # Take the minimum of the two target values to get a conservative estimate. This is the "Clipped" part.
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)

        # Calculate the bootstrapped target 'q' using the conservative next-state value.
        q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v_t

        # To calculate the advantage (for weighting the loss), we need the value of the *current* state.
        # We also get this from the target networks for stability.
        (v1_t, v2_t) = self.network.select("target_value")(
            batch["observations"], batch["value_goals"]
        )
        # The advantage is typically calculated against the average of the two critics.
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        # --- Expectile Loss Calculation ---
        # Now, calculate the loss for the *online* networks (the ones we are taking gradients for).
        # We need to compute the full TD targets again, but this time without clipping, one for each critic head.
        q1 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v1_t
        q2 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v2_t
        # Get the current value predictions from the online networks. `grad_params` ensures gradients flow here.
        (v1, v2) = self.network.select("value")(
            batch["observations"], batch["value_goals"], params=grad_params
        )
        v = (v1 + v2) / 2

        # Calculate the expectile loss for each of the two value networks.
        # The `adv` (from target nets) is used for weighting, but the `diff` (q1 - v1) is what's being minimized.
        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config["expectile"]).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config["expectile"]).mean()
        # The total loss is the sum of the losses for the two heads.
        value_loss = value_loss1 + value_loss2

        # Return the loss and metrics for logging.
        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """
        Phase 2: Learn Q-functions (critics) from the learned V-function via supervised regression.
        The goal is to make Q(s,a,g) predict V(s',g).
        """
        info = {}

        # --- High-Level Critic Loss: L(Q_h) ---
        # The target value for the high-level critic Q_h(s, g, z) is V(s_k, g),
        # where s_k is the k-step future state, representing the achieved subgoal.
        # We use the target V-network for stability.
        high_level_q_target_1, high_level_q_target_2 = self.network.select(
            "target_value"
        )(batch["high_actor_targets"], batch["high_actor_goals"])
        # We still use clipping for the target value to be conservative.
        high_level_q_target = jnp.minimum(high_level_q_target_1, high_level_q_target_2)
        # CRITICAL: Stop gradients here. We treat V(s_k, g) as a fixed label.
        # The error in the Q-function should not change the V-function.
        high_level_q_target = jax.lax.stop_gradient(high_level_q_target)

        # The "action" for the high-level critic is the subgoal representation z = phi(s_k).
        # We get this by passing the subgoal state s_k through the representation function.
        subgoal_reps = self.network.select("goal_rep")(
            jnp.concatenate(
                [batch["observations"], batch["high_actor_targets"]], axis=-1
            )
        )
        # Stop gradients as this is part of the "target" action.
        subgoal_reps = jax.lax.stop_gradient(subgoal_reps)

        # Predict Q_h for the given state, goal, and subgoal representation.
        q_h_pred_1, q_h_pred_2 = self.network.select("high_critic")(
            batch["observations"],
            batch["high_actor_goals"],
            subgoal_reps,
            params=grad_params,
        )
        # The loss is simple Mean-Squared-Error, as we're doing supervised regression.
        high_critic_loss = (
            (q_h_pred_1 - high_level_q_target) ** 2
            + (q_h_pred_2 - high_level_q_target) ** 2
        ).mean()
        info["high_critic_loss"] = high_critic_loss

        # --- Low-Level Critic Loss: L(Q_l) ---
        # logic is identical to the high-level critic, but with different inputs.
        # target for Q_l(s_t, z_t, a_t) is V(s_{t+1}, z_t), i.e., the value of the next state w.r.t the current subgoal.
        low_level_q_target_1, low_level_q_target_2 = self.network.select(
            "target_value"
        )(batch["next_observations"], batch["low_actor_goals"])
        low_level_q_target = jnp.minimum(low_level_q_target_1, low_level_q_target_2)
        low_level_q_target = jax.lax.stop_gradient(low_level_q_target)

        # The "action" is the primitive action `a_t` from the dataset.
        # The subgoal `z_t` is part of the state/context for the low-level critic.
        subgoal_reps_low = self.network.select("goal_rep")(
            jnp.concatenate([batch["observations"], batch["low_actor_goals"]], axis=-1),
        )
        # Predict Q_l.
        q_l_pred_1, q_l_pred_2 = self.network.select("low_critic")(
            batch["observations"],
            subgoal_reps_low,
            batch["actions"],
            goal_encoded=True,
            params=grad_params,
        )
        # Compute MSE loss.
        low_critic_loss = (
            (q_l_pred_1 - low_level_q_target) ** 2
            + (q_l_pred_2 - low_level_q_target) ** 2
        ).mean()
        info["low_critic_loss"] = low_critic_loss

        # The total critic loss is the sum of both.
        total_critic_loss = high_critic_loss + low_critic_loss
        return total_critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Phase 3: Extract policies mu_h and mu_l using DDPG+BC objective."""
        info = {}
        high_actor_rng, low_actor_rng = jax.random.split(
            rng
        )  # Not used by mode() but good practice

        # --- High-Level Actor Loss: J(mu_h) ---
        # Get the high-level policy's output (a deterministic subgoal representation).
        high_actor_dist = self.network.select("high_actor")(
            batch["observations"], batch["high_actor_goals"], params=grad_params
        )
        # for deterministic policy, use the mode of the distribution (mean for a Gaussian)
        pred_subgoal_reps = high_actor_dist.mode()

        # DDPG component: Maximize the Q-value of the action chosen by the policy.
        # We evaluate the predicted subgoal representation using the high-level critic.
        q_h_1, q_h_2 = self.network.select("high_critic")(
            batch["observations"], batch["high_actor_goals"], pred_subgoal_reps
        )
        # Use the minimum of the two critics to get a conservative Q-estimate (Clipped Q-Learning).
        q_h = jnp.minimum(q_h_1, q_h_2)
        # Maximizing Q is equivalent to minimizing -Q.
        high_q_loss = -q_h.mean()

        # Behavioral Cloning (BC) component: Regularize the policy to stay close to the dataset "actions".
        # The target "action" for the high-level policy is the representation of the future state s_k.
        target_subgoal_reps = self.network.select("goal_rep")(
            jnp.concatenate(
                [batch["observations"], batch["high_actor_targets"]], axis=-1
            )
        )
        # We compute MSE between the policy's output and the target representation.
        # The target is a fixed label, so we stop gradients.
        high_bc_loss = (
            (pred_subgoal_reps - jax.lax.stop_gradient(target_subgoal_reps)) ** 2
        ).mean()

        # The final actor loss is a weighted sum of the DDPG and BC components.
        high_actor_loss = self.config["high_lambda"] * high_q_loss + high_bc_loss
        info["high_actor_loss"] = high_actor_loss
        info["high_q_loss"] = high_q_loss
        info["high_bc_loss"] = high_bc_loss

        # --- Low-Level Actor Loss: J(mu_l) ---
        # The logic is identical, but for the low-level policy.
        # Get the subgoal representation for the low-level policy's context.
        subgoal_reps_low = self.network.select("goal_rep")(
            jnp.concatenate([batch["observations"], batch["low_actor_goals"]], axis=-1),
            # Pass grad_params to allow gradient flow
            # from the low-level actor loss into the goal representation network (phi).
            # This is controlled by the `low_actor_rep_grad` flag.
            params=grad_params if self.config["low_actor_rep_grad"] else None,
        )
        # Get the deterministic primitive action from the low-level policy.
        low_actor_dist = self.network.select("low_actor")(
            batch["observations"],
            subgoal_reps_low,
            goal_encoded=True,
            params=grad_params,
        )
        pred_actions = low_actor_dist.mode()

        # DDPG component: Evaluate the predicted action with the low-level critic.
        q_l_1, q_l_2 = self.network.select("low_critic")(
            batch["observations"], subgoal_reps_low, pred_actions, goal_encoded=True
        )
        q_l = jnp.minimum(q_l_1, q_l_2)
        low_q_loss = -q_l.mean()

        # BC component: Regularize towards the primitive actions from the dataset.
        low_bc_loss = ((pred_actions - batch["actions"]) ** 2).mean()

        # Combine the losses.
        low_actor_loss = self.config["low_lambda"] * low_q_loss + low_bc_loss
        info["low_actor_loss"] = low_actor_loss
        info["low_q_loss"] = low_q_loss
        info["low_bc_loss"] = low_bc_loss

        total_actor_loss = high_actor_loss + low_actor_loss
        return total_actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss by combining all three phases."""
        info = {}
        # Get a new random key for this update step.
        rng = rng if rng is not None else self.rng
        rng, actor_rng = jax.random.split(rng)

        # Compute the loss for Phase 1 (learning V).
        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        # Compute the loss for Phase 2 (learning Qs).
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        # Compute the loss for Phase 3 (learning policies).
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        # Combine all losses into a single scalar for the optimizer.
        # The weights allow tuning the relative importance of each component.
        loss = (
            self.config["value_loss_weight"] * value_loss
            + self.config["critic_loss_weight"] * critic_loss
            + self.config["actor_loss_weight"] * actor_loss
        )
        return loss, info

    def target_update(self, network, module_name):
        """
        Performs a soft update of the target network parameters (Polyak averaging).
        target_params = τ * online_params + (1 - τ) * target_params
        This makes training more stable than a hard copy.
        """
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        """
        Performs a single gradient update step for the entire agent.
        """
        # Get a new random key for this step.
        new_rng, rng = jax.random.split(self.rng)

        # Define the function that computes the total loss.
        # `jax.grad` will differentiate this function with respect to its first argument (`grad_params`).
        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        # `apply_loss_fn` is a helper that computes gradients and applies them.
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        # After updating the online value network, update its target network via Polyak averaging.
        self.target_update(new_network, "value")

        # Return the new agent state (with updated params and rng) and the info dict.
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,  # Seed is not used for deterministic policy but kept for API consistency
        temperature=1.0,  # Temperature is unused for deterministic policy, but kept for API consistency
    ):
        """
        Sample actions from the hierarchical deterministic policy for evaluation/execution.
        1. High-level policy mu_h(s, g) -> z (subgoal representation)
        2. Low-level policy mu_l(s, z) -> a (primitive action)
        """
        # High-level policy predicts a deterministic subgoal representation. `temperature` is ignored.
        high_dist = self.network.select("high_actor")(
            observations, goals, temperature=1.0
        )
        goal_reps = high_dist.mode()

        # Low-level policy takes the state and the predicted subgoal representation to produce a primitive action.
        low_dist = self.network.select("low_actor")(
            observations, goal_reps, goal_encoded=True, temperature=1.0
        )
        actions = low_dist.mode()

        # Clip actions to be within the valid range [-1, 1].
        if not self.config["discrete"]:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """
        Creates and initializes all the networks for the agent.
        This method is called once at the beginning of training.
        """
        # Setup random number generation.
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        # Get example data shapes and dimensions.
        ex_goals = ex_observations
        if config["discrete"]:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # --- Define Shared Goal Representation and V-function Networks (from original HIQL) ---
        # `goal_rep_def` is the phi(g) network. It may start with a CNN encoder for pixels.
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            goal_rep_seq = [encoder_module()]
        else:
            goal_rep_seq = []
        # It's followed by an MLP and a normalization layer.
        goal_rep_seq.extend(
            [
                MLP(
                    hidden_dims=(*config["value_hidden_dims"], config["rep_dim"]),
                    activate_final=False,
                    layer_norm=config["layer_norm"],
                ),
                LengthNormalize(),
            ]
        )
        goal_rep_def = nn.Sequential(goal_rep_seq)

        # `value_encoder_def` defines how inputs are processed before the value MLP.
        if config["encoder"] is not None:
            value_encoder_def = GCEncoder(
                state_encoder=encoder_module(), concat_encoder=goal_rep_def
            )
            target_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
        else:
            value_encoder_def = GCEncoder(
                state_encoder=Identity(), concat_encoder=goal_rep_def
            )
            target_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)

        # `value_def` is the final V-function network, which includes the encoder and an MLP.
        # It's ensembled (two heads) for Clipped Double Q-Learning.
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            gc_encoder=value_encoder_def,
        )
        target_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_value_encoder_def,
        )

        # --- Q-function (Critic) Networks ---
        # Low-level critic Q_l(s, z, a)
        # Inputs: s (observation), z (subgoal rep), a (primitive action)
        # We use a GCEncoder where the "goal" is the subgoal representation z.
        if config["encoder"] is not None:
            low_critic_encoder_def = GCEncoder(state_encoder=encoder_module())
        else:
            low_critic_encoder_def = GCEncoder(state_encoder=Identity())
        low_critic_def = GCValue(
            hidden_dims=config["critic_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            gc_encoder=low_critic_encoder_def,
        )

        # High-level critic Q_h(s, g, z)
        # Inputs: s, g (final goal), z (subgoal rep)
        # We treat [s, g] as the observation and z as the action.
        if config["encoder"] is not None:
            high_critic_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            high_critic_encoder_def = None
        high_critic_def = GCValue(
            hidden_dims=config["critic_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            gc_encoder=high_critic_encoder_def,
        )

        # --- Define Policy (Actor) Networks ---
        # Define low-level actor mu_l(s, z).
        # Inputs: s, z (subgoal rep)
        if config["encoder"] is not None:
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module())
        else:
            low_actor_encoder_def = GCEncoder(state_encoder=Identity())
        # `const_std=True` makes the GCActor output a deterministic mean.
        low_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            const_std=True,
            gc_encoder=low_actor_encoder_def,
        )

        # Define high-level actor mu_h(s, g).
        # Inputs: s, g
        if config["encoder"] is not None:
            high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())
        else:
            high_actor_encoder_def = None
        # It outputs a vector of size `rep_dim` (the subgoal representation).
        high_actor_def = GCActor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=config["rep_dim"],
            const_std=True,
            gc_encoder=high_actor_encoder_def,
        )

        # --- Initialize all networks using example inputs ---
        ex_subgoal_reps = jnp.zeros((ex_observations.shape[0], config["rep_dim"]))
        network_info = dict(
            # V-function networks
            goal_rep=(
                goal_rep_def,
                (jnp.concatenate([ex_observations, ex_goals], axis=-1),),
            ),
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            # critic networks
            high_critic=(high_critic_def, (ex_observations, ex_goals, ex_subgoal_reps)),
            low_critic=(low_critic_def, (ex_observations, ex_subgoal_reps, ex_actions)),
            # actor networks
            low_actor=(
                low_actor_def,
                (ex_observations, ex_subgoal_reps, True),
            ),  # goal_encoded=True
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        # `ModuleDict` groups all network definitions into a single nn.Module.
        network_def = ModuleDict(networks)
        # Initialize an Adam optimizer.
        network_tx = optax.adam(learning_rate=config["lr"])
        # `network_def.init` runs a forward pass with example data to create the network parameters.
        network_params = network_def.init(init_rng, **network_args)["params"]
        # `TrainState` is a container for the network, its parameters, and the optimizer state.
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Initialize the target network parameters to be the same as the online network parameters.
        params = network.params
        params["modules_target_value"] = params["modules_value"]

        # Return the fully initialized agent.
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict(
        {
            # Agent name
            "agent_name": "hiql_ddpgbc",
            # Learning rates and optimizers
            "lr": 3e-4,
            "batch_size": 1024,
            # Network architectures
            "actor_hidden_dims": (256, 256),
            "value_hidden_dims": (256, 256),
            "critic_hidden_dims": (256, 256),
            "layer_norm": True,
            "encoder": ml_collections.config_dict.placeholder(str),
            # RL parameters
            "discount": 0.99,
            "tau": 0.005,  # Target network update rate for V-function
            "expectile": 0.7,  # IQL expectile for V-function
            # HIQL parameters
            "rep_dim": 10,
            "low_actor_rep_grad": False,
            "subgoal_steps": 25,
            # DDPG+BC parameters
            "low_lambda": 1.0,  # Weight for Q-term in low-level actor loss
            "high_lambda": 1.0,  # Weight for Q-term in high-level actor loss
            "value_loss_weight": 1.0,
            "critic_loss_weight": 1.0,
            "actor_loss_weight": 1.0,
            # Other flags
            "discrete": False,
            "const_std": True,  # For deterministic policy in GCActor
            # Dataset processing
            "dataset_class": "HGCDataset",
            "value_p_curgoal": 0.2,
            "value_p_trajgoal": 0.5,
            "value_p_randomgoal": 0.3,
            "value_geom_sample": True,
            "actor_p_curgoal": 0.0,
            "actor_p_trajgoal": 1.0,
            "actor_p_randomgoal": 0.0,
            "actor_geom_sample": False,
            "gc_negative": True,
            "p_aug": 0.0,
            "frame_stack": ml_collections.config_dict.placeholder(int),
        }
    )
    return config
