import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def create_trajectory_subset(dataset_dict, num_transitions, seed):
    """
    Creates a subset of the dataset by selecting the first K trajectories
    from a shuffled list to meet or exceed the desired number of transitions.

    Args:
        dataset_dict: The full dataset as a dictionary of numpy arrays.
        num_transitions: The target number of transitions for the subset.
        seed: A random seed for shuffling trajectories to ensure reproducibility.

    Returns:
        A new dataset dictionary containing the subset of data.
    """
    if 'terminals' not in dataset_dict:
        raise ValueError("Dataset must contain 'terminals' to identify trajectories.")

    print(f"Creating data subset with target of {num_transitions} transitions...")

    # Identify the start and end indices of each trajectory
    terminal_locs = np.nonzero(dataset_dict['terminals'] > 0)[0]
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
    trajectories = list(zip(initial_locs, terminal_locs))

    # Shuffle trajectories for random subsampling
    rng = np.random.RandomState(seed)
    rng.shuffle(trajectories)

    selected_indices = []
    current_transitions = 0
    for start, end in trajectories:
        # The number of transitions in a trajectory is (end - start + 1)
        num_traj_transitions = end - start + 1
        
        selected_indices.extend(range(start, end + 1))
        current_transitions += num_traj_transitions
        if current_transitions >= num_transitions:
            break
    
    if current_transitions < num_transitions:
            print(f"Warning: Requested {num_transitions} transitions, but only found {current_transitions} in the selected trajectories.")

    selected_indices = np.array(selected_indices, dtype=np.int64)
    
    subset_dict = {
        key: arr[selected_indices] for key, arr in dataset_dict.items()
    }

    # The last 'terminal' flag in the new subset must be 1 for HGCDataset's logic to work.
    subset_dict['terminals'][-1] = 1.0
    
    # If the original dataset was compact, we must also fix the 'valids' array.
    if 'valids' in subset_dict:
        subset_dict['valids'] = 1.0 - subset_dict['terminals']

    print(f"Subset created with {current_transitions} actual transitions.")
    return subset_dict

    
def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

        if self.config['frame_stack'] is not None:
            # Only support compact (observation-only) datasets.
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(np.arange(self.size))
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))

    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        """Sample goals for the given indices."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )

            # Goals at the current state.
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        if self.config['frame_stack'] is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Return the frame-stacked observations for the given indices."""
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(self.config['frame_stack'])):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support high-level actor goals and prediction targets. It reads the following
    additional key from the config:
    - subgoal_steps: Subgoal steps (i.e., the number of steps to reach the low-level goal).
    """

    def sample(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals from the dataset. The goals are stored in the keys
        'value_goals', 'low_actor_goals', 'high_actor_goals', and 'high_actor_targets'. It also computes the 'rewards'
        and 'masks' based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        # Sample value goals.
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        batch['value_goals'] = self.get_observations(value_goal_idxs)

        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # Set low-level actor goals.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        low_goal_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
        batch['low_actor_goals'] = self.get_observations(low_goal_idxs)
        
        # Compute immediate rewards and masks for low-level critic
        # r(s_t, z_t) = 1 if s_{t+1} == z_t else 0 (or -1 if gc_negative)
        low_immediate_successes = (idxs == low_goal_idxs).astype(float)
        batch['low_rewards'] = low_immediate_successes - (1.0 if self.config['gc_negative'] else 0.0)
        batch['low_masks'] = 1.0 - low_immediate_successes  # 0 if we reached the subgoal, 1 otherwise

        # Sample high-level actor goals and set prediction targets.
        # High-level future goals.
        if self.config['actor_geom_sample']:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        high_traj_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], high_traj_goal_idxs)

        # High-level random goals.
        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        high_random_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)

        # Pick between high-level future goals and random goals.
        pick_random = np.random.rand(batch_size) < self.config['actor_p_randomgoal']
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)

        batch['high_actor_goals'] = self.get_observations(high_goal_idxs)
        batch['high_actor_targets'] = self.get_observations(high_target_idxs)
        
        # Compute k-step cumulative discounted rewards for high-level critic
        # This is the sum of discounted rewards from s_t to s_k (where s_k is the high_actor_target)
        # print("+++++++++++++++++++++++++++++++++")
        # print(f"High actor goal:\n{high_goal_idxs}")
        # print(f"High actor target:\n{high_target_idxs}")
        # VECTORIZED: Compute k-step cumulative discounted rewards
        k_steps = high_target_idxs - idxs
        max_k = np.max(k_steps)
        # print(f"Max k:\n{max_k}")
        if max_k > 0:
            # Create a 2D grid of trajectory positions (batch_size, max_k)
            # Each row represents positions [0, 1, ..., max_k-1]
            traj_positions = np.arange(max_k)[None, :]  # Shape: (1, max_k)
            
            # Create trajectory indices for each item in batch
            # Shape: (batch_size, max_k)
            base_idxs = idxs[:, None]  # Shape: (batch_size, 1)
            traj_idxs = base_idxs + traj_positions  # Broadcasting to (batch_size, max_k)
            # print(f"Traj ids:\n{traj_idxs}")
            # Create mask for valid positions within each trajectory's actual length
            valid_mask = traj_positions < k_steps[:, None]  # Shape: (batch_size, max_k)
            
            # Compute rewards based on TRANSITIONS, not states
            # Check if transition from traj_idxs[i] leads to goal
            next_states = traj_idxs + 1
            goal_idxs_expanded = high_goal_idxs[:, None]  # Shape: (batch_size, 1)
            final_state_idxs_expanded = final_state_idxs[:, None]
            
            # Ensure transitions don't go beyond trajectory bounds
            within_trajectory = next_states <= final_state_idxs_expanded
            
            # Check if transition leads to goal
            transition_to_goal = (next_states == goal_idxs_expanded) & within_trajectory

            # Compute transition rewards
            if self.config['gc_negative']:
                # +1 for transitions to goal, -1 for other valid transitions, 0 for invalid
                traj_rewards = np.where(transition_to_goal & valid_mask, 1.0,
                                    np.where(valid_mask & within_trajectory, -1.0, 0.0))
            else:
                # +1 for transitions to goal, 0 otherwise  
                traj_rewards = (transition_to_goal & valid_mask).astype(float)
            # print(f"traj reward:\n{traj_rewards}")

            # Create discount factors - shape: (1, max_k)
            discounts = self.config['discount'] ** traj_positions
            # Apply discounts and sum only valid positions
            k_step_rewards = np.sum(traj_rewards * discounts * valid_mask, axis=1)
            # print(f"k step rewards:\n{k_step_rewards}")
        else:
            # All k_steps are 0
            immediate_success = (idxs == high_goal_idxs).astype(float)
            k_step_rewards = immediate_success - (1.0 if self.config['gc_negative'] else 0.0)

        batch['high_k_step_rewards'] = k_step_rewards
        batch['high_k_steps'] = k_steps

        # SIMPLIFIED: Mask is 0 when s_k == g (when subgoal target equals final goal)
        batch['high_masks'] = 1.0 - (high_target_idxs == high_goal_idxs).astype(float)
        # print(f"high masks:\n{batch['high_masks']}")

        assert np.all(batch['high_k_steps'] >= 0)
        assert np.all(batch['high_k_steps'] <= self.config['subgoal_steps'])
        assert np.isfinite(batch['high_k_step_rewards']).all()

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    [
                        'observations',
                        'next_observations',
                        'value_goals',
                        'low_actor_goals',
                        'high_actor_goals',
                        'high_actor_targets',
                    ],
                )

        return batch

    def validate_data_processing(self, batch, batch_size=1024, num_debug_samples=3):
        """Comprehensive validation of HGCDataset sampling with visual debugging"""
        
        print("="*80)
        print("COMPREHENSIVE HGCDataset VALIDATION")
        print("="*80)
        
        # Basic batch statistics
        print("\n1. BATCH OVERVIEW")
        print("-" * 40)
        print(f"Batch size: {batch_size}")
        print(f"Config: subgoal_steps={self.config['subgoal_steps']}, discount={self.config['discount']}")
        print(f"Config: gc_negative={self.config['gc_negative']}")
        
        print("\nBatch shapes:")
        for k, v in batch.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")
        
        # Overall statistics
        print("\n2. OVERALL STATISTICS")
        print("-" * 40)
        print(f"high_k_step_rewards: min={batch['high_k_step_rewards'].min():.3f}, max={batch['high_k_step_rewards'].max():.3f}, mean={batch['high_k_step_rewards'].mean():.3f}")
        print(f"high_k_steps: min={batch['high_k_steps'].min()}, max={batch['high_k_steps'].max()}, mean={batch['high_k_steps'].mean():.2f}")
        print(f"low_rewards: min={batch['low_rewards'].min():.3f}, max={batch['low_rewards'].max():.3f}, mean={batch['low_rewards'].mean():.3f}")
        
        # Sanity checks
        print("\n3. SANITY CHECKS")
        print("-" * 40)
        try:
            assert not np.any(np.isnan(batch['high_k_step_rewards'])), "❌ NaN in k-step rewards"
            print("✅ No NaN in high_k_step_rewards")
        except AssertionError as e:
            print(f"❌ {e}")
        
        try:
            assert not np.any(np.isinf(batch['high_k_step_rewards'])), "❌ Inf in k-step rewards"
            print("✅ No Inf in high_k_step_rewards")
        except AssertionError as e:
            print(f"❌ {e}")
            
        try:
            assert np.all(batch['high_k_steps'] >= 0), "❌ Negative k-steps"
            print("✅ All high_k_steps >= 0")
        except AssertionError as e:
            print(f"❌ {e}")
            
        try:
            assert np.all(batch['high_k_steps'] <= self.config['subgoal_steps']), "❌ k-steps exceed subgoal_steps"
            print("✅ All high_k_steps <= subgoal_steps")
        except AssertionError as e:
            print(f"❌ {e}")
        
        # Reconstruct original indices for debugging
        print("\n4. DETAILED SAMPLE ANALYSIS")
        print("-" * 40)
        
        # Select diverse samples for debugging
        debug_indices = []
        if batch_size >= 3:
            debug_indices = [0, batch_size // 2, batch_size - 1]  # First, middle, last
        elif batch_size == 2:
            debug_indices = [0, 1]
        else:
            debug_indices = [0]
        
        # We need to reconstruct the original sampling to get the trajectory information
        # This is a bit tricky since we only have the batch, but we can infer some information
        
        for sample_idx in debug_indices[:num_debug_samples]:
            print(f"\n{'='*20} SAMPLE {sample_idx + 1}/{len(debug_indices)} {'='*20}")
            
            # Extract data for this sample
            high_k_steps = int(batch['high_k_steps'][sample_idx])
            high_k_reward = float(batch['high_k_step_rewards'][sample_idx])
            high_mask = float(batch['high_masks'][sample_idx])
            low_reward = float(batch['low_rewards'][sample_idx])
            low_mask = float(batch['low_masks'][sample_idx])
            
            print(f"\nSample index in batch: {sample_idx}")
            print(f"High-level k-steps: {high_k_steps}")
            print(f"High-level k-step reward: {high_k_reward:.3f}")
            print(f"High-level mask: {high_mask:.1f}")
            print(f"Low-level reward: {low_reward:.3f}")
            print(f"Low-level mask: {low_mask:.1f}")
            
            # Visual representation of episode position
            # We can't reconstruct exact episode info, but we can show the relative positions
            print(f"\n📍 TRAJECTORY VISUALIZATION (k-steps = {high_k_steps}):")
            
            # Create a visual representation
            total_width = 60
            if high_k_steps > 0:
                step_width = min(total_width // (high_k_steps + 5), 3)  # Adaptive width
                
                visual = ["." for _ in range(total_width)]
                
                # Mark current position
                curr_pos = 10  # Arbitrary starting position for visualization
                if curr_pos < total_width:
                    visual[curr_pos] = "C"  # Current
                
                # Mark target position
                target_pos = min(curr_pos + high_k_steps, total_width - 1)
                if target_pos < total_width:
                    visual[target_pos] = "T"  # Target
                
                # Mark trajectory
                for i in range(curr_pos + 1, target_pos):
                    if i < total_width:
                        visual[i] = "-"
                
                print("   " + "".join(visual))
                print("   " + " " * curr_pos + "C" + " " * (target_pos - curr_pos - 1) + "T")
                print("   Legend: C=Current, T=Target, -=Trajectory, .=Other states")
            else:
                print("   C (k=0: current state is target)")
            
            # Analyze reward computation
            print(f"\n🎯 HIGH-LEVEL REWARD ANALYSIS:")
            print(f"   Expected reward range: [{-high_k_steps if self.config['gc_negative'] else 0}, {1 if not self.config['gc_negative'] else 0}]")
            
            # Check if reward makes sense
            expected_min = -high_k_steps if self.config['gc_negative'] else 0
            expected_max = 1 if not self.config['gc_negative'] else 0
            
            if expected_min <= high_k_reward <= expected_max:
                print(f"   ✅ Reward {high_k_reward:.3f} is within expected range")
            else:
                print(f"   ❌ Reward {high_k_reward:.3f} is outside expected range [{expected_min}, {expected_max}]")
            
            # Analyze specific reward cases
            if high_k_steps == 0:
                expected_reward = 1 - (1 if self.config['gc_negative'] else 0)
                if abs(high_k_reward - expected_reward) < 1e-6:
                    print(f"   ✅ k=0 case: reward correctly computed as {expected_reward}")
                else:
                    print(f"   ❌ k=0 case: expected {expected_reward}, got {high_k_reward}")
            else:
                # For k > 0, reward depends on whether any intermediate state equals goal
                if self.config['gc_negative']:
                    if high_k_reward == 0:
                        print(f"   ✅ Found goal along trajectory (reward = 0)")
                    elif high_k_reward == -high_k_steps:
                        print(f"   ✅ No goal found along trajectory (reward = -k)")
                    else:
                        print(f"   ⚠️  Partial goal achievement (reward = {high_k_reward})")
                else:
                    if high_k_reward == 1:
                        print(f"   ✅ Found goal along trajectory (reward = 1)")
                    elif high_k_reward == 0:
                        print(f"   ✅ No goal found along trajectory (reward = 0)")
                    else:
                        print(f"   ⚠️  Unexpected reward value: {high_k_reward}")
            
            print(f"\n🎯 LOW-LEVEL REWARD ANALYSIS:")
            expected_low_reward = 1 - (1 if self.config['gc_negative'] else 0)
            if abs(low_reward - expected_low_reward) < 1e-6:
                print(f"   ✅ Low-level immediate reward suggests current state equals subgoal")
            else:
                expected_low_reward = 0 - (1 if self.config['gc_negative'] else 0)
                if abs(low_reward - expected_low_reward) < 1e-6:
                    print(f"   ✅ Low-level immediate reward suggests current state ≠ subgoal")
                else:
                    print(f"   ❌ Unexpected low-level reward: {low_reward}")
            
            # Mask analysis
            print(f"\n🎭 MASK ANALYSIS:")
            if high_mask == 0.0:
                print(f"   ✅ High-level mask = 0: target state equals final goal (terminal)")
            elif high_mask == 1.0:
                print(f"   ✅ High-level mask = 1: target state ≠ final goal (continue)")
            else:
                print(f"   ❌ Unexpected high-level mask value: {high_mask}")
                
            if low_mask == 0.0:
                print(f"   ✅ Low-level mask = 0: current state equals subgoal (terminal)")
            elif low_mask == 1.0:
                print(f"   ✅ Low-level mask = 1: current state ≠ subgoal (continue)")
            else:
                print(f"   ❌ Unexpected low-level mask value: {low_mask}")
        
        # Statistical validation
        print(f"\n{'='*20} STATISTICAL VALIDATION {'='*20}")
        
        # Check reward distributions
        unique_high_rewards, counts_high = np.unique(batch['high_k_step_rewards'], return_counts=True)
        unique_low_rewards, counts_low = np.unique(batch['low_rewards'], return_counts=True)
        
        print(f"\nHigh-level reward distribution:")
        for reward, count in zip(unique_high_rewards, counts_high):
            percentage = count / batch_size * 100
            print(f"   {reward:.3f}: {count:4d} samples ({percentage:5.1f}%)")
        
        print(f"\nLow-level reward distribution:")
        for reward, count in zip(unique_low_rewards, counts_low):
            percentage = count / batch_size * 100
            print(f"   {reward:.3f}: {count:4d} samples ({percentage:5.1f}%)")
        
        # Check k-step distribution
        unique_k_steps, counts_k = np.unique(batch['high_k_steps'], return_counts=True)
        print(f"\nK-step distribution:")
        for k_step, count in zip(unique_k_steps, counts_k):
            percentage = count / batch_size * 100
            print(f"   k={k_step:2d}: {count:4d} samples ({percentage:5.1f}%)")
        
        # Correlation analysis
        print(f"\n📊 CORRELATION ANALYSIS:")
        
        # Check if rewards correlate sensibly with k-steps
        zero_k_mask = batch['high_k_steps'] == 0
        nonzero_k_mask = batch['high_k_steps'] > 0
        
        if np.any(zero_k_mask):
            zero_k_rewards = batch['high_k_step_rewards'][zero_k_mask]
            print(f"   k=0 rewards: mean={zero_k_rewards.mean():.3f}, std={zero_k_rewards.std():.3f}")
        
        if np.any(nonzero_k_mask):
            nonzero_k_rewards = batch['high_k_step_rewards'][nonzero_k_mask]
            nonzero_k_steps = batch['high_k_steps'][nonzero_k_mask]
            print(f"   k>0 rewards: mean={nonzero_k_rewards.mean():.3f}, std={nonzero_k_rewards.std():.3f}")
            
            # Check if larger k tends to give more negative rewards (in gc_negative case)
            if self.config['gc_negative'] and len(nonzero_k_steps) > 1:
                correlation = np.corrcoef(nonzero_k_steps, nonzero_k_rewards)[0, 1]
                print(f"   Correlation between k-steps and rewards: {correlation:.3f}")
                if correlation < -0.1:
                    print(f"   ✅ Negative correlation makes sense (larger k → more negative reward)")
                elif correlation > 0.1:
                    print(f"   ❌ Positive correlation is unexpected")
                else:
                    print(f"   ⚠️  Low correlation might indicate sparse rewards")
        
        print(f"\n{'='*80}")
        print("VALIDATION COMPLETE")
        print("="*80)

    def debug_sample_and_validate(self, batch_size=256, num_debug_samples=3):
        """Sample a batch and run comprehensive validation"""
        batch = self.sample(batch_size)
        self.validate_data_processing(batch, batch_size, num_debug_samples)
        return batch

    def test_edge_cases(self):
        """Test specific edge cases in data sampling"""
        print("\n" + "="*80)
        print("TESTING EDGE CASES")
        print("="*80)
        
        # Test case 1: Small batch
        print("\n1. Testing small batch (size=1)")
        batch_small = self.sample(1)
        print(f"   high_k_steps: {batch_small['high_k_steps'][0]}")
        print(f"   high_k_step_rewards: {batch_small['high_k_step_rewards'][0]:.3f}")
        
        # Test case 2: Check if k=0 cases are handled correctly
        print("\n2. Looking for k=0 cases in larger sample...")
        batch_large = self.sample(1000)
        k_zero_mask = batch_large['high_k_steps'] == 0
        k_zero_count = np.sum(k_zero_mask)
        print(f"   Found {k_zero_count} samples with k=0 out of 1000")
        
        if k_zero_count > 0:
            k_zero_rewards = batch_large['high_k_step_rewards'][k_zero_mask]
            expected_k_zero_reward = 1 - (1 if self.config['gc_negative'] else 0)
            correct_k_zero = np.sum(np.abs(k_zero_rewards - expected_k_zero_reward) < 1e-6)
            print(f"   {correct_k_zero}/{k_zero_count} k=0 samples have correct reward ({expected_k_zero_reward})")
        
        # Test case 3: Check maximum k values
        max_k = np.max(batch_large['high_k_steps'])
        print(f"\n3. Maximum k-steps found: {max_k} (config subgoal_steps: {self.config['subgoal_steps']})")
        if max_k == self.config['subgoal_steps']:
            print("   ✅ Max k equals subgoal_steps as expected")
        else:
            print(f"   ⚠️  Max k ({max_k}) differs from subgoal_steps ({self.config['subgoal_steps']})")
        
        # Test case 4: Reward bounds checking
        print(f"\n4. Reward bounds checking:")
        min_possible = -self.config['subgoal_steps'] if self.config['gc_negative'] else 0
        max_possible = 1 if not self.config['gc_negative'] else 0
        
        actual_min = np.min(batch_large['high_k_step_rewards'])
        actual_max = np.max(batch_large['high_k_step_rewards'])
        
        print(f"   Theoretical bounds: [{min_possible}, {max_possible}]")
        print(f"   Actual bounds: [{actual_min:.3f}, {actual_max:.3f}]")
        
        if min_possible <= actual_min and actual_max <= max_possible:
            print("   ✅ All rewards within theoretical bounds")
        else:
            print("   ❌ Some rewards outside theoretical bounds!")
        
        print("\n" + "="*80)

    def validate_trajectory_consistency(self, batch_size=100):
        """Updated validation with correct discounting logic"""
        print("\n" + "="*80)
        print("TRAJECTORY CONSISTENCY VALIDATION")
        print("="*80)
        
        batch = self.sample(batch_size)
        
        print(f"\n1. Checking reward-mask consistency:")
        inconsistencies = 0
        
        for i in range(batch_size):
            high_reward = batch['high_k_step_rewards'][i]
            high_mask = batch['high_masks'][i]
            k_steps = batch['high_k_steps'][i]
            
            if k_steps > 0:
                # Compute what the reward would be if no goals were reached
                discounts = self.config['discount'] ** np.arange(k_steps)
                if self.config['gc_negative']:
                    min_possible_reward = -1 * np.sum(discounts)  # All steps give -1
                    goal_achieved = (high_reward > min_possible_reward)  # Small epsilon for numerical precision
                else:
                    max_possible_reward = 0  # All steps give 0 when no goal reached
                    goal_achieved = (high_reward > max_possible_reward)
            else:
                # k=0 case: reward should be 0 if at goal, -1 otherwise
                if self.config['gc_negative']:
                    goal_achieved = (high_reward == 0)
                else:
                    goal_achieved = (high_reward == 1)
            
            # Check consistency: if goal achieved, mask should be 0
            if goal_achieved and high_mask != 0.0:
                if k_steps > 0:
                    min_reward = -1 * np.sum(self.config['discount'] ** np.arange(k_steps))
                    print(f"  Sample {i}: Goal achieved (reward={high_reward:.3f} > {min_reward:.3f}) but mask={high_mask} (should be 0)")
                else:
                    print(f"  Sample {i}: Goal achieved (reward={high_reward:.3f}) but mask={high_mask} (should be 0)")
                inconsistencies += 1
            elif not goal_achieved and high_mask == 0.0:
                if k_steps > 0:
                    min_reward = -1 * np.sum(self.config['discount'] ** np.arange(k_steps))
                    print(f"  Sample {i}: Goal not achieved (reward={high_reward:.3f} <= {min_reward:.3f}) but mask={high_mask} (should be 1)")
                else:
                    print(f"  Sample {i}: Goal not achieved (reward={high_reward:.3f}) but mask={high_mask} (should be 1)")
                inconsistencies += 1
        
        print(f"   Found {inconsistencies}/{batch_size} reward-mask inconsistencies")
        if inconsistencies == 0:
            print("   ✅ All reward-mask pairs are consistent")
        else:
            print(f"   ❌ {inconsistencies} inconsistencies found!")
        
        # Show some statistics to understand the data better
        print(f"\n   Statistics:")
        print(f"   - High k_steps: min={batch['high_k_steps'].min()}, max={batch['high_k_steps'].max()}, mean={batch['high_k_steps'].mean():.2f}")
        print(f"   - High rewards: min={batch['high_k_step_rewards'].min():.3f}, max={batch['high_k_step_rewards'].max():.3f}, mean={batch['high_k_step_rewards'].mean():.3f}")
        print(f"   - High masks: min={batch['high_masks'].min()}, max={batch['high_masks'].max()}, mean={batch['high_masks'].mean():.3f}")
        
        # Check if all rewards are the same (which would be suspicious)
        unique_rewards = np.unique(batch['high_k_step_rewards'])
        if len(unique_rewards) <= 5:  # Few unique values
            print(f"   - Unique reward values: {unique_rewards}")
            
        print(f"\n2. Checking low-level reward-mask consistency:")
        low_inconsistencies = 0
        
        for i in range(batch_size):
            low_reward = batch['low_rewards'][i]
            low_mask = batch['low_masks'][i]
            
            # Low-level reward should be consistent with mask
            if self.config['gc_negative']:
                at_subgoal = (low_reward == 0)
            else:
                at_subgoal = (low_reward == 1)
            
            if at_subgoal and low_mask != 0.0:
                low_inconsistencies += 1
            elif not at_subgoal and low_mask == 0.0:
                low_inconsistencies += 1
        
        print(f"   Found {low_inconsistencies}/{batch_size} low-level reward-mask inconsistencies")
        if low_inconsistencies == 0:
            print("   ✅ All low-level reward-mask pairs are consistent")
        else:
            print(f"   ❌ {low_inconsistencies} low-level inconsistencies found!")
        
        print("\n" + "="*80)

    def sample_original(self, batch_size, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals from the dataset. The goals are stored in the keys
        'value_goals', 'low_actor_goals', 'high_actor_goals', and 'high_actor_targets'. It also computes the 'rewards'
        and 'masks' based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config['frame_stack'] is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        # Sample value goals.
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        batch['value_goals'] = self.get_observations(value_goal_idxs)

        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # Set low-level actor goals.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        low_goal_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)
        batch['low_actor_goals'] = self.get_observations(low_goal_idxs)

        critic_subgoal_steps = np.random.randint(1, 3, size=batch_size)
        low_critic_goal_idxs = np.minimum(idxs + critic_subgoal_steps, final_state_idxs)
        batch['low_critic_goals'] = self.get_observations(low_critic_goal_idxs)

        # Sample high-level actor goals and set prediction targets.
        # High-level future goals.
        if self.config['actor_geom_sample']:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config['discount'], size=batch_size)  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        high_traj_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], high_traj_goal_idxs)

        # High-level random goals.
        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        high_random_target_idxs = np.minimum(idxs + self.config['subgoal_steps'], final_state_idxs)

        # Pick between high-level future goals and random goals.
        pick_random = np.random.rand(batch_size) < self.config['actor_p_randomgoal']
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)

        batch['high_actor_goals'] = self.get_observations(high_goal_idxs)
        batch['high_actor_targets'] = self.get_observations(high_target_idxs)

        if self.config['p_aug'] is not None and not evaluation:
            if np.random.rand() < self.config['p_aug']:
                self.augment(
                    batch,
                    [
                        'observations',
                        'next_observations',
                        'value_goals',
                        'low_actor_goals',
                        'low_critic_goals',
                        'high_actor_goals',
                        'high_actor_targets',
                    ],
                )

        return batch