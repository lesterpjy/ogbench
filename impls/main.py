import json
import os
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset, create_trajectory_subset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 500000, 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')

# Add flags for data scaling
flags.DEFINE_integer('value_data_transitions', None, 'Number of transitions for the value learning dataset. If None, uses the full dataset.')
flags.DEFINE_integer('policy_data_transitions', None, 'Number of transitions for the policy learning dataset. If None, uses the full dataset.')
flags.DEFINE_boolean('debug_data_and_exit', False, 'If True, run the data subsetting and sampling debug routine and then exit.')

config_flags.DEFINE_config_file('agent', 'agents/gciql.py', lock_config=False)

def debug_dataset_sampling(orig_train_dataset, config):
    """Run comprehensive dataset debugging"""

    train_dataset = HGCDataset(Dataset.create(**orig_train_dataset), config)
    
    print("🔍 STARTING COMPREHENSIVE DATASET VALIDATION")
    print("This will help identify issues in data sampling...")
    
    # 1. Basic validation with visual debugging
    print("\n🔹 Running basic validation...")
    train_dataset.debug_sample_and_validate(batch_size=30, num_debug_samples=5)
    
    # 2. Edge case testing
    print("\n🔹 Testing edge cases...")
    train_dataset.test_edge_cases()
    
    # 3. Trajectory consistency validation
    print("\n🔹 Checking trajectory consistency...")
    train_dataset.validate_trajectory_consistency(batch_size=30)
    
    print("\n✅ Dataset validation complete!")

def debug_subsetting_and_sampling(config):
    """
    Runs a full suite of tests on the dataset subsetting and sampling logic.
    """
    # --- START MODIFICATION: Add helper for visualization ---
    def visualize_compact_data(data_dict, start_idx, end_idx, title="Data Visualization"):
        """Prints a formatted table of the terminals and valids arrays."""
        print("\n" + "-"*60)
        print(f"👁️  {title} (Indices {start_idx}-{end_idx-1}) 👁️")
        print("-" * 60)
        print("This shows the structure of a 'compact' dataset. A valid transition")
        print("from index `i` to `i+1` exists only if `valids[i]` is 1.")
        print("\n" + "="*60)
        print(f"{'Index':>6s} | {'Terminals':>10s} | {'Valids':>8s} | Notes")
        print(f"{'-'*6} | {'-'*10} | {'-'*8} | {'-'*30}")

        # Reconstruct original terminals to identify true trajectory starts
        original_terminals = 1.0 - data_dict['valids']
        initial_locs = np.nonzero(np.concatenate([[1.0], np.diff(original_terminals) < 0]))[0]

        for i in range(start_idx, end_idx):
            if i >= len(data_dict['terminals']):
                break
            
            term = data_dict['terminals'][i]
            valid = data_dict['valids'][i]
            note = ""
            
            if valid == 0:
                note += "🔚 END OF TRAJECTORY (next obs is invalid)"
            
            if i in initial_locs:
                note += "🎬 START OF TRAJECTORY"

            # The penultimate state also has terminal=1 in compact format
            if term == 1 and valid == 1:
                note += " PENULTIMATE STATE"

            print(f"{i:6d} | {term:10.1f} | {valid:8.1f} | {note}")
        print("="*60 + "\n")
    # --- END MODIFICATION ---

    print("="*80)
    print("🔬 RUNNING DATA SUBSETTING AND SAMPLING VALIDATION")
    print("="*80)

    # 1. Load a large dataset to test on
    print("\n[STEP 1/5] Loading full compact dataset...")
    env, full_train_dataset, _ = make_env_and_datasets(
        'visual-puzzle-3x3-play-v0', frame_stack=config['frame_stack']
    )
    full_size = len(full_train_dataset['observations'])
    print(f"✅ Full compact dataset loaded with {full_size} transitions.")

    # 2. Verify the structure of the loaded compact dataset
    print("\n[STEP 2/5] Verifying structure of the full compact dataset...")
    original_terminals_reconstructed = 1.0 - full_train_dataset['valids']
    num_trajs_from_valids = int(np.sum(original_terminals_reconstructed))
    num_double_terminals = int(np.sum(full_train_dataset['terminals'] > 0))
    print(f"  - Number of trajectories (inferred from 'valids'): {num_trajs_from_valids}")
    print(f"  - Number of non-zero 'terminals' entries: {num_double_terminals}")
    assert num_double_terminals > num_trajs_from_valids, "Full dataset does not have the expected compact 'terminals' structure."
    print("✅ Structure appears correct.")

    # 3. Test the subsetting logic
    print("\n[STEP 3/5] Testing create_trajectory_subset function...")
    subset_size = 300_000
    DATA_SUBSET_SEED = 42

    subset_dict = create_trajectory_subset(
        full_train_dataset, subset_size, DATA_SUBSET_SEED
    )
    assert len(subset_dict['observations']) >= subset_size, "Subset is smaller than requested size."
    print("✅ Subsetting function executed successfully.")

    # 4. Verify the structure of the created subset
    print("\n[STEP 4/5] Verifying structure of the created subset...")
    assert subset_dict['terminals'][-1] == 1.0, "Subset's final terminal flag must be 1.0"
    assert subset_dict['valids'][-1] == 0.0, "Subset's final valid flag must be 0.0"
    subset_orig_terminals = 1.0 - subset_dict['valids']
    num_subset_trajs = int(np.sum(subset_orig_terminals))
    num_subset_double_terminals = int(np.sum(subset_dict['terminals'] > 0))
    print(f"  - Subset trajectories (inferred from 'valids'): {num_subset_trajs}")
    print(f"  - Subset non-zero 'terminals' entries: {num_subset_double_terminals}")
    assert num_subset_double_terminals > num_subset_trajs, "Subset does not have the correct compact 'terminals' structure."
    print("✅ Subset structure appears correct.")

    # --- START MODIFICATION: Add new visualization step ---
    # 5. Visualize the data structure to manually confirm correctness
    print("\n[STEP 5/5] Visualizing subset data structure...")
    
    # Show the very beginning of the dataset
    visualize_compact_data(subset_dict, 0, 20, title="Start of Subset")

    # Show the data around the end of the first trajectory
    first_traj_end_idx = np.where(subset_orig_terminals > 0)[0][0]
    start_viz_idx = max(0, first_traj_end_idx - 8)
    end_viz_idx = first_traj_end_idx + 8
    visualize_compact_data(subset_dict, start_viz_idx, end_viz_idx, title="Boundary Between Trajectory 1 and 2")
    
    print("✅ Manual verification step complete. Check the output above.")
    # --- END MODIFICATION ---

    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE ✅")
    print("="*80)

def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(entity='lesterpjy-university-of-amsterdam', project='project-ai', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent
    if FLAGS.debug_data_and_exit:
        debug_subsetting_and_sampling(config)
        return # Exit

    env, full_train_dataset, full_val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])
    DATA_SUBSET_SEED = 42 # Use a fixed seed for reproducible subsets

    # Create the value learning dataset dictionary
    if FLAGS.value_data_transitions is not None:
        value_train_dict = create_trajectory_subset(
            full_train_dataset, FLAGS.value_data_transitions, DATA_SUBSET_SEED
        )
    else:
        value_train_dict = full_train_dataset

    # Create the policy learning dataset dictionary
    if FLAGS.policy_data_transitions is not None:
        policy_train_dict = create_trajectory_subset(
            full_train_dataset, FLAGS.policy_data_transitions, DATA_SUBSET_SEED
        )
    else:
        policy_train_dict = full_train_dataset

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]

    value_dataset = dataset_class(Dataset.create(**value_train_dict), config)
    policy_dataset = dataset_class(Dataset.create(**policy_train_dict), config)

    if full_val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**full_val_dataset), config)
    else:
        val_dataset = None

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = full_train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        value_batch = value_dataset.sample(config['batch_size'])
        policy_batch = policy_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(value_batch, policy_batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                # For validation loss, we can sample a hybrid batch as well
                val_value_batch = val_dataset.sample(config['batch_size'])
                val_policy_batch = val_dataset.sample(config['batch_size']) # sample again for simplicity
                _, val_info = agent.total_loss(val_value_batch, val_policy_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i == 1 or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
            for task_id in tqdm.trange(1, num_tasks + 1):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)
            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
