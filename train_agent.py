import gymnasium as gym
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple, Any
from task import Task
from environment import TaskSchedulingEnv
from generate_tasks import (
    generate_tasks_procedurally,
    save_task_registry,
    load_task_registry,
)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import (
    evaluate_policy,
)  # Optional: For evaluating the trained agent

from stable_baselines3.common.callbacks import CheckpointCallback

# --- Main Training Script Logic ---
if __name__ == "__main__":
    # --- Configuration ---
    # Parameters for Task Generation (if regeneration is needed)
    NUM_TASKS_FOR_TRAINING = (
        200  # <--- Set the number of tasks to generate if no file is found
    )

    # File paths
    TASK_REGISTRY_FILE = (
        "generated_task_registry.pkl"  # File path for the task registry
    )
    MODEL_SAVE_PATH = (
        "./task_scheduling_ppo_model"  # Path to save/load the trained model
    )
    CHECKPOINT_SAVE_PATH = "./train_checkpoints/"  # Path to save checkpoints

    # Define the environment horizon. Tasks should ideally have been generated
    # with deadlines relevant to this horizon. Ensure this matches the horizon
    # used when generating tasks in generate_tasks.py
    ENV_START_TIME = datetime(2025, 5, 1, 8, 0, 0)
    ENV_END_TIME = datetime(2025, 5, 15, 18, 0, 0)  # Example: 2-week horizon

    # Environment parameters (must match TaskSchedulingEnv __init__)
    SLOT_DURATION_HOURS = 1
    MAX_TASKS_IN_STATE = 50  # Must match the value used in env init

    # Training parameters
    # You can set this to 0 if you ONLY want to do inference when a model exists
    TOTAL_TIMESTEPS = 10000  # <--- Adjust this for more training!
    CHECKPOINT_FREQ = (
        10000  # Save checkpoint every this many timesteps # <-- Defined frequency
    )

    # --- 1. Load or Generate Tasks ---
    # Attempt to load existing tasks first using the function from generate_tasks.py
    # load_task_registry populates Task.registry directly upon success
    loaded_registry = load_task_registry(TASK_REGISTRY_FILE)

    if loaded_registry is None:
        # If loading failed or file not found, generate tasks using the function from generate_tasks.py
        print(f"Task registry not found at {TASK_REGISTRY_FILE}. Generating new tasks.")
        # Pass environment horizon times to the generator for realistic deadlines
        # generate_tasks_procedurally populates Task.registry directly
        generate_tasks_procedurally(
            num_tasks=NUM_TASKS_FOR_TRAINING,
            env_start_time=ENV_START_TIME,
            env_end_time=ENV_END_TIME,
            priority_range=(1, 10),
            duration_range_hours=(0.5, 20.0),
            deadline_margin_hours=(24.0, 7 * 24.0),
            max_dependencies=5,
            seed=42,  # Use a seed for reproducible generation
            # Add other generator args if needed
        )
        # Save the newly generated tasks using the function from generate_tasks.py
        save_task_registry(TASK_REGISTRY_FILE, Task.registry)
    else:
        print(f"Loaded {len(Task.registry)} tasks from {TASK_REGISTRY_FILE}.")

    # --- Crucial Check: Ensure Task.registry is populated ---
    if not Task.registry:
        print(
            "Error: Task registry is empty after loading or generation. Cannot create environment."
        )
        exit()  # Exit if no tasks are available

    # --- 2. Create the Environment for potential Training ---
    # We still need to create the vectorized environment if we might train.
    # If we only run inference with a loaded model, this 'env' object
    # might not be fully used beyond model loading, but it's needed for the PPO.load()
    # unless we refactor further.
    num_envs = 1  # Start with 1 environment for simplicity

    print("\nCreating Task Scheduling Environment (for potential training/loading)...")
    # Pass the environment class and its required arguments using env_kwargs
    env = make_vec_env(
        TaskSchedulingEnv,
        env_kwargs={
            "task_registry": Task.registry,
            "start_time": ENV_START_TIME,
            "end_time": ENV_END_TIME,
            "slot_duration_hours": SLOT_DURATION_HOURS,
            "max_tasks_in_state": MAX_TASKS_IN_STATE,
        },
        n_envs=num_envs,
        seed=42,
    )

    # --- 3. Define and Create the Agent Model ---
    print("\nDefining/Loading PPO agent model...")

    # Flag to track if we are performing training or just inference
    is_training_run = True  # Assume training by default

    # Check if a model already exists and load it
    if os.path.exists(MODEL_SAVE_PATH + ".zip"):
        print(f"Loading existing model from {MODEL_SAVE_PATH}")
        try:
            # Load the model, passing the vectorized environment 'env' as it was used during training
            model = PPO.load(MODEL_SAVE_PATH, env=env, verbose=1)
            print("Model loaded successfully.")
            is_training_run = False  # Mark as inference-only run if model loaded
        except Exception as e:
            print(f"Failed to load model: {e}. Creating a new model for training.")
            model = PPO("MlpPolicy", env, verbose=1)
            is_training_run = True
    else:
        print("No existing model found. Creating a new model for training.")
        model = PPO("MlpPolicy", env, verbose=1)
        is_training_run = True

    # --- Conditional Training ---
    if is_training_run:
        # --- 4. Train the Agent ---
        print(f"\nStarting agent training for {TOTAL_TIMESTEPS} timesteps...")

        # Create Checkpoint Callback
        checkpoint_callback = CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,  # Save every CHECKPOINT_FREQ timesteps
            save_path=CHECKPOINT_SAVE_PATH,  # Save to the specified path
            name_prefix="scheduling_model",  # Prefix for checkpoint file names
        )
        print(
            f"Checkpoints will be saved every {CHECKPOINT_FREQ} timesteps to {CHECKPOINT_SAVE_PATH}"
        )

        # Start training with the callback
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,  # <-- Pass the callback here
            reset_num_timesteps=True,  # Resets the number of timesteps processed for the callback
        )

        print("Training finished.")

        # --- 5. Save the Trained Model (after training) ---
        # The checkpoint callback saves periodically, but saving the final model is also good practice
        model.save(MODEL_SAVE_PATH)
        print(f"Trained model saved to {MODEL_SAVE_PATH}")

        # If training happened, close the training environment
        env.close()
        print("Training environment closed.")
        # The inference below will use a new raw environment instance

    else:  # Not a training run (model was loaded successfully)
        print("Skipping training step as a model was loaded.")
        # If we skipped training, the 'env' object created by make_vec_env
        # is technically not used anymore in the inference section below.
        # Close it here.
        env.close()
        print("Environment used for loading closed.")

    # --- 6. Run the Trained Agent (Inference/Deployment) on RAW Environment ---
    # This section runs either after training (with the just-trained model)
    # or directly after loading (with the loaded model).
    # We are using a raw environment instance here to completely bypass the make_vec_env wrapping
    # and test if the core environment can run an episode correctly.

    print("\nRunning inference with the trained agent on RAW environment...")

    # Create a raw environment instance directly for inference
    inference_env_params = {
        "task_registry": Task.registry,
        "start_time": ENV_START_TIME,
        "end_time": ENV_END_TIME,
        "slot_duration_hours": SLOT_DURATION_HOURS,
        "max_tasks_in_state": MAX_TASKS_IN_STATE,
    }
    raw_inference_env = TaskSchedulingEnv(**inference_env_params)

    # IMPORTANT: Reset the raw environment
    # raw env reset returns (observation, info) tuple
    print("\n--- Resetting RAW Inference Environment ---")
    try:
        obs, info = raw_inference_env.reset(
            seed=42, options={}
        )  # Pass seed and options to the raw env reset
        print("RAW Inference Environment reset successful.")
    except Exception as e:
        print(f"Error resetting RAW Inference Environment: {e}")
        raw_inference_env.close()
        exit()

    episode_reward = 0
    episode_length = 0
    terminated = False  # Initialize as single booleans for raw env
    truncated = False  # Initialize as single booleans for raw env

    # Get the max steps from the raw environment instance
    max_steps_in_episode = raw_inference_env.num_time_slots
    print(
        f"Running inference for max {max_steps_in_episode} steps on RAW environment..."
    )

    # Create a list to store the planned schedule [slot_index, task_id or None] for easy parsing
    # Initialize with None for all possible slots within the horizon
    planned_schedule_raw = (
        [None] * max_steps_in_episode if max_steps_in_episode > 0 else []
    )

    # Loop through each possible time slot (step) within the horizon
    # Stop if the environment is done or max steps are reached
    while not (terminated or truncated) and episode_length < max_steps_in_episode:
        # Get the action from the trained model based on the current observation
        # model.predict expects a batch, so wrap the single observation in a list/array
        # Ensure obs is a numpy array before wrapping if it's not already
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        action, _states = model.predict(
            obs[np.newaxis, ...], deterministic=True
        )  # Use np.newaxis to add batch dimension

        # model.predict returns a batch of actions, extract the single action for this env
        # action should be a numpy array like np.array([action_int])
        if isinstance(action, np.ndarray) and action.shape[0] == 1:
            action_to_take = action[0]  # Extract action for this single environment
        elif isinstance(action, list) and len(action) == 1:
            action_to_take = action[0]
        else:
            print(
                f"Error: Model predict returned action batch of unexpected type/shape {type(action)}/{getattr(action, 'shape', 'N/A')}. Expected numpy array of shape (1,...)."
            )
            # Handle this as a critical error - cannot proceed
            terminated = True  # Force exit loop
            truncated = True
            break  # Exit loop immediately

        # Step the RAW environment
        # raw_inference_env.step expects a single integer action
        # The step method now returns single values for raw env
        obs, reward, terminated, truncated, info = raw_inference_env.step(
            action_to_take
        )

        # Sum episode reward (reward is a single float for raw env)
        episode_reward += reward

        # Record the scheduled task for the slot *just processed*
        # The index of the slot just processed is current_time_slot_index - 1 in the environment instance
        # Make sure current_time_slot_index is accessible and correct after the step
        # Access this attribute from the raw env instance
        try:
            current_slot_index_processed = raw_inference_env.current_time_slot_index - 1

            # The scheduled_slots dict within the environment instance is updated *during* the step.
            # We access the result after the step is complete.
            scheduled_slots_this_env = raw_inference_env.scheduled_slots
            task_id_scheduled_in_this_slot = scheduled_slots_this_env.get(
                current_slot_index_processed
            )

            # Print information about the action taken in this slot
            # episode_length is the 0-based index of the step being processed
            current_step_index = episode_length

            if task_id_scheduled_in_this_slot is not None:
                # Use the task_id from the dict for the planned schedule raw list
                # Ensure current_slot_index_processed is a valid index for planned_schedule_raw
                if 0 <= current_slot_index_processed < len(planned_schedule_raw):
                    planned_schedule_raw[current_slot_index_processed] = (
                        task_id_scheduled_in_this_slot
                    )
                else:
                    print(
                        f"Warning: Calculated slot index {current_slot_index_processed} out of bounds for planned_schedule_raw (length {len(planned_schedule_raw)}). Skipping schedule recording for this slot."
                    )

                task_scheduled = Task.get_task_by_id(
                    task_id_scheduled_in_this_slot
                )  # Get task details from registry
                task_name = task_scheduled.name if task_scheduled else "Unknown"
                # Print using the step index (0-based) and slot index
                print(
                    f"Time Step: {current_step_index} (Slot {current_slot_index_processed}): Scheduled Task {task_id_scheduled_in_this_slot} ('{task_name}')"
                )
            else:
                # Action was 0 (Wait) or an invalid schedule attempt that didn't result in scheduling
                action_name = (
                    "Wait"
                    if action_to_take == 0
                    else f"Attempted Schedule Action {action_to_take} (Invalid?)"
                )
                print(
                    f"Time Step: {current_step_index} (Slot {current_slot_index_processed}): {action_name}"
                )

        except Exception as e:
            print(
                f"Error accessing env attributes or recording schedule after step {episode_length}: {e}"
            )
            # Decide how to handle this error - maybe terminate the episode?
            # For now, just print the error and continue if possible.

        # Increment episode length at the end of the loop body
        episode_length += 1

        # raw_inference_env.render(mode='human') # Optional: uncomment to see the environment step-by-step during inference

        # The check for termination conditions leading to break is done by the while loop condition itself
        # However, printing the episode outcome here *immediately* when terminated/truncated becomes true is useful.
        if terminated or truncated:
            print(f"\nEpisode finished after {episode_length} steps.")
            print(f"Total episode reward: {episode_reward:.2f}")
            # Access additional info from the info dictionary
            if info:  # Check if info dictionary is not None/empty
                if "episode_outcome" in info:
                    print(f"Episode outcome: {info['episode_outcome']}")
                if "num_missed_deadlines" in info:
                    print(f"Missed deadlines: {info['num_missed_deadlines']}")
            else:
                print("Info dictionary is not available or is empty after step.")

    # --- 7. Parse the Final Schedule for Obsidian ---
    # This section runs AFTER the inference loop terminates.
    # The final complete schedule dictionary is stored in the env instance.
    # Access it AFTER the loop terminates.
    final_scheduled_slots_dict = {}  # Initialize for safety
    try:
        final_scheduled_slots_dict = (
            raw_inference_env.scheduled_slots
        )  # Dictionary mapping slot index -> Task ID
    except Exception as e:
        print(f"\nError accessing final scheduled slots from env after episode: {e}")

    print("\n--- Final Generated Schedule (Slot Index -> Task ID) ---")
    # Print sorted schedule for readability
    if final_scheduled_slots_dict:
        for slot_idx in sorted(final_scheduled_slots_dict.keys()):
            task_id = final_scheduled_slots_dict[slot_idx]
            task = Task.get_task_by_id(task_id)  # Get task details from registry
            task_name = task.name if task else "Unknown"
            # Get the actual datetime for the slot from the environment's time_slots list
            try:
                slot_datetime = raw_inference_env.time_slots[slot_idx]
            except Exception:
                # Fallback calculation if time_slots attribute access fails
                slot_datetime = ENV_START_TIME + timedelta(
                    hours=slot_idx * SLOT_DURATION_HOURS
                )

            print(
                f"  Slot {slot_idx} ({slot_datetime.strftime('%Y-%m-%d %H:%M')}): Task {task_id} ('{task_name}')"
            )
    else:
        print("  No tasks were scheduled in this episode.")

    print("\n--- Generated Output for Obsidian ---")
    obsidian_output = "## Generated Task Schedule\n\n"
    if final_scheduled_slots_dict:
        # Sort by slot index to get chronological order
        sorted_scheduled_slots = sorted(final_scheduled_slots_dict.keys())
        for slot_index in sorted_scheduled_slots:
            task_id = final_scheduled_slots_dict[slot_index]
            # Need the actual datetime for the slot from the environment's time_slots list
            try:
                slot_datetime = raw_inference_env.time_slots[slot_index]
            except Exception:
                # Fallback calculation if time_slots attribute access fails
                slot_datetime = ENV_START_TIME + timedelta(
                    hours=slot_index * SLOT_DURATION_HOURS
                )

            # Need the task name or other details from the Task registry
            task = Task.get_task_by_id(task_id)
            task_name = task.name if task else "Unknown Task"  # Fallback name
            # Format the output line
            obsidian_output += f"- {slot_datetime.strftime('%Y-%m-%d %H:%M')}: Task {task_id} ({task_name})\n"
    else:
        obsidian_output += "No tasks were scheduled in this episode.\n"

    print(obsidian_output)

    # You can save this output to a file if needed
    with open("schedule_output.md", "w") as f:
        f.write(obsidian_output)

    # --- 8. Close the environment(s) ---
    # Close the raw environment instance used for inference
    raw_inference_env.close()
    print("\nRAW Inference Environment closed.")
