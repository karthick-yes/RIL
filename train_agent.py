
import gymnasium as gym # Keep this import
import numpy as np # Keep this import
from datetime import datetime, timedelta
import os
# traceback is no longer needed, remove for clean code
# import traceback
from typing import Dict, List, Optional, Tuple, Any # Keep for clarity

# --- Import classes/functions from your specific files ---

# Import the Task class
from task import Task # Assuming Task class is defined in task.py

# Import the environment
from environment import TaskSchedulingEnv # Assuming TaskSchedulingEnv is defined in environment.py

# Import task generation and save/load functions
# Assuming these are all defined in generate_tasks.py
from generate_tasks import (
    generate_tasks_procedurally,
    save_task_registry,
    load_task_registry
)

# Import Stable Baselines3 components
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy # Optional: For evaluating the trained agent

# --- Main Training Script Logic ---
if __name__ == "__main__":
    # --- Configuration ---
    # Parameters for Task Generation (if regeneration is needed)
    NUM_TASKS_FOR_TRAINING = 200 # <--- Set the number of tasks to generate if no file is found

    # File paths
    TASK_REGISTRY_FILE = "generated_task_registry.pkl" # File path for the task registry
    MODEL_SAVE_PATH = "./task_scheduling_ppo_model" # Path to save/load the trained model

    # Define the environment horizon. Tasks should ideally have been generated
    # with deadlines relevant to this horizon. Ensure this matches the horizon
    # used when generating tasks in generate_tasks.py
    ENV_START_TIME = datetime(2025, 5, 1, 8, 0, 0)
    ENV_END_TIME = datetime(2025, 5, 15, 18, 0, 0) # Example: 2-week horizon

    # Environment parameters (must match TaskSchedulingEnv __init__)
    SLOT_DURATION_HOURS = 1
    MAX_TASKS_IN_STATE = 50 # Must match the value used in env init

    # Training parameters
    TOTAL_TIMESTEPS = 100000 # <--- Adjust this for more training!

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
            deadline_margin_hours=(24.0, 7*24.0),
            max_dependencies=5,
            seed=42 # Use a seed for reproducible generation
            # Add other generator args if needed
        )
        # Save the newly generated tasks using the function from generate_tasks.py
        save_task_registry(TASK_REGISTRY_FILE, Task.registry)
    else:
        print(f"Loaded {len(Task.registry)} tasks from {TASK_REGISTRY_FILE}.")

    # --- Crucial Check: Ensure Task.registry is populated ---
    if not Task.registry:
        print("Error: Task registry is empty after loading or generation. Cannot create environment.")
        exit() # Exit if no tasks are available

    # --- 2. Create the Environment ---
    # Ensure Task.registry is populated BEFORE creating the environment instance.
    # make_vec_env is a helper for Stable Baselines3, creates vectorized envs (even if num_envs=1)
    num_envs = 1 # Start with 1 environment for simplicity

    print("\nCreating Task Scheduling Environment...")
    # Pass the environment class and its required arguments using env_kwargs
    # The environment's __init__ will read the already populated Task.registry class variable.
    env = make_vec_env(TaskSchedulingEnv, env_kwargs={
        'task_registry': Task.registry, # Pass the dictionary of tasks (Task.registry is already populated globally)
        'start_time': ENV_START_TIME,
        'end_time': ENV_END_TIME,
        'slot_duration_hours': SLOT_DURATION_HOURS,
        'max_tasks_in_state': MAX_TASKS_IN_STATE,
        # Add other TaskSchedulingEnv __init__ args here if you have them
    }, n_envs=num_envs, seed=42) # Use a seed for reproducibility of environment dynamics


    # --- 3. Define and Create the Agent Model ---
    # PPO (Proximal Policy Optimization) is a good default algorithm
    # "MlpPolicy" uses a Multi-Layer Perceptron neural network for the policy
    print("\nCreating PPO agent model...")
    # Check if a model already exists and load it to continue training
    # Stable Baselines3 saves models as .zip files
    if os.path.exists(MODEL_SAVE_PATH + ".zip"):
        print(f"Loading existing model from {MODEL_SAVE_PATH}")
        try:
            # Load the model, passing the environment
            # This is needed if you want to continue training or evaluate the model.
            model = PPO.load(MODEL_SAVE_PATH, env=env, verbose=1)
            print("Model loaded successfully. Ready for training or inference.")
        except Exception as e:
            print(f"Failed to load model: {e}. Creating a new model.")
            model = PPO("MlpPolicy", env, verbose=1) # Create a new model if loading fails
    else:
        print("No existing model found. Creating a new model.")
        model = PPO("MlpPolicy", env, verbose=1) # verbose=1 prints training progress to console


    # --- 4. Train the Agent ---
    # This is where the agent learns by interacting with the environment
    # Increase TOTAL_TIMESTEPS significantly for better performance on complex task sets.
    print(f"\nStarting agent training for {TOTAL_TIMESTEPS} additional timesteps...")

    # Use callbacks for more advanced training control, saving checkpoints, etc.
    # from stable_baselines3.common.callbacks import CheckpointCallback
    # checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./train_checkpoints/',
    #                                          name_prefix='scheduling_model')
    # model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

    model.learn(total_timesteps=TOTAL_TIMESTEPS) # Simple learn call


    print("Training finished.")


    # --- 5. Save the Trained Model ---
    # Saves the learned policy and value function
    model.save(MODEL_SAVE_PATH)
    print(f"Trained model saved to {MODEL_SAVE_PATH}")


    # --- 6. Run the Trained Agent (Inference/Deployment) ---
    # Use the trained model to generate a schedule for one episode
    print("\nRunning inference with the trained agent...")

    # IMPORTANT: Reset the environment to start a fresh episode for inference
    # For vectorized envs, reset returns a tuple of observations/infos
    # For num_envs=1, it returns a tuple (array, dict)
    # The reset() also re-populates Task.registry from _initial_task_states, clearing schedule data.
    # --- WORKAROUND for ValueError: not enough values to unpack (from reset) ---
    reset_output = env.reset() # Call reset and capture the output

    if isinstance(reset_output, tuple) and len(reset_output) == 2:
        # Standard case: Returns (observation, info)
        obs, info = reset_output
        print("Inference reset returned 2 items (standard behavior).")
    elif isinstance(reset_output, np.ndarray) and reset_output.shape[0] == num_envs:
         # Possible non-standard case: Returns just the observation batch
         obs = reset_output
         # Create a dummy info list matching the expected structure for vectorized envs
         info = [{} for _ in range(num_envs)]
         print("Inference reset returned 1 item (observation batch). Using reset workaround.")
    else:
         # Unexpected case: Handle if it's neither a 2-tuple nor a single observation batch
         print(f"Inference reset returned unexpected type/shape: {type(reset_output)}, length {len(reset_output) if hasattr(reset_output, '__len__') else 'N/A'}.")
         print("Cannot proceed with inference.")
         exit() # Exit gracefully if reset output is unusable
    # --- END RESET WORKAROUND ---


    episode_reward = 0
    episode_length = 0
    # terminated and truncated should be arrays for vectorized envs
    terminated = np.array([False] * num_envs)
    truncated = np.array([False] * num_envs)


    # Run the episode step by step until it's done (either by horizon end or task completion)
    # Use env.get_attr('num_time_slots')[0] to get the num_time_slots from the underlying env instance
    max_steps_in_episode = env.get_attr('num_time_slots')[0]
    print(f"Running inference for max {max_steps_in_episode} steps...")

    # Create a list to store the planned schedule [slot_index, task_id or None] for easy parsing
    # Initialize with None for all possible slots within the horizon
    planned_schedule_raw = [None] * max_steps_in_episode


    # Loop through each possible time slot (step) within the horizon
    # Stop if any environment is done (for num_envs=1, this is just the one env)
    while not (terminated[0] or truncated[0]) and episode_length < max_steps_in_episode:

        # Get the action from the trained model based on the current observation
        # deterministic=True ensures consistent output for a given state.
        # model.predict expects the observation batch format returned by env.reset()
        action, _states = model.predict(obs, deterministic=True)

        # Stable Baselines3 vectorized envs return/expect arrays, even for num_envs=1.
        # action is typically a numpy array like np.array([action_int]) for num_envs=1.
        action_to_take = action[0]

        # --- WORKAROUND for ValueError: not enough values to unpack (from step) ---
        # Call step and capture the output
        step_output = env.step([action_to_take]) # env.step expects a list of actions

        # Check the length of the step output
        if isinstance(step_output, tuple) and len(step_output) == 5:
            # Standard case: Returns (observation, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = step_output
            # print(f"Step {episode_length}: Returned 5 items (standard behavior).") # Optional debug print
        elif isinstance(step_output, tuple) and len(step_output) == 4:
            # Non-standard case (seen error): Returns 4 items.
            # Assume the standard order is maintained, and the missing item is 'info'.
            # If your environment/wrappers have a weird order, this might need adjustment.
            obs, reward, terminated, truncated = step_output
            info = [{} for _ in range(num_envs)] # Create dummy info list for consistency
            # print(f"Step {episode_length}: Returned 4 items. Assuming 'info' is missing. Using step workaround.") # Optional debug print
        else:
             # Unexpected case: Handle if it's neither a 5-tuple nor a 4-tuple
             print(f"\nStep {episode_length}: Returned unexpected type/shape: {type(step_output)}, length {len(step_output) if hasattr(step_output, '__len__') else 'N/A'}.")
             print("Cannot proceed with inference.")
             # Print the content for inspection
             print(f"Content of unexpected step return: {step_output}")
             # Set termination flags to true to exit the loop gracefully
             terminated[0] = True
             truncated[0] = True
             break # Exit loop immediately

        # --- END STEP WORKAROUND ---


        # Sum episode reward (reward is also an array for vec env)
        episode_reward += reward[0]
        episode_length += 1 # Increment episode length AFTER the step


        # Record the scheduled task for the slot *just processed*
        # The index of the slot just processed is current_time_slot_index - 1 in the environment instance
        # Make sure current_time_slot_index is accessible and correct after the step
        # Access this attribute AFTER the step returns
        try:
            current_slot_index_processed = env.get_attr('current_time_slot_index')[0] - 1

            # The scheduled_slots dict within the environment instance is updated *during* the step.
            # We access the result after the step is complete.
            scheduled_slots_this_env = env.get_attr('scheduled_slots')[0]
            task_id_scheduled_in_this_slot = scheduled_slots_this_env.get(current_slot_index_processed)

            # Print information about the action taken in this slot
            if task_id_scheduled_in_this_slot is not None:
                 # Use the task_id from the dict for the planned schedule raw list
                 # Ensure current_slot_index_processed is a valid index for planned_schedule_raw
                 if 0 <= current_slot_index_processed < len(planned_schedule_raw):
                      planned_schedule_raw[current_slot_index_processed] = task_id_scheduled_in_this_slot
                 else:
                      print(f"Warning: Calculated slot index {current_slot_index_processed} out of bounds for planned_schedule_raw (length {len(planned_schedule_raw)}). Skipping schedule recording for this slot.")

                 task_scheduled = Task.get_task_by_id(task_id_scheduled_in_this_slot) # Get task details from registry
                 task_name = task_scheduled.name if task_scheduled else "Unknown"
                 print(f"Time Step: {episode_length -1} (Slot {current_slot_index_processed}): Scheduled Task {task_id_scheduled_in_this_slot} ('{task_name}')")
            else:
                 # Action was 0 (Wait) or an invalid schedule attempt that didn't result in scheduling
                 action_name = "Wait" if action_to_take == 0 else f"Attempted Schedule Action {action_to_take} (Invalid?)"
                 print(f"Time Step: {episode_length -1} (Slot {current_slot_index_processed}): {action_name}")

        except Exception as e:
             print(f"Error accessing env attributes or recording schedule after step {episode_length -1}: {e}")
             # Decide how to handle this error - maybe terminate the episode?
             # For now, just print the error and continue if possible.


        # env.render(mode='human') # Optional: uncomment to see the environment step-by-step during inference

        # Check termination conditions (apply to the *next* step)
        # terminated and truncated are arrays for vectorized envs, check the first element [0]
        # The loop condition also checks these, but printing the outcome here is useful
        if terminated[0] or truncated[0]:
            print(f"\nEpisode finished after {episode_length} steps.") # episode_length is step_count + 1
            print(f"Total episode reward: {episode_reward:.2f}")
            # Access additional info from the info dictionary of the first environment (index 0)
            # Note: info itself is an array or list of dictionaries for vectorized envs
            # If the workaround created a dummy info list, info[0] will be the dummy dict {}
            if info and isinstance(info, list) and len(info) > 0: # Check if info is a non-empty list
                if 'episode_outcome' in info[0]:
                     print(f"Episode outcome: {info[0]['episode_outcome']}")
                if 'num_missed_deadlines' in info[0]:
                     print(f"Missed deadlines: {info[0]['num_missed_deadlines']}")
                # Add checks for other info keys you might have
            else:
                print("Info dictionary is not available or is empty after step.")

            # The final complete schedule dictionary is stored in the env instance(s)
            # Access it after the loop terminates
            # Make sure env.get_attr is still working correctly
            try:
                 final_scheduled_slots_dict = env.get_attr('scheduled_slots')[0] # Dictionary mapping slot index -> Task ID
                 print("\n--- Final Generated Schedule (Slot Index -> Task ID) ---")
                 # Print sorted schedule for readability
                 if final_scheduled_slots_dict:
                     for slot_idx in sorted(final_scheduled_slots_dict.keys()):
                          task_id = final_scheduled_slots_dict[slot_idx]
                          task = Task.get_task_by_id(task_id) # Get task details from registry
                          task_name = task.name if task else "Unknown"
                          # Get the actual datetime for the slot from the environment's time_slots list
                          slot_datetime = env.get_attr('time_slots')[0][slot_idx] # time_slots is also an attribute of the env instance

                          print(f"  Slot {slot_idx} ({slot_datetime.strftime('%Y-%m-%d %H:%M')}): Task {task_id} ('{task_name}')")
                 else:
                     print("  No tasks were scheduled in this episode.")

            except Exception as e:
                 print(f"\nError accessing final scheduled slots from env: {e}")


            break # Stop the inference loop once the episode is done


    # --- 7. Parse the Final Schedule for Obsidian ---
    # The `final_scheduled_slots_dict` dictionary contains the output you need.
    # Ensure final_scheduled_slots_dict is defined even if loop didn't finish cleanly
    if 'final_scheduled_slots_dict' not in locals():
        try:
            # Attempt to get it one last time if loop broke early before termination check
            final_scheduled_slots_dict = env.get_attr('scheduled_slots')[0]
        except Exception:
            final_scheduled_slots_dict = {} # Default to empty if access fails

    print("\n--- Generated Output for Obsidian ---")
    obsidian_output = "## Generated Task Schedule\n\n"
    if final_scheduled_slots_dict:
        # Sort by slot index to get chronological order
        sorted_scheduled_slots = sorted(final_scheduled_slots_dict.keys())
        for slot_index in sorted_scheduled_slots:
            task_id = final_scheduled_slots_dict[slot_index]
            # Need the actual datetime for the slot from the environment's time_slots list
            try:
                 slot_datetime = env.get_attr('time_slots')[0][slot_index]
            except Exception:
                 slot_datetime = ENV_START_TIME + timedelta(hours=slot_index * SLOT_DURATION_HOURS) # Fallback calculation

            # Need the task name or other details from the Task registry
            task = Task.get_task_by_id(task_id)
            task_name = task.name if task else "Unknown Task" # Fallback name
            # Format the output line
            obsidian_output += f"- {slot_datetime.strftime('%Y-%m-%d %H:%M')}: Task {task_id} ({task_name})\n"
    else:
        obsidian_output += "No tasks were scheduled in this episode.\n"

    print(obsidian_output)

    # You can save this output to a file if needed
    with open("schedule_output.md", "w") as f:
        f.write(obsidian_output)


    # --- 8. Close the environment(s) ---
    env.close()
    print("\nEnvironment closed.")