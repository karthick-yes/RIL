import random
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
from typing import List, Dict, Optional, Tuple
from task import Task


def save_task_registry(file_path: str, registry: Dict[int, Task]):
    """Saves the Task.registry to a file using pickle."""
    try:
        with open(file_path, "wb") as f:
            pickle.dump(registry, f)
        print(f"Task registry successfully saved to {file_path}")
    except IOError as e:
        print(f"Error saving task registry to {file_path}: {e}")


def load_task_registry(file_path: str) -> Optional[Dict[int, Task]]:
    """Loads the Task.registry from a file using pickle."""
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "rb") as f:
            registry = pickle.load(f)
        print(f"Task registry successfully loaded from {file_path}")
        # After loading, update the class registry and next_id_counter
        Task.registry.clear()  # Clear existing before loading
        Task.registry.update(registry)  # Add loaded tasks
        if registry:
            # Find the maximum existing ID and set next_id_counter
            max_id = max(registry.keys())
            Task.next_id_counter = max_id + 1
        else:
            Task.next_id_counter = 1  # Reset if loaded registry was empty

        print(f"Task.registry updated. Next ID counter set to {Task.next_id_counter}")
        return registry
    except (IOError, pickle.UnpicklingError, EOFError) as e:
        print(f"Error loading task registry from {file_path}: {e}")
        return None
    except Exception as e:  # Catch any other unexpected errors during loading
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return None


# --- Procedural Task Generation Logic ---


def generate_tasks_procedurally(
    num_tasks: int,
    env_start_time: datetime,
    env_end_time: datetime,
    priority_range: Tuple[int, int] = (1, 10),  # Min, Max priority (inclusive)
    duration_range_hours: Tuple[float, float] = (
        0.5,
        20.0,
    ),  # Min, Max duration (hours)
    deadline_margin_hours: Tuple[float, float] = (
        24.0,
        7 * 24.0,
    ),  # Min, Max hours margin BEFORE env_end_time
    max_dependencies: int = 5,  # Max number of dependencies per task (selected from previous tasks)
    seed: Optional[int] = None,  # Optional random seed for reproducibility
) -> Dict[int, Task]:
    """
    Generates a set of random tasks suitable for the scheduling environment.
    Populates the Task.registry class variable directly.

    Args:
        num_tasks: The number of tasks to generate.
        env_start_time: The start time of the scheduling horizon. Used for setting deadlines.
        env_end_time: The end time of the scheduling horizon. Used for setting deadlines.
        priority_range: Tuple (min_priority, max_priority).
        duration_range_hours: Tuple (min_duration_hours, max_duration_hours).
        deadline_margin_hours: Tuple (min_margin_hours, max_margin_hours BEFORE end time).
                                Deadlines will be set between (end_time - max_margin) and (end_time - min_margin).
        max_dependencies: Maximum number of dependencies a task can have (selected from previous tasks).
        seed: Optional random seed.

    Returns:
        Dict[int, Task]: The populated Task.registry.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f"Using random seed: {seed}")

    print(f"--- Generating {num_tasks} Tasks ---")
    Task.registry.clear()  # Start with an empty registry for fresh generation
    Task.next_id_counter = 1  # Reset ID counter

    total_horizon_seconds = (env_end_time - env_start_time).total_seconds()
    if total_horizon_seconds <= 0:
        raise ValueError(
            "Environment end time must be strictly after start time for task generation."
        )

    min_possible_deadline_offset_seconds = (
        env_end_time - env_start_time
    ).total_seconds() - deadline_margin_hours[1] * 3600.0
    max_possible_deadline_offset_seconds = (
        env_end_time - env_start_time
    ).total_seconds() - deadline_margin_hours[0] * 3600.0

    # Ensure the deadline range is valid (min_margin < max_margin and both result in deadlines after start_time)
    # A simple way is to just clamp the final generated deadline
    min_valid_deadline = (
        env_start_time  # Deadlines should not be before the start of the horizon
    )

    generated_tasks_list = []  # Keep a list if needed, main storage is Task.registry

    for i in range(num_tasks):
        task_id = Task.next_id_counter  # Get the next available ID
        name = f"AutoTask_{task_id}"

        # Generate properties within specified ranges
        priority = random.randint(*priority_range)
        duration = random.uniform(*duration_range_hours)

        # Generate deadline: Set deadline relative to the end of the horizon
        # Pick a random time between (end_time - max_margin) and (end_time - min_margin)
        # This makes tasks more likely to be relevant/urgent towards the end of the horizon
        # Ensure the range is valid (lower bound <= upper bound)
        lower_bound_seconds_from_start = max(
            0.0, total_horizon_seconds - deadline_margin_hours[1] * 3600.0
        )
        upper_bound_seconds_from_start = max(
            lower_bound_seconds_from_start,
            total_horizon_seconds - deadline_margin_hours[0] * 3600.0,
        )  # Ensure upper is >= lower

        # Generate random offset from env_start_time within the valid range
        deadline_offset_seconds = random.uniform(
            lower_bound_seconds_from_start, upper_bound_seconds_from_start
        )
        deadline_datetime = env_start_time + timedelta(seconds=deadline_offset_seconds)

        # Final clamp just to be safe (shouldn't be needed if ranges are set well)
        deadline_datetime = max(
            deadline_datetime, min_valid_deadline
        )  # Ensure >= env_start_time
        deadline_datetime = min(
            deadline_datetime, env_end_time + timedelta(seconds=1)
        )  # Ensure roughly <= env_end_time (allow slight overshoot)

        # Generate dependencies: Pick randomly from tasks already generated (IDs 1 to task_id-1)
        # Ensure dependencies are valid task IDs that already exist in the registry
        dependencies = []
        if i > 0:  # Tasks after the first one can depend on previous tasks
            possible_deps_ids = list(
                Task.registry.keys()
            )  # Get IDs of tasks already in the registry
            # Filter out any tasks that might not be valid dependencies (e.g., if some were removed, not applicable here)
            # Ensure we only pick from tasks with IDs less than the current task's ID
            valid_previous_task_ids = [
                tid for tid in possible_deps_ids if tid < task_id
            ]

            num_deps = random.randint(
                0, min(max_dependencies, len(valid_previous_task_ids))
            )  # Number of dependencies for this task
            if num_deps > 0:
                dependencies = random.sample(
                    valid_previous_task_ids, num_deps
                )  # Pick unique dependency IDs

        # Instantiate the Task. Task.__init__ adds it to Task.registry and increments next_id_counter.
        try:
            task = Task(
                name=name,
                ID=task_id,  # Pass the assigned ID
                priority=priority,
                deadline=deadline_datetime.isoformat(),  # Store as ISO format string as per Task.__init__ expects
                duration=duration,
                dependencies=dependencies,  # Pass the generated dependency IDs
                source_file="generated_script",  # Indicate origin
                # status and time_spent default to 0.0 in Task.__init__
            )
            # Note: Task.__init__ is responsible for adding 'task' to Task.registry
            generated_tasks_list.append(task)  # Keep a list if needed elsewhere

        except Exception as e:
            print(f"Error creating task {task_id}: {e}. Skipping this task.")
            # Need to decrement next_id_counter if task creation failed after getting ID
            # Or handle ID generation more carefully. Let's assume Task.__init__
            # is robust and ID was incremented only on success. If not, adjust Task class.
            pass

    print(f"Finished generating {len(Task.registry)} tasks and added to Task.registry.")
    return (
        Task.registry
    )  # Return the populated registry (which is Task.registry itself)


# --- Usage ---
if __name__ == "__main__":
    # --- Define Generation Parameters ---
    NUM_TASKS_TO_GENERATE = 200  # <--- Adjust this for more tasks!
    OUTPUT_FILE = "generated_task_registry.pkl"  # File to save the generated tasks

    # Define the environment horizon these tasks are intended for.
    # This is used to generate realistic deadlines.
    # Make sure these times are consistent with your TaskSchedulingEnv setup.
    ENV_START_TIME = datetime(2025, 5, 1, 8, 0, 0)
    ENV_END_TIME = datetime(2025, 5, 15, 18, 0, 0)  # Example: 2-week horizon

    # --- Generate the tasks ---
    # This function will clear Task.registry and populate it with new tasks
    generated_registry = generate_tasks_procedurally(
        num_tasks=NUM_TASKS_TO_GENERATE,
        env_start_time=ENV_START_TIME,
        env_end_time=ENV_END_TIME,
        priority_range=(1, 10),  # Tasks have priorities between 1 and 10
        duration_range_hours=(0.5, 20.0),  # Tasks take between 0.5 and 20 hours
        deadline_margin_hours=(
            24.0,
            7 * 24.0,
        ),  # Deadlines are between 1 day and 7 days *before* the env end time
        max_dependencies=5,  # Tasks have up to 5 dependencies on previous tasks
        seed=42,  # Use a seed for reproducible task sets
    )

    # --- Save the generated tasks ---
    # This uses the save function defined above (or imported)
    save_task_registry(OUTPUT_FILE, generated_registry)

    print(
        f"\nProcedurally generated {len(generated_registry)} tasks and saved to {OUTPUT_FILE}."
    )

    # # --- Optional: if in case we have to Verify loading ---
    # print(f"\nVerifying load from {OUTPUT_FILE}")
    # Task.registry.clear() # Clear registry before loading to simulate a fresh start
    # Task.next_id_counter = 1
    # loaded_registry = load_task_registry(OUTPUT_FILE)
    # if loaded_registry:
    #     print(f"Successfully loaded {len(loaded_registry)} tasks.")
    #     for task_id, task in list(loaded_registry.items())[:5]: # Print first 5
    #         print(f"  Loaded Task {task.ID}: Name='{task.name}', Priority={task.priority}, Deadline={task.deadline}, Duration={task.duration}, Deps={task.dependencies}")
