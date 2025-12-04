import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from task import Task
from dag_builder import build_task_dag
import gymnasium as gym


class TaskSchedulingEnv(gym.Env):
    """
    Reinforcement Learning Environment for Task Scheduling.
    Defines the state space, action space, reward function, and environment dynamics in step().
    """

    def __init__(
        self,
        task_registry: Dict[int, Task],
        start_time: datetime,
        end_time: datetime,
        slot_duration_hours: int = 1,
        max_tasks_in_state: int = 50,
    ):
        """
        Initializes the task scheduling environment.

        Args:
            task_registry: The dictionary of Task objects (Task.registry).
            start_time: The datetime object representing the start of the scheduling horizon.
            end_time: The datetime object representing the end of the scheduling horizon.
            slot_duration_hours: The duration of each time slot in hours.
            max_tasks_in_state: The maximum number of tasks whose state will be
                                included in the observation vector.
        """
        super().__init__()

        if not isinstance(task_registry, dict):
            raise TypeError("task_registry must be a dictionary.")
        if not all(isinstance(task, Task) for task in task_registry.values()):
            print("Warning: task_registry contains non-Task objects.")  # Or raise error

        self.task_registry = task_registry
        # Build the DAG from the registry. Note: The DAG will reflect the registry state
        # at the time the environment is initialized or reset with a new registry.
        self.dag = build_task_dag(self.task_registry)

        self.start_time = start_time
        self.end_time = end_time
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time.")

        self.slot_duration = timedelta(hours=slot_duration_hours)
        if self.slot_duration <= timedelta(0):
            raise ValueError("slot_duration_hours must be positive.")

        # Generate all possible time slots within the horizon
        self.time_slots = self._generate_time_slots(
            self.start_time, self.end_time, self.slot_duration
        )
        self.num_time_slots = len(self.time_slots)
        if self.num_time_slots == 0:
            print("Warning: Scheduling horizon is too short to create any time slots.")

        # --- Time Slot Availability during an episode ---
        # This will track which slots are scheduled with tasks during the current episode.
        # Maps time slot index to task ID scheduled in that slot.
        self.scheduled_slots: Dict[int, int] = {}
        # Tracks how much time was spent on a task in a specific slot during the current episode.
        # Maps (task_id, slot_index) to time spent.
        self.time_spent_in_slots: Dict[Tuple[int, int], float] = {}

        # --- Assumed UNAVAILABLE slots (Static Constraint defined at init) ---
        # we will define the indices of the slots that are never available during the scheduling.
        # based on our assumptions (e.g., lunch breaks, meetings).
        self.unavailable_slots_indices: Set[int] = set()

        # --- Populate unavailable_slots_indices based on your specific assumptions ---
        # Example Logic (replace with your actual logic):
        # Assuming a fixed start time (e.g., 8 am) and 1-hour slots.
        # This is a placeholder example.
        try:
            # Calculate the total number of slots per full 24-hour cycle
            slots_per_day = int(timedelta(hours=24) / self.slot_duration)
            if slots_per_day <= 0:
                # Handle cases where slot duration is longer than a day or invalid
                slots_per_day = self.num_time_slots  # Or raise an error

            # Example: Make 12pm-1pm and 6pm-7pm unavailable *every day* within the horizon
            unavail_hours_daily = [12, 18]  # Example: 12:00 (noon) and 18:00 (6 PM)

            # Calculate the index offset for these hours on a 'base' day starting at self.start_time.hour
            base_hour_offset = self.start_time.hour
            unavail_slot_offsets_from_start_hour = [
                (h - base_hour_offset + 24) % 24 for h in unavail_hours_daily
            ]  # Ensure positive offset

            # Iterate through all slots and mark those corresponding to the unavailable times
            for i in range(self.num_time_slots):
                slot_datetime = self.time_slots[i]
                # Calculate the hour offset from the start time of the horizon
                hour_diff = (slot_datetime - self.start_time).total_seconds() / 3600.0
                # Calculate the effective hour within a 24-hour cycle relative to the start hour
                effective_hour_offset = (
                    int(round(hour_diff)) % slots_per_day
                )  # Use round for potential floating point

                if effective_hour_offset in unavail_slot_offsets_from_start_hour:
                    # Check if the actual hour of the slot matches the unavailable hour
                    slot_hour = slot_datetime.hour
                    if slot_hour in unavail_hours_daily:
                        self.unavailable_slots_indices.add(i)
                    # else:
                    # print(f"Calculated offset {effective_hour_offset} for slot {i} (hour {slot_hour}) does not match unavailable hour pattern.")

        except Exception as e:
            print(f"Error calculating assumed unavailable slots: {e}")
            # will Decide how to handle this error - maybe proceed with no unavailable slots?

        print(
            f"Initialized with {len(self.unavailable_slots_indices)} assumed unavailable time slot indices."
        )
        # -----------------------------------------------------------

        # The maximum number of tasks the state can represent.
        self.max_tasks_in_state = max_tasks_in_state
        if self.max_tasks_in_state <= 0:
            raise ValueError("max_tasks_in_state must be positive.")

        # --- Action Space Definition ---
        # The agent needs to decide what to do in the CURRENT time slot.
        # Action 0: Do nothing (wait).
        # Actions 1 to max_tasks_in_state: Attempt to schedule the task at index i-1
        # (in the sorted list of tasks considered in the state) in the CURRENT time slot.
        self.action_space_size = 1 + self.max_tasks_in_state
        print(
            f"Action Space Size: {self.action_space_size} (1 for wait + {self.max_tasks_in_state} for scheduling attempts)"
        )

        # Action space for GYMNASIUM
        self.action_space = gym.spaces.Discrete(self.action_space_size)
        print(f"Gymnasium Action Space: {self.action_space}")

        # Observation Space for GYMNASIUM
        # Defines the structure, shape, and bounds of the state vector returned by get_state()
        task_feature_size = self._get_task_state_size()  # Which is currently 8
        observation_space_size = (
            1 + self.num_time_slots + (self.max_tasks_in_state * task_feature_size)
        )

        # Define bounds for the observation space features
        # low: typically 0 for most normalized/binary features
        low = np.zeros(observation_space_size, dtype=np.float32)

        # high: depends on the feature scaling
        high = np.zeros(observation_space_size, dtype=np.float32)

        # 1. Normalized current slot: [0.0, 1.0]
        high[0] = 1.0

        # 2. Time slot status vector: [0.0, 2.0] (values are 0, 1, or 2)
        # This part of the vector spans indices from 1 up to 1 + num_time_slots - 1 = num_time_slots
        high[1 : 1 + self.num_time_slots] = 2.0

        # 3. Task states flat: max_tasks_in_state * task_feature_size
        # This part spans indices from 1 + num_time_slots onwards
        # We use a safe high bound for normalized remaining duration just in case it exceeds 1.0.
        high_task_features = np.array(
            [
                1.0,  # 1. Status (0 to 1)
                100.0,  # 2. Remaining Duration (normalized - using a large safe number like 100)
                1.0,  # 3. Time until Deadline (normalized 0 to 1)
                1.0,  # 4. Base Priority (normalized 0 to 1)
                1.0,  # 5. Dependencies Met (Binary 0 or 1)
                1.0,  # 6. Is scheduled in current slot (Binary 0 or 1)
                1.0,  # 7. Has deadline passed (Binary 0 or 1)
                1.0,  # 8. Dynamic Priority Score (0 to 1)
            ],
            dtype=np.float32,
        )

        # Tile the high bounds for each task feature across all max_tasks_in_state slots
        high[1 + self.num_time_slots :] = np.tile(
            high_task_features, self.max_tasks_in_state
        )

        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=(observation_space_size,), dtype=np.float32
        )
        print(f"Gymnasium Observation Space Shape: {self.observation_space.shape}")

        # Keep track of the current state - will be updated by step()
        self.current_time_slot_index = 0

        # Store the initial task states or a copy of the initial registry
        # so we can reset the environment accurately for new episodes.
        # A deep copy might be necessary if tasks are mutable.
        self._initial_task_states = {
            task_id: (task.status, task.time_spent)
            for task_id, task in self.task_registry.items()
        }
        # Optionally store initial dependencies if they could change (less likely in this project)

        # Reset the environment to a starting state
        self.reset()

    def _generate_time_slots(
        self, start: datetime, end: datetime, duration: timedelta
    ) -> List[datetime]:
        """Generates a list of start times for each time slot."""
        slots = []
        current = start
        while current < end:
            slots.append(current)
            current += duration
            # Add a small epsilon to avoid floating point issues near the end time
            if current + timedelta(seconds=0.001) > end and current < end:
                # Ensure the last slot doesn't go beyond the end time
                if (
                    start + timedelta(seconds=(end - start).total_seconds()) < end
                ):  # Add only if it's a valid slot start before end
                    slots.append(
                        start + timedelta(seconds=(end - start).total_seconds())
                    )
                break  # Stop after adding the last potential slot
        return slots

    def _get_current_datetime(self) -> datetime:
        """Calculates the actual datetime for the current time slot."""
        if 0 <= self.current_time_slot_index < self.num_time_slots:
            return self.time_slots[self.current_time_slot_index]
        elif self.num_time_slots > 0:
            # If beyond the last slot, return the start time of the last slot + duration
            return self.time_slots[-1] + self.slot_duration
        else:
            # Should not happen if num_time_slots is calculated correctly based on horizon
            return self.end_time  # Or handle as an error state (e.g., raise Exception)

    def _get_task_state_representation(self, task_id: int) -> np.ndarray:
        """
        Generates a numerical representation for a single task's state.
        This vector will be part of the overall environment state.
        Ensure all features are scaled or normalized appropriately (e.g., 0 to 1).
        """
        task = Task.get_task_by_id(task_id)  # Get task from the class registry
        # If task is None, it might have been removed from the registry during an episode.
        # Return a zero vector, but this scenario needs careful handling in step().
        if task is None:
            # Return a default/invalid state representation (all zeros) if task not found
            return np.zeros(self._get_task_state_size(), dtype=np.float32)

        features = []

        # --- Task Features ---
        # 1. Status (already 0.0 to 1.0)
        features.append(task.status)

        # 2. Remaining Duration (normalize by maximum possible duration in the horizon or max task duration)
        # Let's normalize by a predefined max task duration
        max_task_duration = 20.0  # Example max duration, adjust based on your data
        remaining_duration = max(0.0, task.duration - task.time_spent)
        features.append(
            remaining_duration / max_task_duration if max_task_duration > 0 else 0.0
        )

        # 3. Time until Deadline (normalize by total scheduling horizon duration)
        total_horizon_seconds = (self.end_time - self.start_time).total_seconds()
        current_datetime = self._get_current_datetime()
        time_until_deadline_seconds = max(
            0.0, (task.deadline - current_datetime).total_seconds()
        )
        # Normalize between 0 (deadline is now or passed) and 1 (deadline is at the end of the horizon or later)
        normalized_time_until_deadline = (
            time_until_deadline_seconds / total_horizon_seconds
            if total_horizon_seconds > 0
            else 0.0
        )
        features.append(normalized_time_until_deadline)

        # 4. Base Priority (normalize by maximum possible priority)
        max_base_priority = 10  # Example max priority, adjust based on your data
        features.append(
            task.priority / max_base_priority if max_base_priority > 0 else 0.0
        )

        # 5. Dependencies Met Status (Binary: 1 if all dependencies are completed, 0 otherwise)
        dependencies_met = True
        # Check dependencies using the DAG or task registry
        for dep_id in task.dependencies:  # Assuming task.dependencies is up-to-date
            dep_task = Task.get_task_by_id(dep_id)
            # A dependency is met if the dependency task exists and is completed
            # Handle case where a dependency might not be in the registry (e.g., was removed)
            if dep_task is None or not dep_task.is_completed():
                dependencies_met = False
                break
        features.append(1.0 if dependencies_met else 0.0)

        # 6. Is this task currently scheduled in the CURRENT time slot? (Binary)
        # This checks if the task_id is in the scheduled_slots dictionary at the current index
        is_scheduled_in_current_slot = (
            1.0
            if self.scheduled_slots.get(self.current_time_slot_index) == task_id
            else 0.0
        )
        features.append(is_scheduled_in_current_slot)

        # 7. Has the task's deadline passed? (Binary)
        current_datetime = (
            self._get_current_datetime()
        )  # Ensure current_datetime is up to date
        deadline_passed = (
            1.0 if current_datetime > task.deadline and not task.is_completed() else 0.0
        )
        features.append(deadline_passed)

        # --- Feature 8: Dynamic Priority Score ---
        current_datetime = (
            self._get_current_datetime()
        )  # Needed for computePriorityScore

        # Calculate the total environment horizon duration in days as a scaling factor.
        # This value is constant for a given environment setup (start_time to end_time).
        total_env_horizon_seconds = (self.end_time - self.start_time).total_seconds()
        total_env_horizon_days_for_scaling = max(
            1.0, total_env_horizon_seconds / (24 * 3600.0)
        )

        dynamic_score = task.computePriorityScore(
            current_datetime=current_datetime,  # Pass current time (used internally by Task for clamped_days)
            max_horizon=total_env_horizon_days_for_scaling,
            max_priority=10,  # Passing configured max priority
            max_duration=20.0,  # Passing configured max duration
            max_deps=max(
                1, len(task.dependencies) + 5
            ),  # Pass a scale for max dependencies
        )
        features.append(dynamic_score)

        return np.array(features, dtype=np.float32)

    def _get_task_state_size(self) -> int:
        """Returns the number of features in the state representation for a single task."""
        # Updated to reflect the added dynamic priority score
        return 8  # status, remaining_duration, time_until_deadline, base_priority, dependencies_met, scheduled_in_current_slot, deadline_passed, dynamic_priority_score

    def get_state(self) -> np.ndarray:
        """
        Returns the current state of the environment as a fixed-size NumPy array.
        The state vector is composed of:
        [normalized_current_slot, time_slot_status_vector, task_states_flat]
        where:
        - normalized_current_slot: float (shape (1,))
        - time_slot_status_vector: np.ndarray of shape (num_time_slots,)
        - task_states_flat: np.ndarray of shape (max_tasks_in_state * _get_task_state_size,)

        The total state size is 1 + num_time_slots + (max_tasks_in_state * _get_task_state_size).
        """
        state_parts = []

        # 1. Current Time Slot Index (Normalized)
        # Normalize between 0 (start) and 1 (end of last slot index)
        # Use num_time_slots - 1 as the maximum index for normalization denominator
        normalized_current_slot = (
            self.current_time_slot_index / max(1, self.num_time_slots - 1)
            if self.num_time_slots > 1
            else 0.0
        )
        # Clamp between 0.0 and 1.0 just in case current_time_slot_index goes slightly beyond
        normalized_current_slot = np.clip(
            normalized_current_slot, 0.0, 1.0
        )  # Added clip for robustness
        state_parts.append(normalized_current_slot)

        # 2. Availability/Status of Each Time Slot
        # Represent this as a vector using categories: 0.0: Free, 1.0: Occupied, 2.0: Unavailable
        time_slot_status_vector = np.zeros(self.num_time_slots, dtype=np.float32)
        for i in range(self.num_time_slots):
            if i in self.unavailable_slots_indices:
                time_slot_status_vector[i] = 2.0
            elif i in self.scheduled_slots:
                time_slot_status_vector[i] = 1.0
            # else it remains 0.0 (Free)

        # Add the time slot vector as a list of floats or a numpy array directly
        # Extending with a numpy array is cleaner
        state_parts.append(time_slot_status_vector)

        # 3. State representation for each Task (Fixed Size)
        # We'll concatenate the state vectors of a fixed maximum number of tasks.
        task_states_list = []
        # Get tasks. Process tasks in a consistent order (e.g., by ID) to ensure the state vector's structure is stable.
        # Note: This selects tasks by sorted ID. For large task sets, consider selecting
        # the most relevant tasks (e.g., dependency-ready, high priority, near deadline).
        all_task_ids = sorted(list(Task.registry.keys()))

        for task_id in all_task_ids:
            # Only add task state if we haven't reached the max number of tasks for the state vector
            if len(task_states_list) < self.max_tasks_in_state:
                task_states_list.append(self._get_task_state_representation(task_id))
            else:
                # If more tasks than max_tasks_in_state, ignore the rest for the state vector.
                # The agent won't "see" tasks beyond this limit in the state.
                pass  # Explicitly pass for clarity

        # Pad with zero vectors if the number of tasks is less than max_tasks_in_state
        # This ensures the task_states_flat part always has the same size.
        task_feature_size = self._get_task_state_size()  # Which is currently 8
        while len(task_states_list) < self.max_tasks_in_state:
            task_states_list.append(np.zeros(task_feature_size, dtype=np.float32))

        # Flatten the list of task state arrays into a single NumPy array
        task_states_flat = np.concatenate(task_states_list)
        state_parts.append(task_states_flat)

        # Convert the list of state parts into a single NumPy array
        # The state_parts list now contains [float, np.array, np.array]
        # We need to concatenate these into a single flat array.
        # The first element (normalized_current_slot) needs to be in an array format for concatenate
        final_state_vector = np.concatenate(
            [
                np.array([state_parts[0]], dtype=np.float32),
                state_parts[1],
                state_parts[2],
            ]
        )

        return final_state_vector

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to an initial state for a new episode.
        Resets task statuses and time spent to their initial values,
        clears the schedule, and resets the current time slot.
        Returns the initial state of the environment AND an info dictionary (Gymnasium API).
        """
        print("\n--- Resetting Environment ---")
        # Reset task statuses and time spent using the stored initial states
        # Ensure Task.registry is accessible and is the one we are modifying (it's a class variable)
        # This assumes Task.registry is the primary source of truth and reset modifies it in place.
        # In a multi-agent or more complex setup, you might work on copies.
        for task_id, (
            initial_status,
            initial_time_spent,
        ) in self._initial_task_states.items():
            task = Task.get_task_by_id(task_id)
            if task:  # Ensure task still exists in registry
                task.status = initial_status  # Typically 0.0 if starting fresh
                task.time_spent = initial_time_spent  # Typically 0.0 if starting fresh
                # If your initial state isn't always 0, load those initial values

        # Clear the schedule for the new episode
        self.scheduled_slots = {}
        self.time_spent_in_slots = {}

        # Reset the current time slot index to the beginning of the horizon
        self.current_time_slot_index = 0

        print("Environment reset to initial state.")
        initial_state = self.get_state()
        info = {}  # Create an empty info dictionary for Gymnasium API

        return initial_state, info  #  Return state and info (Gymnasium API)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one step in the environment based on the chosen action.
        Updates the environment's state, calculates rewards, and determines if the episode is done.

        Args:
            action: An integer representing the chosen action (0 for wait, 1+ for task index).
                    Expected to be a single integer for an unwrapped env, or a list/array
                    of integers for a vectorized env (even if num_envs=1).
                    We expect a list/array of size 1 in this script.

        Returns:
            observation (np.ndarray): The state of the environment after executing the action.
            reward (float): The immediate reward received for the action.
            terminated (bool): True if the episode terminated (success or failure end state).
            truncated (bool): True if the episode was truncated (e.g., time limit/horizon reached).
            info (dict): A dictionary containing additional information.
        """
        # In a vectorized environment, action comes in as a list/array [action_int].
        # We need to extract the single action for our env logic since num_envs=1.
        # Add a check here to handle the expected input format
        if isinstance(action, (list, np.ndarray)):
            if len(action) == 1:
                action_to_take = action[0]
            else:
                # This shouldn't happen with num_envs=1 and make_vec_env unless something is wrong.
                print(
                    f"Error: Expected action list/array of size 1, but got size {len(action)}"
                )
                # Handle this as a critical error - cannot proceed
                # Return terminal state, 0 reward, terminated=True, truncated=False, error info
                error_info = {
                    "is_valid": False,
                    "error": "unexpected_action_batch_size",
                }
                current_state = (
                    self.get_state()
                )  # Or return zeros if state is unreliable
                return current_state, 0.0, True, False, error_info
        else:
            # If not using make_vec_env, action might be a single int.
            # For consistency with the script's use of make_vec_env,
            # we expect the list/array format. Print a warning if it's just int.
            print(
                f"Warning: Received single integer action {action}. Expected list/array [action]. Assuming num_envs=1 format."
            )
            action_to_take = action

        print(
            f"\n--- Processing Action {action_to_take} at Time Slot {self.current_time_slot_index} ({self._get_current_datetime().strftime('%Y-%m-%d %H:%M')}) ---"
        )

        reward = 0.0  # Immediate reward for this step
        info = {"is_valid": True, "scheduled_task_id": None}  # Info dictionary

        current_slot_index = self.current_time_slot_index
        current_datetime = self._get_current_datetime()

        # --- 1. Validate and Process the Action ---

        # Check if the current slot is unavailable (static constraint)
        if current_slot_index in self.unavailable_slots_indices:
            if (
                action_to_take != 0
            ):  # If the agent tried to schedule something here (not wait)
                print("Invalid Action: Attempted to schedule in an unavailable slot.")
                reward = -1.0  # Penalty
                info["is_valid"] = False
                info["error"] = "scheduled_unavailable_slot"
            else:  # Action was 'wait' (0) in an unavailable slot
                print("Action: Waiting in an unavailable slot.")
                # reward remains 0.0

        # Process Action 0: Wait (only if slot is available)
        elif action_to_take == 0:
            print("Action: Waiting in the current time slot.")
            # reward remains 0.0

        # Process Actions > 0: Attempt to Schedule Task (only if slot is available)
        else:  # action_to_take > 0
            # Determine which task the agent is trying to schedule based on action index
            task_index_in_state = action_to_take - 1

            # Get the actual task ID from the sorted list of tasks considered in the state
            # This uses the assumption that tasks in state are sorted by ID (ensure this is true in get_state)
            all_task_ids_sorted = sorted(list(Task.registry.keys()))

            # --- Validate Scheduling Attempt against Environment Rules ---
            task_to_schedule = None  # Initialize task_to_schedule

            # Check if the action index corresponds to a valid task within the current task registry size
            if task_index_in_state >= len(all_task_ids_sorted):
                print(
                    f"Invalid Action: Attempted to schedule a task from state index {task_index_in_state}, but only {len(all_task_ids_sorted)} tasks exist in registry."
                )
                reward = -1.0  # Penalty
                info["is_valid"] = False
                info["error"] = "scheduled_padded_task"
                # task_to_schedule remains None

            else:
                task_id_to_schedule = all_task_ids_sorted[task_index_in_state]
                task_to_schedule = Task.get_task_by_id(
                    task_id_to_schedule
                )  # Retrieve the actual task object

                if (
                    task_to_schedule is None
                ):  # Should not happen if task_id came from registry keys
                    print(
                        f"Error: Task ID {task_id_to_schedule} not found in registry after validation."
                    )
                    reward = -1.0  # Penalty
                    info["is_valid"] = False
                    info["error"] = (
                        "internal_task_not_found"  # Indicates an unexpected state
                    )

                elif current_slot_index in self.scheduled_slots:
                    print(
                        f"Invalid Action: Slot {current_slot_index} is already occupied by Task {self.scheduled_slots[current_slot_index]}."
                    )
                    reward = -1.0  # Penalty
                    info["is_valid"] = False
                    info["error"] = "slot_occupied"

                elif task_to_schedule.is_completed():
                    print(
                        f"Invalid Action: Task {task_to_schedule.ID} ('{task_to_schedule.name}') is already completed."
                    )
                    reward = (
                        -0.5
                    )  # Small penalty for trying to schedule a completed task
                    info["is_valid"] = False
                    info["error"] = "task_already_completed"

                else:
                    # Check dependencies
                    dependencies_met = True
                    for dep_id in task_to_schedule.dependencies:
                        dep_task = Task.get_task_by_id(dep_id)
                        # Dependency is met if dependency task exists AND is completed
                        # If dep_task is None, or not completed, dependencies are not met
                        if dep_task is None or not dep_task.is_completed():
                            dependencies_met = False
                            if dep_task is None:
                                print(
                                    f"Warning: Dependency task {dep_id} not found in registry for task {task_to_schedule.ID}."
                                )
                            break  # Stop checking dependencies if one is unmet/missing

                    if not dependencies_met:
                        print(
                            f"Invalid Action: Dependencies not met for task {task_to_schedule.ID} ('{task_to_schedule.name}')."
                        )
                        reward = -1.0  # Penalty for violating dependencies
                        info["is_valid"] = False
                        info["error"] = "dependencies_not_met"
                    else:
                        # --- Valid Scheduling Action ---
                        # If we reach here, the action is valid: schedule the task in the current slot.
                        print(
                            f"Valid Action: Scheduling Task {task_to_schedule.ID} ('{task_to_schedule.name}') in slot {current_slot_index}."
                        )
                        self.scheduled_slots[current_slot_index] = task_to_schedule.ID
                        info["scheduled_task_id"] = task_to_schedule.ID

                        # Simulate working on the task for the duration of the time slot
                        work_duration_hours = (
                            self.slot_duration.total_seconds() / 3600.0
                        )

                        # Update task's time spent and status
                        task_to_schedule.time_spent += work_duration_hours
                        # Recalculate status based on new time_spent, capping at 1.0
                        if task_to_schedule.duration > 0:
                            task_to_schedule.status = (
                                task_to_schedule.time_spent / task_to_schedule.duration
                            )
                        else:  # Handle duration = 0 case
                            task_to_schedule.status = (
                                1.0 if task_to_schedule.time_spent > 0 else 0.0
                            )
                        task_to_schedule.status = min(
                            task_to_schedule.status, 1.0
                        )  # Cap status at 1.0

                        # Record time spent in this specific slot (optional, for tracking)
                        self.time_spent_in_slots[
                            (task_to_schedule.ID, current_slot_index)
                        ] = work_duration_hours

                        print(
                            f"Task {task_to_schedule.ID} status updated to {task_to_schedule.status:.2f}. Total time spent: {task_to_schedule.time_spent:.2f}"
                        )

                        # --- Calculate Immediate Reward for Valid Scheduling Action ---
                        # Reward for making progress on a task (proportional to work done)
                        reward += (
                            0.1 * work_duration_hours
                        )  # Small positive reward for working

                        # Check for task completion *after* updating status in this step
                        if task_to_schedule.is_completed():
                            print(
                                f"Task {task_to_schedule.ID} ('{task_to_schedule.name}') completed!"
                            )
                            # Significant positive reward for completing a task, scaled by base priority
                            completion_reward = (
                                5.0 * task_to_schedule.priority
                            )  # Example reward
                            reward += completion_reward
                            info["completed_task"] = task_to_schedule.ID

        # --- 2. Advance Environment Time ---
        # Time always advances by one slot per step taken, regardless of action validity (unless terminal)
        self.current_time_slot_index += 1

        # --- 3. Check if Episode is Terminated or Truncated ---
        terminated = False  # Initialize flags for this check
        truncated = False

        # Condition 1: End of time slots reached (check using the NEW index)
        if self.current_time_slot_index >= self.num_time_slots:
            truncated = True  # Episode truncated by reaching horizon
            print("End of scheduling horizon reached.")  # Print truncated message

        # Condition 2: All tasks completed (check using the latest statuses)
        # Check completion status AFTER advancing time and processing potential completion from THIS step
        all_tasks_completed = all(
            task.is_completed() for task in Task.registry.values()
        )
        if all_tasks_completed:
            terminated = True  # Episode terminates by completing all tasks
            # Only print if this is the primary reason for termination
            if not truncated:
                print("All tasks completed.")

        # --- 4. Calculate Final Rewards/Penalties at the END of the episode ---
        # This block runs if the episode just terminated OR was truncated.
        if terminated or truncated:
            print("Calculating final episode rewards/penalties.")
            # Penalties for missed deadlines for tasks that are NOT completed by the end time
            # Use the time corresponding to the start of the NEW current slot index.
            # If current_time_slot_index == num_time_slots, this is the time *after* the last slot ends.
            current_datetime_at_end = self._get_current_datetime()

            num_missed_deadlines = 0
            for task in Task.registry.values():
                # Only penalize uncompleted tasks whose deadlines were missed by the episode end time
                if not task.is_completed() and current_datetime_at_end > task.deadline:
                    num_missed_deadlines += 1
                    print(f"Task {task.ID} ('{task.name}') missed deadline.")
                    # Penalty scaled by priority and how much work was remaining
                    remaining_work_ratio = max(
                        0.0, 1.0 - task.status
                    )  # Ensure ratio is not negative
                    penalty = (
                        -10.0 * task.priority * remaining_work_ratio
                    )  # Example penalty - adjust magnitude!
                    reward += penalty  # Add penalty to the final reward

            # Update info dictionary with episode outcome and metrics
            if terminated and all_tasks_completed:
                info["episode_outcome"] = "completed"
            elif truncated:
                info["episode_outcome"] = "truncated_horizon"
            elif (
                terminated
            ):  # Should only happen if another termination condition was added?
                info["episode_outcome"] = "terminated"
            else:  # Should not happen if done is True
                info["episode_outcome"] = "unknown_termination"

            info["num_missed_deadlines"] = num_missed_deadlines

        # --- 5. Get Next State ---
        next_state = self.get_state()

        # --- DEBUG PRINT ---
        print(
            f"DEBUG: Step returning - Terminated: {terminated}, Truncated: {truncated}"
        )
        if terminated or truncated:
            # This print should ideally match the "Episode finished after..." message
            print(
                f"DEBUG: Episode ending after processing slot {self.current_time_slot_index - 1}. Final step reward: {reward}"
            )

        # --- 6. Return the results of the step ---
        return next_state, reward, terminated, truncated, info


def render(self, mode="human"):
    """
    Renders the current state of the environment as text to the console.
    """
    if mode != "human":
        # In a real environment, you might return an image array or string here
        super().render(mode=mode)
        return

    print("\n" + "=" * 40)
    print(f"Current Time Slot: {self.current_time_slot_index}")
    print(
        f"Current Datetime: {self._get_current_datetime().strftime('%Y-%m-%d %H:%M')}"
    )
    print(f"Total Time Slots: {self.num_time_slots}")
    print("=" * 40)

    # --- Visualize Time Slots ---
    print("\nTime Slot Status:")
    status_line = ""
    # Characters to represent slot status:
    # . : Free
    # O : Occupied
    # X : Unavailable
    # > : Current Slot Marker

    # Let's show a window around the current slot, or the whole horizon if small
    window_size = 20  # Number of slots to show around current slot
    start_idx = max(0, self.current_time_slot_index - window_size // 2)
    end_idx = min(self.num_time_slots, start_idx + window_size)
    # Adjust start_idx if end_idx hits the limit
    if end_idx == self.num_time_slots:
        start_idx = max(0, end_idx - window_size)

    # Optional: Print header row with indices
    header_indices = " " * 5  # Offset for label
    markers = " " * 5  # Offset for label
    indices_line = " " * 5
    for i in range(start_idx, end_idx):
        if i % 5 == 0:  # Print index every 5 slots
            indices_line += f"{i:<5}"[:5]  # Print index, pad to 5, truncate if needed
        else:
            indices_line += " " * 5
        header_indices += "     "  # Space for the status character below

    print(header_indices)
    print(indices_line)
    print(
        "-" * (5 + (end_idx - start_idx) * 1)
    )  # Separator line based on number of slots shown

    # Print status characters
    status_line = "Slots:"
    for i in range(start_idx, end_idx):
        char = "."  # Default to free
        if i in self.unavailable_slots_indices:
            char = "X"  # Unavailable overrides free/occupied
        elif i in self.scheduled_slots:
            char = "O"  # Occupied

        if i == self.current_time_slot_index:
            status_line += f"[{char}]"  # Mark the current slot
        else:
            status_line += f" {char} "  # Regular slot

    print(status_line)

    # --- Summarize Task Statuses ---
    print("\nTask Status Summary:")
    if not Task.registry:
        print("No tasks in the registry.")
    else:
        # Sort tasks for consistent rendering order
        sorted_task_ids = sorted(Task.registry.keys())
        for task_id in sorted_task_ids:
            task = Task.get_task_by_id(task_id)
            if (
                task
            ):  # Ensure task exists (should always be true if ID from registry keys)
                dependencies_met = True
                # Re-check dependencies here for the display
                for dep_id in task.dependencies:
                    dep_task = Task.get_task_by_id(dep_id)
                    if dep_task is None or not dep_task.is_completed():
                        dependencies_met = False
                        break

                deadline_passed_now = self._get_current_datetime() > task.deadline

                print(
                    f"  Task {task.ID} ('{task.name}'): "
                    f"Status: {task.status:.2f}/{task.duration:.2f} ({task.status * 100:.0f}%) "  # Show current work / total duration (% complete)
                    f"| Time Spent: {task.time_spent:.2f} "
                    f"| Deadline: {task.deadline.strftime('%Y-%m-%d %H:%M')} {'(PASSED)' if deadline_passed_now and not task.is_completed() else ''} "
                    f"| Dependencies Met: {'Yes' if dependencies_met else 'No'} "
                    f"| Completed: {'Yes' if task.is_completed() else 'No'}"
                )

    print("=" * 40 + "\n")
