import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from task import Task # Assuming task.py exists and contains the Task class
from dag_builder import build_task_dag # Assuming dag_builder.py exists and contains build_task_dag
import gymnasium as gym

class TaskSchedulingEnv(gym.Env):
    """
    Reinforcement Learning Environment for Task Scheduling.
    Defines the state space, action space, reward function, and environment dynamics in step().
    """
    def __init__(self, task_registry: Dict[int, Task], start_time: datetime, end_time: datetime, slot_duration_hours: int = 1, max_tasks_in_state: int = 50):
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
        super().__init__() # <--- MODIFICATION: Call the base class constructor

        if not isinstance(task_registry, dict):
            raise TypeError("task_registry must be a dictionary.")
        if not all(isinstance(task, Task) for task in task_registry.values()):
             print("Warning: task_registry contains non-Task objects.") # Or raise error

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
        self.time_slots = self._generate_time_slots(self.start_time, self.end_time, self.slot_duration)
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
                slots_per_day = self.num_time_slots # Or raise an error

            # Example: Make 12pm-1pm and 6pm-7pm unavailable *every day* within the horizon
            unavail_hours_daily = [12, 18] # Example: 12:00 (noon) and 18:00 (6 PM)

            # Calculate the index offset for these hours on a 'base' day starting at self.start_time.hour
            base_hour_offset = self.start_time.hour
            unavail_slot_offsets_from_start_hour = [(h - base_hour_offset + 24) % 24 for h in unavail_hours_daily] # Ensure positive offset

            # Iterate through all slots and mark those corresponding to the unavailable times
            for i in range(self.num_time_slots):
                 slot_datetime = self.time_slots[i]
                 # Calculate the hour offset from the start time of the horizon
                 hour_diff = (slot_datetime - self.start_time).total_seconds() / 3600.0
                 # Calculate the effective hour within a 24-hour cycle relative to the start hour
                 effective_hour_offset = int(round(hour_diff)) % slots_per_day # Use round for potential floating point

                 if effective_hour_offset in unavail_slot_offsets_from_start_hour:
                     # Check if the actual hour of the slot matches the unavailable hour
                     slot_hour = slot_datetime.hour
                     if slot_hour in unavail_hours_daily:
                         self.unavailable_slots_indices.add(i)
                      #else:
                         # print(f"Calculated offset {effective_hour_offset} for slot {i} (hour {slot_hour}) does not match unavailable hour pattern.")

        except Exception as e:
             print(f"Error calculating assumed unavailable slots: {e}")
             # Decide how to handle this error - maybe proceed with no unavailable slots?


        print(f"Initialized with {len(self.unavailable_slots_indices)} assumed unavailable time slot indices.")
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
        print(f"Action Space Size: {self.action_space_size} (1 for wait + {self.max_tasks_in_state} for scheduling attempts)")

        # <--- MODIFICATION: Define the action space for Gymnasium ---
        self.action_space = gym.spaces.Discrete(self.action_space_size)
        print(f"Gymnasium Action Space: {self.action_space}")


        # <--- MODIFICATION: Define the Observation Space Definition ---
        # Defines the structure, shape, and bounds of the state vector returned by get_state()
        task_feature_size = self._get_task_state_size() # Which is currently 8
        observation_space_size = 1 + self.num_time_slots + (self.max_tasks_in_state * task_feature_size)

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
        # Based on your _get_task_state_representation and comments, and assuming
        # your max_task_duration/max_base_priority lead to intended scaling.
        # We use a safe high bound for normalized remaining duration just in case it exceeds 1.0.
        high_task_features = np.array([
            1.0,   # 1. Status (0 to 1)
            100.0, # 2. Remaining Duration (normalized - using a large safe number like 100)
            1.0,   # 3. Time until Deadline (normalized 0 to 1)
            1.0,   # 4. Base Priority (normalized 0 to 1)
            1.0,   # 5. Dependencies Met (Binary 0 or 1)
            1.0,   # 6. Is scheduled in current slot (Binary 0 or 1)
            1.0,   # 7. Has deadline passed (Binary 0 or 1)
            1.0    # 8. Dynamic Priority Score (0 to 1)
        ], dtype=np.float32)

        # Tile the high bounds for each task feature across all max_tasks_in_state slots
        high[1 + self.num_time_slots : ] = np.tile(high_task_features, self.max_tasks_in_state)


        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(observation_space_size,),
            dtype=np.float32
        )
        print(f"Gymnasium Observation Space Shape: {self.observation_space.shape}")


        # Keep track of the current state - will be updated by step()
        self.current_time_slot_index = 0

        # Store the initial task states or a copy of the initial registry
        # so we can reset the environment accurately for new episodes.
        # A deep copy might be necessary if tasks are mutable.
        self._initial_task_states = {task_id: (task.status, task.time_spent) for task_id, task in self.task_registry.items()}
        # Optionally store initial dependencies if they could change (less likely in this project)

        # Reset the environment to a starting state
        self.reset()


    def _generate_time_slots(self, start: datetime, end: datetime, duration: timedelta) -> List[datetime]:
        """Generates a list of start times for each time slot."""
        slots = []
        current = start
        while current < end:
            slots.append(current)
            current += duration
            # Add a small epsilon to avoid floating point issues near the end time
            if current + timedelta(seconds=0.001) > end and current < end:
                # Ensure the last slot doesn't go beyond the end time
                if start + timedelta(seconds=(end - start).total_seconds()) < end: # Add only if it's a valid slot start before end
                    slots.append(start + timedelta(seconds=(end - start).total_seconds()))
                break # Stop after adding the last potential slot
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
            return self.end_time # Or handle as an error state (e.g., raise Exception)


    def _get_task_state_representation(self, task_id: int) -> np.ndarray:
        """
        Generates a numerical representation for a single task's state.
        This vector will be part of the overall environment state.
        Ensure all features are scaled or normalized appropriately (e.g., 0 to 1).
        """
        task = Task.get_task_by_id(task_id) # Get task from the class registry
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
        max_task_duration = 20.0 # Example max duration, adjust based on your data
        remaining_duration = max(0.0, task.duration - task.time_spent)
        features.append(remaining_duration / max_task_duration if max_task_duration > 0 else 0.0)

        # 3. Time until Deadline (normalize by total scheduling horizon duration)
        total_horizon_seconds = (self.end_time - self.start_time).total_seconds()
        current_datetime = self._get_current_datetime()
        time_until_deadline_seconds = max(0.0, (task.deadline - current_datetime).total_seconds())
        # Normalize between 0 (deadline is now or passed) and 1 (deadline is at the end of the horizon or later)
        normalized_time_until_deadline = time_until_deadline_seconds / total_horizon_seconds if total_horizon_seconds > 0 else 0.0
        features.append(normalized_time_until_deadline)

        # 4. Base Priority (normalize by maximum possible priority)
        max_base_priority = 10 # Example max priority, adjust based on your data
        features.append(task.priority / max_base_priority if max_base_priority > 0 else 0.0)

        # 5. Dependencies Met Status (Binary: 1 if all dependencies are completed, 0 otherwise)
        dependencies_met = True
        # Check dependencies using the DAG or task registry
        for dep_id in task.dependencies: # Assuming task.dependencies is up-to-date
             dep_task = Task.get_task_by_id(dep_id)
             # A dependency is met if the dependency task exists and is completed
             # Handle case where a dependency might not be in the registry (e.g., was removed)
             if dep_task is None or not dep_task.is_completed():
                 dependencies_met = False
                 break
        features.append(1.0 if dependencies_met else 0.0)

        # 6. Is this task currently scheduled in the CURRENT time slot? (Binary)
        # This checks if the task_id is in the scheduled_slots dictionary at the current index
        is_scheduled_in_current_slot = 1.0 if self.scheduled_slots.get(self.current_time_slot_index) == task_id else 0.0
        features.append(is_scheduled_in_current_slot)

        # 7. Has the task's deadline passed? (Binary)
        current_datetime = self._get_current_datetime() # Ensure current_datetime is up to date
        deadline_passed = 1.0 if current_datetime > task.deadline and not task.is_completed() else 0.0
        features.append(deadline_passed)

       # --- Feature 8: Dynamic Priority Score ---
        current_datetime = self._get_current_datetime() # Needed for computePriorityScore

        # Calculate the total environment horizon duration in days as a scaling factor.
        # This value is constant for a given environment setup (start_time to end_time).
        total_env_horizon_seconds = (self.end_time - self.start_time).total_seconds()
        total_env_horizon_days_for_scaling = max(1.0, total_env_horizon_seconds / (24 * 3600.0))


        # Assuming Task.computePriorityScore is designed to use 'max_horizon' as a scaling factor
        dynamic_score = task.computePriorityScore(
             current_datetime=current_datetime, # Pass current time (used internally by Task for clamped_days)
             max_horizon=total_env_horizon_days_for_scaling, # <--- THIS IS THE MODIFICATION
             max_priority=10, # Pass your configured max priority (adjust if different)
             max_duration=20.0, # Pass your configured max duration (adjust if different)
             max_deps=max(1, len(task.dependencies) + 5) # Pass a scale for max dependencies
             )
        features.append(dynamic_score)

        return np.array(features, dtype=np.float32)

    def _get_task_state_size(self) -> int:
         """Returns the number of features in the state representation for a single task."""
         # Updated to reflect the added dynamic priority score
         return 8 # status, remaining_duration, time_until_deadline, base_priority, dependencies_met, scheduled_in_current_slot, deadline_passed, dynamic_priority_score


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
        normalized_current_slot = self.current_time_slot_index / max(1, self.num_time_slots - 1) if self.num_time_slots > 1 else 0.0
        # Clamp between 0.0 and 1.0 just in case current_time_slot_index goes slightly beyond
        normalized_current_slot = np.clip(normalized_current_slot, 0.0, 1.0) # Added clip for robustness
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
                 pass # Explicitly pass for clarity


        # Pad with zero vectors if the number of tasks is less than max_tasks_in_state
        # This ensures the task_states_flat part always has the same size.
        task_feature_size = self._get_task_state_size() # Which is currently 8
        while len(task_states_list) < self.max_tasks_in_state:
            task_states_list.append(np.zeros(task_feature_size, dtype=np.float32))

        # Flatten the list of task state arrays into a single NumPy array
        task_states_flat = np.concatenate(task_states_list)
        state_parts.append(task_states_flat)


        # Convert the list of state parts into a single NumPy array
        # The state_parts list now contains [float, np.array, np.array]
        # We need to concatenate these into a single flat array.
        # The first element (normalized_current_slot) needs to be in an array format for concatenate
        final_state_vector = np.concatenate([np.array([state_parts[0]], dtype=np.float32), state_parts[1], state_parts[2]])


        return final_state_vector


    def reset(self, seed: Optional[int] = None, 
              options: Optional[dict] = None, ) -> Tuple[np.ndarray, Dict[str, Any]]: # <--- MODIFICATION: Update return type hint for Gymnasium
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
        for task_id, (initial_status, initial_time_spent) in self._initial_task_states.items():
             task = Task.get_task_by_id(task_id)
             if task: # Ensure task still exists in registry
                 task.status = initial_status # Typically 0.0 if starting fresh
                 task.time_spent = initial_time_spent # Typically 0.0 if starting fresh
                 # If your initial state isn't always 0, load those initial values

        # Clear the schedule for the new episode
        self.scheduled_slots = {}
        self.time_spent_in_slots = {}

        # Reset the current time slot index to the beginning of the horizon
        self.current_time_slot_index = 0

        print("Environment reset to initial state.")
        initial_state = self.get_state()
        info = {} # <--- MODIFICATION: Create an empty info dictionary for Gymnasium API

        return initial_state, info # <--- MODIFICATION: Return state and info (Gymnasium API)


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: # <--- MODIFICATION: Update return type hint for Gymnasium (terminated, truncated)
        """
        Executes one step in the environment based on the chosen action.
        Updates the environment's state, calculates the reward, and determines if the episode is done.

        Args:
            action: An integer representing the chosen action.
                    Action 0: Wait in the current time slot.
                    Action 1 to max_tasks_in_state: Attempt to schedule the task
                    at index (action - 1) in the sorted list of tasks considered
                    in the state, into the CURRENT time slot.

        Returns:
            observation (object): The state of the environment after executing the action.
            reward (float): The immediate reward received for the action.
            terminated (bool): True if the episode terminated (failure or success condition met).
            truncated (bool): True if the episode was truncated (e.g., time limit exceeded, but not a failure/success).
            info (dict): A dictionary containing additional information (e.g., debugging, metrics).
        """
        # Ensure action is within the valid range
        if not 0 <= action < self.action_space_size:
             print(f"Invalid action received: {action}. Must be between 0 and {self.action_space_size - 1}.")
             # Penalize invalid action and transition to the next time slot
             reward = -2.0 # Large penalty for an invalid action
             terminated = False # <--- MODIFICATION: Use 'terminated'
             truncated = False # <--- MODIFICATION: Set 'truncated' (False as no explicit truncation condition)
             info = {'is_valid': False, 'error': 'invalid_action_index'} # Added error info

             # Advance time even on invalid action (original logic)
             self.current_time_slot_index += 1

             # Check if this advancement caused us to hit the end of the horizon
             if self.current_time_slot_index >= self.num_time_slots:
                 terminated = True # End of episode if we run out of slots
                 print("Step: Reached end of time slots after invalid action.")
                 # Note: Terminal rewards/penalties for this case are calculated below
                 # in the 'if terminated:' block, after time advancement.

             next_state = self.get_state()
             # <--- MODIFICATION: Update return value format for Gymnasium ---
             return next_state, reward, terminated, truncated, info


        print(f"\n--- Processing Action {action} at Time Slot {self.current_time_slot_index} ({self._get_current_datetime().strftime('%Y-%m-%d %H:%M')}) ---")

        reward = 0.0
        terminated = False # <--- MODIFICATION: Initialize 'terminated'
        truncated = False # <--- MODIFICATION: Initialize 'truncated'
        info = {'is_valid': True} # Assume valid initially

        current_slot_index = self.current_time_slot_index
        # current_datetime = self._get_current_datetime() # Already called below if needed

        # --- Check if the current slot is unavailable (static constraint) ---
        if current_slot_index in self.unavailable_slots_indices:
             print(f"Slot {current_slot_index} is unavailable. Any scheduled action will be invalid.")
             # If the agent tried to schedule something here, it's invalid.
             # If the agent chose 'wait', it's a valid action in an unavailable slot.
             if action != 0: # If action is not 'wait'
                 print("Invalid Action: Attempted to schedule in an unavailable slot.")
                 reward = -1.0 # Penalty for trying to use an unavailable slot
                 info['is_valid'] = False
                 info['error'] = 'scheduled_unavailable_slot'
             else: # Action is 'wait' in an unavailable slot
                  print("Action: Waiting in an unavailable slot.")
                  # reward = 0.0 # Already initialized to 0.0


        # --- Process Action 0: Wait ---
        elif action == 0:
            print("Action: Waiting in the current time slot.")
            # reward = 0.0 # Already initialized to 0.0

        # --- Process Actions 1 to max_tasks_in_state: Attempt to Schedule Task ---
        else: # Action > 0, attempting to schedule
             # Determine which task the agent is trying to schedule based on action index
             task_index_in_state = action - 1
             # Get the actual task ID from the sorted list of tasks considered in the state
             # This uses the assumption that tasks in state are sorted by ID
             all_task_ids_sorted = sorted(list(Task.registry.keys()))

             # --- Validate Scheduling Action ---
             # Check if the action index corresponds to a valid task within the current task registry size
             # This prevents trying to schedule a "padded" task slot if there are fewer actual tasks than max_tasks_in_state
             if task_index_in_state >= len(all_task_ids_sorted):
                 print(f"Invalid Action: Attempted to schedule a task from state index {task_index_in_state}, but only {len(all_task_ids_sorted)} tasks exist in registry.")
                 reward = -1.0 # Penalty for trying to schedule a task that doesn't exist
                 info['is_valid'] = False
                 info['error'] = 'scheduled_padded_task'
                 task_to_schedule = None # Ensure task_to_schedule is None for subsequent checks
             else:
                 task_id_to_schedule = all_task_ids_sorted[task_index_in_state]
                 task_to_schedule = Task.get_task_by_id(task_id_to_schedule) # Retrieve the actual task object

                 # Validate the scheduling attempt against environment rules
                 # Note: `current_slot_index in self.unavailable_slots_indices` is checked BEFORE this `else` block.
                 # So, if we are in this block, the slot is considered AVAILABLE according to the static rules.
                 if task_to_schedule is None: # This should theoretically not happen if task_id came from registry keys
                     print(f"Error: Task ID {task_id_to_schedule} not found in registry after validation.")
                     reward = -1.0 # Penalty
                     info['is_valid'] = False
                     info['error'] = 'internal_task_not_found' # Indicates an unexpected state

                 elif current_slot_index in self.scheduled_slots:
                     print(f"Invalid Action: Slot {current_slot_index} is already occupied by Task {self.scheduled_slots[current_slot_index]}.")
                     reward = -1.0 # Penalty
                     info['is_valid'] = False
                     info['error'] = 'slot_occupied'

                 elif task_to_schedule.is_completed():
                     print(f"Invalid Action: Task {task_to_schedule.ID} ('{task_to_schedule.name}') is already completed.")
                     reward = -0.5 # Small penalty for trying to schedule a completed task
                     info['is_valid'] = False
                     info['error'] = 'task_already_completed'

                 else:
                     # Check dependencies
                     dependencies_met = True
                     # Assume Task.get_task_by_id handles non-existent dep_id gracefully (e.g., returns None)
                     for dep_id in task_to_schedule.dependencies:
                         dep_task = Task.get_task_by_id(dep_id)
                         # Dependency is met if dependency task exists AND is completed
                         if dep_task is None:
                             print(f"Warning: Dependency task {dep_id} not found in registry for task {task_to_schedule.ID}.")
                             # Treat missing dependency as unmet
                             dependencies_met = False
                             break # Stop checking dependencies if one is unmet/missing
                         if not dep_task.is_completed():
                              dependencies_met = False
                              break # Stop checking dependencies if one is not completed


                     if not dependencies_met:
                         print(f"Invalid Action: Dependencies not met for task {task_to_schedule.ID} ('{task_to_schedule.name}').")
                         reward = -1.0 # Penalty for violating dependencies
                         info['is_valid'] = False
                         info['error'] = 'dependencies_not_met'
                     else:
                         # --- Valid Scheduling Action ---
                         # If we reach here, the action is valid: schedule the task in the current slot.
                         print(f"Valid Action: Scheduling Task {task_to_schedule.ID} ('{task_to_schedule.name}') in slot {current_slot_index}.")
                         self.scheduled_slots[current_slot_index] = task_to_schedule.ID

                         # Simulate working on the task for the duration of the time slot
                         work_duration_hours = self.slot_duration.total_seconds() / 3600.0 # Work for the full slot duration

                         # Update task's time spent and status
                         task_to_schedule.time_spent += work_duration_hours
                         # Recalculate status based on new time_spent, capping at 1.0
                         # If duration is 0, status is 1 if time spent > 0, else 0.
                         task_to_schedule.status = task_to_schedule.time_spent / task_to_schedule.duration if task_to_schedule.duration > 0 else (1.0 if task_to_schedule.time_spent > 0 else 0.0)
                         task_to_schedule.status = min(task_to_schedule.status, 1.0) # Cap status at 1.0

                         # Record time spent in this specific slot
                         self.time_spent_in_slots[(task_to_schedule.ID, current_slot_index)] = work_duration_hours

                         print(f"Task {task_to_schedule.ID} status updated to {task_to_schedule.status:.2f}. Total time spent: {task_to_schedule.time_spent:.2f}")

                         # --- Calculate Immediate Reward for Valid Scheduling Action ---
                         # Reward for making progress on a task (proportional to work done, which is slot duration)
                         reward += 0.1 * work_duration_hours # Small positive reward for working

                         # Check for task completion *after* updating status
                         if task_to_schedule.is_completed():
                              print(f"Task {task_to_schedule.ID} ('{task_to_schedule.name}') completed!")
                              # Significant positive reward for completing a task, scaled by base priority
                              completion_reward = 5.0 * task_to_schedule.priority # Example reward
                              reward += completion_reward
                              info['completed_task'] = task_to_schedule.ID


        # --- Advance Environment Time ---
        # In this discrete time slot environment, time advances by one slot per step,
        # regardless of whether a task was scheduled or the action was valid (except for the invalid action index case above).
        if action != 0 or current_slot_index not in self.unavailable_slots_indices: # Advance time unless it was an invalid action index (already handled) or wait in unavailable
            # Re-checking the invalid index condition just to be crystal clear time advances unless it was that specific error
            # A simpler logic is time always advances unless the episode just ended from the *previous* step's logic.
            # Let's rely on the main time advance logic always running unless episode ends.
            pass # Time advancement is handled *after* this validation block

        # Time advances *one* slot per step AFTER processing the action for the *current* slot.
        # The initial invalid action index check handles time advancement internally for that specific case.
        # For all other cases (action 0, or valid/invalid scheduling attempts), time advances here.
        if not (not 0 <= action < self.action_space_size and self.current_time_slot_index + 1 <= self.num_time_slots):
             # This condition is a bit convoluted. Let's simplify: time always advances by 1 unless
             # the episode termination logic below determines the episode is over *before* the next step.
             # The initial invalid index case already handled its own time advance and termination check.
             # Let's remove the redundant time advance inside the invalid action check at the top,
             # and let time always advance by 1 at the end of the step. This is the standard RL loop.
             pass # Keep the main time advance logic below.

        # --- Standard Time Advance (occurs for every step taken) ---
        # This should happen after processing the action and calculating immediate rewards,
        # but before checking termination conditions *for the next state*.
        # Let's move the time advance to the end of the step logic.
        # The original code had it before the done check, which means the 'current_time_slot_index'
        # could be beyond num_time_slots *before* the done check happens, correctly signaling end.
        # Let's stick to that structure.

        # Original code structure:
        # 1. Process action, calc immediate reward
        # 2. Advance time (self.current_time_slot_index += 1)
        # 3. Check if done (using the NEW index)
        # 4. Calc terminal rewards (if done)
        # 5. Get next state (using the NEW index)
        # 6. Return

        # This structure implies that the time index refers to the *start* of the slot
        # being processed, and after the step, the index points to the *next* slot.
        # This is standard. Let's keep the time advance *before* the main done check.

        # If the *initial* check for action index was invalid, time already advanced and checked done.
        # If we are in the main `elif action == 0:` or `else:` block, the initial index check passed.
        # In these cases, time should advance *once* at the end of processing the current slot.

        # Let's simplify:
        # Handle invalid action index (top) - it does its own time advance and return.
        # For *all other valid action indices*, process action, calculate immediate reward,
        # THEN advance time by 1, THEN check if terminated/truncated using the new index,
        # THEN calculate terminal rewards, THEN get next state, THEN return.

        # The time advance IS already at the correct logical point in the original code structure
        # (before the main done check). My previous edits added a conflicting time advance
        # inside the invalid action index check. Let's remove that conflicting one.

        # REMOVED: self.current_time_slot_index += 1 # Advance time
        # REMOVED: if self.current_time_slot_index >= self.num_time_slots: ... done = True ...
        # REMOVED: next_state = self.get_state()
        # REMOVED: return next_state, reward, done, info # <-- Invalid action branch return


        # --- Main Time Advance (For all valid actions 0 to action_space_size - 1) ---
        # This occurs after processing the action for the *current* slot.
        self.current_time_slot_index += 1 # Advance to the next time slot


        # --- Check if Episode is Terminated ---
        # The episode ends (terminates) when:
        # 1. All time slots in the horizon are used (the new index is >= num_time_slots).
        # 2. All tasks are completed.

        # Condition 1: End of time slots reached (check using the new index)
        if self.current_time_slot_index >= self.num_time_slots:
             terminated = True # Episode terminates
             print("End of scheduling horizon reached. Episode Terminated.")
             # Note: Terminal rewards/penalties for this case are calculated below.

        # Condition 2: All tasks completed
        # Check completion status AFTER advancing time, so reward applies in the step that finishes the task
        all_tasks_completed = all(task.is_completed() for task in Task.registry.values())
        if all_tasks_completed:
             # If all tasks are completed, the episode terminates, potentially before reaching the horizon end.
             # Setting terminated=True here ensures the episode ends.
             if not terminated: # Only print the primary reason if not already set by horizon end
                 print("All tasks completed. Episode Terminated.")
             terminated = True
             # Maybe add a bonus reward for finishing early if applicable (calculate here before terminal penalties)


        # --- Calculate Final Rewards/Penalties at the END of the episode ---
        # This block runs if the episode just terminated (`terminated` is True).
        if terminated:
             print("Calculating final episode rewards/penalties.")
             # Penalties for missed deadlines for tasks that are NOT completed
             # Use the time corresponding to the start of the NEW current slot index.
             # If current_time_slot_index == num_time_slots, this is the time *after* the last slot ends.
             current_datetime_at_end = self._get_current_datetime()

             num_missed_deadlines = 0
             for task in Task.registry.values():
                 # Only penalize uncompleted tasks whose deadlines were missed by the time the episode ended
                 if not task.is_completed() and current_datetime_at_end > task.deadline:
                     num_missed_deadlines += 1
                     print(f"Task {task.ID} ('{task.name}') missed deadline.")
                     # Penalty scaled by priority and how much work was remaining
                     remaining_work_ratio = max(0.0, 1.0 - task.status) # Ensure ratio is not negative
                     penalty = -10.0 * task.priority * remaining_work_ratio # Example penalty
                     reward += penalty # Add penalty to the final reward

             info['episode_outcome'] = 'completed' if all_tasks_completed else 'horizon_ended'
             info['num_missed_deadlines'] = num_missed_deadlines


        # --- Get Next State ---
        # This is the state corresponding to the NEW current_time_slot_index
        next_state = self.get_state()

        # Return the results of the step - Use Gymnasium's 5-element return
        # <--- MODIFICATION: Update return value format for Gymnasium ---
        return next_state, reward, terminated, truncated, info



    # Optional: Implement render method if visualization is needed
    # Add this method inside your TaskSchedulingEnv class definition

def render(self, mode='human'):
    """
    Renders the current state of the environment as text to the console.
    """
    if mode != 'human':
        # In a real environment, you might return an image array or string here
        super().render(mode=mode)
        return

    print("\n" + "="*40)
    print(f"Current Time Slot: {self.current_time_slot_index}")
    print(f"Current Datetime: {self._get_current_datetime().strftime('%Y-%m-%d %H:%M')}")
    print(f"Total Time Slots: {self.num_time_slots}")
    print("="*40)

    # --- Visualize Time Slots ---
    print("\nTime Slot Status:")
    status_line = ""
    # Characters to represent slot status:
    # . : Free
    # O : Occupied
    # X : Unavailable
    # > : Current Slot Marker

    # Let's show a window around the current slot, or the whole horizon if small
    window_size = 20 # Number of slots to show around current slot
    start_idx = max(0, self.current_time_slot_index - window_size // 2)
    end_idx = min(self.num_time_slots, start_idx + window_size)
    # Adjust start_idx if end_idx hits the limit
    if end_idx == self.num_time_slots:
        start_idx = max(0, end_idx - window_size)


    # Optional: Print header row with indices
    header_indices = " " * 5 # Offset for label
    markers = " " * 5 # Offset for label
    indices_line = " " * 5
    for i in range(start_idx, end_idx):
         if i % 5 == 0: # Print index every 5 slots
             indices_line += f"{i:<5}"[:5] # Print index, pad to 5, truncate if needed
         else:
             indices_line += " " * 5
         header_indices += "     " # Space for the status character below


    print(header_indices)
    print(indices_line)
    print("-" * (5 + (end_idx - start_idx) * 1)) # Separator line based on number of slots shown


    # Print status characters
    status_line = "Slots:"
    for i in range(start_idx, end_idx):
        char = "." # Default to free
        if i in self.unavailable_slots_indices:
            char = "X" # Unavailable overrides free/occupied
        elif i in self.scheduled_slots:
            char = "O" # Occupied

        if i == self.current_time_slot_index:
             status_line += f"[{char}]" # Mark the current slot
        else:
             status_line += f" {char} " # Regular slot


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
            if task: # Ensure task exists (should always be true if ID from registry keys)
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
                    f"Status: {task.status:.2f}/{task.duration:.2f} ({task.status*100:.0f}%) " # Show current work / total duration (% complete)
                    f"| Time Spent: {task.time_spent:.2f} "
                    f"| Deadline: {task.deadline.strftime('%Y-%m-%d %H:%M')} {'(PASSED)' if deadline_passed_now and not task.is_completed() else ''} "
                    f"| Dependencies Met: {'Yes' if dependencies_met else 'No'} "
                    f"| Completed: {'Yes' if task.is_completed() else 'No'}"
                )

    print("="*40 + "\n")


