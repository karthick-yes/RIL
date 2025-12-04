from datetime import datetime
from typing import List, Dict, Optional, Set
import networkx as nx


# --- Task Class ---
class Task:
    # Class variables shared among all Task instances
    registry: Dict[int, "Task"] = {}  # Stores all tasks, keyed by ID
    next_id_counter: int = 1  # Tracks the next available unique ID for new tasks

    def __init__(
        self,
        name: str,
        ID: int,
        priority: int,
        deadline: str,
        duration: float,
        dependencies: List[int],
        source_file: str,
        status: float = 0.0,
        time_spent: float = 0.0,
    ):
        self.name = name
        self.ID = ID
        self.priority = priority

        # --- Deadline Parsing ---
        # Store deadline as a datetime object
        if isinstance(
            deadline, str
        ):  # Handle cases where deadline might already be datetime
            try:
                self.deadline = datetime.fromisoformat(deadline)
            except ValueError:
                print(
                    f"Warning: Invalid deadline format '{deadline}' for task '{name}'. Using default past date."
                )
                self.deadline = datetime.fromisoformat(
                    "1970-01-01T00:00:00"
                )  # Default past date
        elif isinstance(deadline, datetime):
            self.deadline = deadline
        else:
            print(
                f"Warning: Unexpected deadline type {type(deadline)} for task '{name}'. Using default past date."
            )
            self.deadline = datetime.fromisoformat("1970-01-01T00:00:00")

        self.duration = duration
        self.dependencies = dependencies

        self.time_spent = time_spent
        # how do we do --- Status Calculation ---
        # Calculate status based on time_spent / duration. Clamp between 0 and 1.
        # Handle duration = 0 case: status is 1.0 if time_spent > 0, else 0.0.
        if self.duration > 0:
            self.status = self.time_spent / self.duration
        else:
            self.status = 1.0 if self.time_spent > 0 else 0.0
        self.status = max(
            0.0, min(self.status, 1.0)
        )  # Clamp status between 0.0 and 1.0

        self.source_file = source_file  # Track the origin file

        # --- Weights for priority scoring ( option souhld be adjusted) ---
        # These weights are used in the computePriorityScore method
        # They determine the relative importance of different factors.
        self.w1, self.w2, self.w3, self.w4, self.w5, self.w6 = (
            0.166,
            0.166,
            0.166,
            0.166,
            0.166,
            0.166,
        )
        # Sum of weights is approximately 1.0, which is common for normalization.

        # --- Add task instance to the class registry ---
        if self.ID in Task.registry:
            # This handles potential ID conflicts if manually assigning IDs
            print(
                f"Warning: Task with ID {self.ID} already exists in registry. Overwriting."
            )
        Task.registry[self.ID] = self  # Add this instance to the shared registry

        # --- Update next ID counter ---
        # Ensure next_id_counter is always greater than the highest existing ID + 1
        Task.next_id_counter = max(Task.next_id_counter, self.ID + 1)

    # --- Class Methods (operate on the class or registry) ---

    @classmethod
    def get_task_by_id(cls, task_id: int) -> Optional["Task"]:
        """Retrieves a task instance from the registry by its ID."""
        return cls.registry.get(task_id)  # .get() returns None if key is not found

    @classmethod
    def get_task_by_name(cls, name: str) -> Optional["Task"]:
        """Retrieves a task instance from the registry by its name (searches linearly)."""
        for task in cls.registry.values():
            if task.name == name:
                return task
        return None  # Return None if no task with that name is found

    @classmethod
    def generate_next_id(cls) -> int:
        """Generates a unique sequential ID and increments the counter."""
        current_id = cls.next_id_counter
        cls.next_id_counter += 1
        return current_id

    @classmethod
    def remove_task(cls, task_id: int):
        """Safely remove a task instance from the registry by its ID."""
        if task_id in cls.registry:
            # Note: Removing a task doesn't automatically update dependencies in other tasks.
            # This might need additional logic depending on desired behavior.
            del cls.registry[task_id]
            print(f"Task ID {task_id} removed from registry.")
        else:
            print(f"Warning: Tried to remove Task ID {task_id}, but it was not found.")

    # --- Instance Methods (operate on a specific task instance) ---

    def update_from_dict(self, task_dict: Dict, source_file: str) -> None:
        """Update task properties from a dictionary of values."""
        self.source_file = source_file  # Always update source file

        # Safely update properties if they exist in the dictionary and are valid types
        if "priority" in task_dict:
            try:
                self.priority = int(task_dict["priority"])
            except (ValueError, TypeError):
                print(
                    f"Warning: Invalid priority format '{task_dict.get('priority')}' for task '{self.name}'. Keeping old value: {self.priority}"
                )

        if "deadline" in task_dict:
            # Handle both string and datetime objects if needed, assuming ISO format string input
            deadline_val = task_dict["deadline"]
            if isinstance(deadline_val, str):
                try:
                    self.deadline = datetime.fromisoformat(deadline_val)
                except ValueError:
                    print(
                        f"Warning: Invalid deadline format '{deadline_val}' for task '{self.name}'. Keeping old value: {self.deadline}"
                    )
            elif isinstance(deadline_val, datetime):
                self.deadline = deadline_val
            else:
                print(
                    f"Warning: Unexpected deadline type {type(deadline_val)} for task '{self.name}'. Keeping old value: {self.deadline}"
                )

        if "duration" in task_dict:
            try:
                self.duration = float(task_dict["duration"])
                # Recalculate status if duration changes (unless status/time_spent are also updated)
                recalculate_status = True
            except (ValueError, TypeError):
                print(
                    f"Warning: Invalid duration format '{task_dict.get('duration')}' for task '{self.name}'. Keeping old value: {self.duration}"
                )
                recalculate_status = False  # Duration didn't change validly

        # Update status and time_spent ONLY if provided
        update_status_explicitly = False
        if "status" in task_dict:
            try:
                self.status = float(task_dict["status"])
                self.status = max(0.0, min(self.status, 1.0))  # Clamp
                update_status_explicitly = True  # Status was explicitly provided
            except (ValueError, TypeError):
                print(
                    f"Warning: Invalid status format '{task_dict.get('status')}' for task '{self.name}'."
                )

        if "time_spent" in task_dict:
            try:
                self.time_spent = float(task_dict["time_spent"])
                self.time_spent = max(0.0, self.time_spent)  # Ensure non-negative
                # If time_spent is updated, recalculate status unless status was also explicitly provided
                if not update_status_explicitly:
                    if self.duration > 0:
                        self.status = self.time_spent / self.duration
                    else:
                        self.status = (
                            1.0 if self.time_spent > 0 else 0.0
                        )  # Handle duration 0
                    self.status = max(0.0, min(self.status, 1.0))  # Clamp
            except (ValueError, TypeError):
                print(
                    f"Warning: Invalid time_spent format '{task_dict.get('time_spent')}' for task '{self.name}'."
                )

        # If duration changed but not status/time_spent explicitly, recalculate status
        # This was already handled inside the time_spent block or after if update_status_explicitly was false
        # Simplified logic: if status wasn't explicitly given, recalculate based on new time_spent/duration
        if not update_status_explicitly and (
            "duration" in task_dict or "time_spent" in task_dict
        ):
            if self.duration > 0:
                self.status = self.time_spent / self.duration
            else:
                self.status = 1.0 if self.time_spent > 0 else 0.0  # Handle duration 0
            self.status = max(0.0, min(self.status, 1.0))  # Clamp

        # Dependencies are typically handled in a separate pass after all tasks are created

    def is_completed(self) -> bool:
        """Checks if the task is completed (status >= 1.0)."""
        return self.status >= 1.0

    def get_dependency_ids(self) -> List[int]:
        """Returns the list of task IDs this task depends on."""
        return self.dependencies

    # --- Dynamic Priority Score Calculation ---
    # This method was updated to accept current_datetime and use passed scaling factors
    def computePriorityScore(
        self,
        current_datetime: datetime,  # Accepts the current simulation time
        max_horizon: float,  # Scaling factor for deadline urgency (total horizon days, > 0)
        max_priority: int,  # Scaling factor for base priority
        max_duration: float,  # Scaling factor for duration
        max_deps: int,  # Scaling factor for dependency count
    ) -> float:
        """
        Calculates a dynamic priority score for the task based on current time and context.

        Args are scaling factors/contextual info from the environment.

        Returns:
            A float score between 0.0 and 1.0.
        """
        # --- Deadline Urgency ---
        # Calculate days remaining using the passed current_datetime
        # Use total_seconds for more precision than .days
        time_until_deadline_seconds = (self.deadline - current_datetime).total_seconds()
        days_to_deadline = time_until_deadline_seconds / (
            24 * 3600.0
        )  # Convert to days as float

        # Ensure max_horizon is positive for division (guaranteed by environment.py now, but defensive)
        if max_horizon <= 0:
            deadline_urgency = (
                1.0  # Maximum urgency if horizon is invalid or zero scale
            )
        else:
            # Clamp days_to_deadline for urgency calculation.
            # 0 means due now or past, max_horizon means no urgency from deadline factor.
            # The clamping needs to be relative to the total horizon scale.
            # A task due far in the future (days_to_deadline > max_horizon) should have 0 urgency from deadline.
            # A task due now or in the past (days_to_deadline <= 0) should have max urgency (1.0).
            # Interpolate between 0 urgency (at max_horizon days) and 1 urgency (at 0 days or less).
            # Urgency decreases linearly from 1.0 to 0.0 as days_to_deadline goes from 0 to max_horizon.
            # days_to_deadline > max_horizon -> clamped to max_horizon
            # days_to_deadline <= 0 -> clamped to 0
            clamped_days_for_urgency = max(0.0, min(days_to_deadline, max_horizon))

            # Normalized clamped days: 0 (most urgent) to 1 (least urgent wrt deadline)
            normalized_clamped_days = clamped_days_for_urgency / max_horizon

            # Deadline urgency: 1.0 (most urgent) to 0.0 (least urgent)
            deadline_urgency = 1.0 - normalized_clamped_days

        # --- Base Priority ---
        # Normalize priority using the passed max_priority (clamp between 0 and 1)
        base_priority = (
            min(self.priority / max_priority, 1.0) if max_priority > 0 else 0.0
        )

        # --- Dependency Factor ---
        # Consider the count and priority of dependencies
        num_deps = len(self.dependencies)
        # Normalize number of dependencies using the passed max_deps scale
        normalized_num_deps = min(num_deps / max_deps, 1.0) if max_deps > 0 else 0.0

        dep_priorities = []
        for dep_id in self.dependencies:
            dep_task = Task.get_task_by_id(dep_id)
            if dep_task:
                # Normalize dependency priority using the passed max_priority scale
                dep_priorities.append(
                    min(dep_task.priority / max_priority, 1.0)
                    if max_priority > 0
                    else 0.0
                )

        # Average normalized priority of dependencies
        avg_norm_priority_deps = (
            sum(dep_priorities) / len(dep_priorities) if dep_priorities else 0.0
        )

        # Combine number of dependencies and their average priority (example weighting)
        dependency_factor = (0.4 * normalized_num_deps) + (0.6 * avg_norm_priority_deps)
        dependency_factor = max(
            0.0, min(dependency_factor, 1.0)
        )  # Clamp combined factor

        # --- Completion Status ---
        # Score higher for tasks that are less complete (more work remaining)
        completion_status_factor = 1.0 - max(
            0.0, min(self.status, 1.0)
        )  # Ensure self.status is clamped 0-1

        # --- Duration Factor ---
        # Shorter tasks might be prioritized slightly higher (quicker wins) - inverted normalization
        # Use the passed max_duration scale for normalization
        normalized_duration = (
            min(self.duration / max_duration, 1.0) if max_duration > 0 else 0.0
        )
        duration_factor = 1.0 - normalized_duration
        duration_factor = max(0.0, min(duration_factor, 1.0))  # Clamp

        # --- Subtask Influence (Simplified - based on direct dependencies' urgency & priority) ---
        # This attempts to boost the priority of a task if its dependencies are urgent or high priority.
        subtask_scores = []
        for dep_id in self.dependencies:
            subtask = Task.get_task_by_id(dep_id)
            if subtask:
                # Recalculate subtask urgency and priority relative to the SAME current_datetime and max_horizon scale
                # Use the passed current_datetime and max_horizon for subtask urgency calculation
                sub_time_until_deadline_seconds = (
                    subtask.deadline - current_datetime
                ).total_seconds()
                sub_days = sub_time_until_deadline_seconds / (24 * 3600.0)

                if max_horizon <= 0:  # Defensive check
                    sub_urgency = 1.0
                else:
                    sub_clamped = max(0.0, min(sub_days, max_horizon))
                    sub_urgency = 1.0 - (sub_clamped / max_horizon)

                sub_base_prio = (
                    min(subtask.priority / max_priority, 1.0)
                    if max_priority > 0
                    else 0.0
                )  # Normalize subtask priority
                subtask_scores.append(
                    (sub_urgency + sub_base_prio) / 2.0
                )  # Simple average of two factors for dependency's influence

        subtask_influence = (
            sum(subtask_scores) / len(subtask_scores) if subtask_scores else 0.0
        )
        subtask_influence = max(0.0, min(subtask_influence, 1.0))  # Clamp influence

        # --- Weighted Sum ---
        # Combine all factors using weights. Weights determine relative importance.
        # Ensure weights sum to 1.0 if intended (0.166 * 6 is approx 0.996)
        score = (
            self.w1 * deadline_urgency
            + self.w2 * base_priority
            + self.w3 * dependency_factor
            + self.w4 * completion_status_factor
            + self.w5 * duration_factor
            + self.w6 * subtask_influence
        )

        # Clamp the final score to be between 0.0 and 1.0.
        final_score = max(0.0, min(score, 1.0))

        return final_score

    # --- Class Methods for Updating Tasks (operate on the registry) ---

    @classmethod
    def update_task_status(cls, task_id: int, status: float) -> bool:
        """Updates the status of a task by its ID in the registry."""
        task = cls.get_task_by_id(task_id)
        if task:
            task.status = max(0.0, min(status, 1.0))  # Clamp status
            # Note: This direct status update does NOT automatically change time_spent.
            # If you update status directly, ensure consistency with time_spent elsewhere if needed.
            print(
                f"Task '{task.name}' (ID: {task_id}) status updated to {task.status:.2f}"
            )
            return True
        else:
            print(f"Task with ID {task_id} not found for status update.")
            return False

    @classmethod
    def update_time_spent(cls, task_id: int, time_spent: float) -> bool:
        """Updates the time spent on a task and recalculates its status."""
        task = cls.get_task_by_id(task_id)
        if task:
            task.time_spent = max(0.0, time_spent)  # Ensure non-negative time spent
            # Recalculate status based on new time_spent and duration
            if task.duration > 0:
                task.status = task.time_spent / task.duration
            else:
                task.status = 1.0 if task.time_spent > 0 else 0.0  # Handle duration 0
            task.status = max(0.0, min(task.status, 1.0))  # Clamp status
            print(
                f"Task '{task.name}' (ID: {task_id}) time spent updated to {task.time_spent:.2f}, status recalculated to {task.status:.2f}"
            )
            return True
        else:
            print(f"Task with ID {task_id} not found for time spent update.")
            return False
