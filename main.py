import requests
from datetime import datetime
import json
import re
from typing import List, Dict, Optional, Set
import urllib3
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import pickle
from dag_builder import build_task_dag
import networkx as nx
from task import Task
import matplotlib.pyplot as plt

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# --- TaskParser Class ---
class TaskParser:
    # Use environment variables or a config file for sensitive info like Bearer tokens
    AUTH_TOKEN = "25d97ac0d39b00e3f3990c32103eb8953b958b6f1819f00fa0dac6092a315022"  # Consider moving this
    BASE_URL = "https://127.0.0.1:27124"  # Obsidian Local REST API base URL

    def __init__(self):
        self.headers = {
            "accept": "application/vnd.olrapi.note+json",
            "Authorization": f"Bearer {self.AUTH_TOKEN}",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.session.verify = False  # Disable SSL verification (use carefully)

    # Inside TaskParser class
    def _make_request(
        self, method: str, endpoint: str, **kwargs
    ) -> Optional[requests.Response]:
        url = f"{self.BASE_URL}{endpoint}"
        # --- ADD TIMEOUT ---
        default_timeout = 15  # Increased timeout (seconds) - adjust if needed
        request_timeout = kwargs.pop("timeout", default_timeout)
        # --- END ADD TIMEOUT ---
        print(
            f"...... Making {method} request to {url} with timeout={request_timeout}s"
        )  # DEBUG PRINT
        try:
            # Pass timeout to the request call
            response = self.session.request(
                method, url, timeout=request_timeout, **kwargs
            )
            print(
                f"...... Request completed with status: {response.status_code}"
            )  # DEBUG PRINT
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response
        except requests.exceptions.Timeout:  # Catch the specific Timeout exception
            print(
                f"Error: Request timed out after {request_timeout} seconds for {url}."
            )
            return None  # Ensure None is returned on timeout
        except requests.exceptions.ConnectionError as e:
            print(
                f"Error: Connection refused. Is Obsidian running and Local REST API plugin enabled at {self.BASE_URL}? Details: {e}"
            )
            return None  # Ensure None is returned
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else "Unknown"
            text = e.response.text if e.response else "No Response Body"
            print(f"Error: HTTP Error {status_code} for {url}. Response: {text}")
            return None  # Ensure None is returned
        except requests.exceptions.RequestException as e:
            print(
                f"Error: An unexpected error occurred during the request to {url}. Details: {e}"
            )
            return None  # Ensure None is returned
        # Ensure None is returned if try block finishes without returning response (shouldn't happen often)
        return None

    def connect_and_get_content(
        self, relative_path: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Fetches content for a specific relative path or the current daily note.
        Tries to create the daily note if fetching the default fails.
        Returns the parsed JSON content as a dictionary.
        """
        if relative_path:
            # Fetch specific note by vault path
            endpoint = f"/vault/{relative_path}"
            response = self._make_request("GET", endpoint)
            if response:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    print(
                        f"Error: Could not decode JSON response from {endpoint}. Content: {response.text}"
                    )
                    return None  # Return None if JSON is invalid
            else:
                # If fetch fails for specific path, don't try to create it, just return None
                print(f"Failed to fetch specific note at path: {relative_path}")
                return None
        else:
            # Fetch current daily note
            daily_endpoint = "/periodic/daily/"
            response = self._make_request("GET", daily_endpoint)
            if response:
                try:
                    return response.json()
                except json.JSONDecodeError:
                    print(
                        f"Error: Could not decode JSON response from {daily_endpoint}. Content: {response.text}"
                    )
                    return None
            else:
                # If fetching daily note fails, try creating it
                print("Failed to fetch current daily note, attempting to create...")
                create_endpoint = "/commands/daily-notes/"
                create_response = self._make_request("POST", create_endpoint)
                if create_response:
                    print("Daily note potentially created. Retrying fetch...")
                    time.sleep(1)  # Give Obsidian a moment
                    retry_response = self._make_request("GET", daily_endpoint)
                    if retry_response:
                        try:
                            return retry_response.json()
                        except json.JSONDecodeError:
                            print(
                                f"Error: Could not decode JSON response from {daily_endpoint} after creation. Content: {retry_response.text}"
                            )
                            return None
                    else:
                        print(
                            "Failed to fetch daily note even after attempting creation."
                        )
                        return None
                else:
                    print("Failed to create daily note.")
                    return None

    def parse_task_line(self, task_line: str) -> Optional[Dict[str, str]]:
        """
        Parse a task line like '- [ ] name: Task A, priority: 5, ...' into a dictionary.
        Handles potential commas within the name field more robustly.
        """
        task_dict = {}
        # Regex to find key:value pairs, allowing spaces around ':' and commas in value
        # It captures the key and the value up to the next key or end of string
        pattern = re.compile(r"([\w]+)\s*:\s*(.*?)(?=\s*,\s*\w+\s*:|\s*$)")
        matches = pattern.findall(task_line)

        if not matches:
            print(
                f"Warning: Could not parse any key-value pairs in line: '{task_line}'"
            )
            return None

        for key, value in matches:
            task_dict[key.strip()] = value.strip()

        # Basic validation: Check if 'name' exists
        if "name" not in task_dict or not task_dict["name"]:
            print(f"Warning: Task line missing 'name' or name is empty: '{task_line}'")
            return None

        # Provide defaults for optional fields if they are missing
        task_dict.setdefault("priority", "1")  # Default priority
        task_dict.setdefault("deadline", "1970-01-01T00:00:00")  # Default deadline
        task_dict.setdefault("duration", "0.0")  # Default duration
        task_dict.setdefault("subTasksname", "")  # Default dependencies (empty)
        task_dict.setdefault("status", None)  # Default status (let Task calculate)
        task_dict.setdefault("time_spent", "0.0")  # Default time spent

        # Handle explicit status if duration is 0
        if task_dict.get("status") is None and float(task_dict["duration"]) == 0:
            task_dict["status"] = (
                "1.0"  # Assume tasks with 0 duration are completed unless time spent is 0
            )

        # Remove status if time_spent is present, let Task class calculate
        if "time_spent" in task_dict and task_dict["time_spent"] != "0.0":
            task_dict.pop(
                "status", None
            )  # Remove explicit status if time spent is given

        return task_dict

    def parse_content_and_update_registry(
        self, content_json: Dict, relative_file_path: str
    ):
        """
        Parses tasks from the provided content JSON of a specific file
        and updates the global Task.registry accordingly.
        - Adds new tasks found in the file.
        - Updates existing tasks found in the file.
        - Removes tasks from the registry if they were associated with this file but are no longer present.
        """
        print(f"\n--- Parsing file: {relative_file_path} ---")
        if not content_json or "content" not in content_json:
            print("Error: Invalid or empty content_json received.")
            return

        parsed_content = content_json["content"]
        # Regex to find "#### Daily Tasks" section more reliably, handling optional whitespace
        daily_tasks_section = re.search(
            r"^\#{4}\s+Daily Tasks\s*(.*?)(?:^\s*?-{3,}\s*$|^\#{1,4}\s+|\Z)",
            parsed_content,
            re.MULTILINE | re.DOTALL,
        )

        if not daily_tasks_section:
            print(
                f"Info: '#### Daily Tasks' section not found in {relative_file_path}. No tasks parsed from this file."
            )
            tasks_in_this_file = []  # No tasks found
        else:
            section_text = daily_tasks_section.group(1)
            # Find lines starting with '- [ ]' (unchecked task)
            task_lines = re.findall(
                r"^\s*-\s*\[\s\]\s*(.+)", section_text, re.MULTILINE
            )
            print(f"Found {len(task_lines)} task lines in Daily Tasks section.")
            tasks_in_this_file = [self.parse_task_line(line) for line in task_lines]
            tasks_in_this_file = [
                task_dict for task_dict in tasks_in_this_file if task_dict is not None
            ]  # Filter out lines that failed parsing

        # --- Pass 1: Add/Update Tasks ---
        task_names_found_in_file: Set[str] = set()
        tasks_processed_this_run: Dict[
            str, Task
        ] = {}  # Store tasks processed in this specific file run

        for task_dict in tasks_in_this_file:
            name = task_dict["name"]
            task_names_found_in_file.add(name)

            existing_task = Task.get_task_by_name(name)

            if existing_task:
                print(
                    f"Updating existing task: '{name}' (ID: {existing_task.ID}) from file {relative_file_path}"
                )
                existing_task.update_from_dict(task_dict, relative_file_path)
                tasks_processed_this_run[name] = existing_task
            else:
                # Create new task
                try:
                    priority = int(task_dict.get("priority", "1"))
                    deadline = task_dict.get("deadline", "1970-01-01T00:00:00")
                    duration = float(task_dict.get("duration", "0.0"))
                    # Status and time_spent are handled by update_from_dict logic or init defaults
                    status_str = task_dict.get("status")
                    time_spent_str = task_dict.get("time_spent", "0.0")

                    # Determine initial status/time_spent carefully
                    time_spent = float(time_spent_str)
                    status = None
                    if status_str is not None:
                        status = float(status_str)
                    # else status will be calculated in __init__ based on time_spent/duration

                    new_id = Task.generate_next_id()
                    print(
                        f"Creating new task: '{name}' (ID: {new_id}) from file {relative_file_path}"
                    )
                    task = Task(
                        name=name,
                        ID=new_id,
                        priority=priority,
                        deadline=deadline,
                        duration=duration,
                        dependencies=[],  # Dependencies added in Pass 2
                        source_file=relative_file_path,
                        status=status
                        if status is not None
                        else 0.0,  # Pass explicit status if provided
                        time_spent=time_spent,
                    )
                    # If status was not explicitly provided, recalculate it now after init
                    if status is None:
                        task.status = (
                            task.time_spent / task.duration
                            if task.duration > 0
                            else (1.0 if task.time_spent > 0 else 0.0)
                        )

                    tasks_processed_this_run[name] = task

                except (ValueError, KeyError) as e:
                    print(f"Error creating task from dict {task_dict}: {e}")

        # --- Pass 2: Update Dependencies for tasks processed in this run ---
        # We iterate through the *original* parsed data again to find dependency names
        for task_dict in tasks_in_this_file:
            name = task_dict.get("name")
            if not name or name not in tasks_processed_this_run:
                continue  # Skip if task wasn't successfully created/updated or has no name

            current_task = tasks_processed_this_run[name]
            subTasksname_str = task_dict.get("subTasksname", "")
            new_dependency_ids = []
            if subTasksname_str:
                subtask_names = [
                    n.strip() for n in subTasksname_str.split(",") if n.strip()
                ]
                for sub_name in subtask_names:
                    dependency_task = Task.get_task_by_name(sub_name)
                    if dependency_task:
                        new_dependency_ids.append(dependency_task.ID)
                    else:
                        print(
                            f"Warning: Dependency task '{sub_name}' listed for '{name}' not found in registry."
                        )

            # Only update if dependencies changed to avoid unnecessary logs/updates
            if set(current_task.dependencies) != set(new_dependency_ids):
                print(
                    f"Updating dependencies for task '{name}' (ID: {current_task.ID}) to: {[Task.get_task_by_id(id).name for id in new_dependency_ids] if new_dependency_ids else 'None'}"
                )
                current_task.dependencies = new_dependency_ids

        # --- Pass 3: Remove Tasks Associated ONLY With This File If Not Found ---
        task_ids_to_remove: List[int] = []
        # Check all tasks in the registry
        for task_id, task in list(
            Task.registry.items()
        ):  # Use list to avoid modification issues during iteration
            # If a task's source is THIS file, but its name wasn't found in THIS parse...
            if (
                task.source_file == relative_file_path
                and task.name not in task_names_found_in_file
            ):
                print(
                    f"Task '{task.name}' (ID: {task_id}) associated with {relative_file_path} but not found in current parse. Marking for removal."
                )
                task_ids_to_remove.append(task_id)

        # Perform removal after iteration
        for task_id in task_ids_to_remove:
            Task.remove_task(task_id)

        print(f"--- Finished parsing file: {relative_file_path} ---")
        # Optional: Print current registry state after update
        self.print_registry_summary()

    def print_registry_summary(self):
        print("\n--- Current Task Registry Summary ---")
        if not Task.registry:
            print("Registry is empty.")
            return
        for task_id, task in sorted(Task.registry.items()):
            dep_names = [
                Task.get_task_by_id(dep_id).name
                for dep_id in task.dependencies
                if Task.get_task_by_id(dep_id)
            ]
            print(
                f"  ID: {task_id}, Name: '{task.name}', Prio: {task.priority}, Deadline: {task.deadline.strftime('%Y-%m-%d')}, "
                f"Status: {task.status:.2f}, Source: '{task.source_file}', DependsOn: {dep_names or 'None'}"
            )
        print("-------------------------------------\n")


# --- TaskWatcher Class ---
class TaskWatcher(FileSystemEventHandler):
    def __init__(self, task_parser: TaskParser, vault_dir: str):
        self.task_parser = task_parser
        self.last_processed_event = {}  # Store last processed time per file path
        self.debounce_time = 2  # 2 seconds debounce per file
        self.vault_dir = vault_dir
        print(f"TaskWatcher initialized for directory: {vault_dir}")

    def _get_relative_path(self, src_path: str) -> Optional[str]:
        """Calculates the relative path suitable for the API."""
        try:
            # Ensure the path is absolute before calculating relative path
            abs_src_path = os.path.abspath(src_path)
            abs_vault_dir = os.path.abspath(self.vault_dir)
            if not abs_src_path.startswith(abs_vault_dir):
                print(
                    f"Warning: Modified path {abs_src_path} is outside vault directory {abs_vault_dir}. Skipping."
                )
                return None

            rel_path = os.path.relpath(abs_src_path, abs_vault_dir)
            # Convert to forward slashes for URL/API
            return rel_path.replace(os.path.sep, "/")
        except ValueError as e:
            print(
                f"Error calculating relative path for {src_path} relative to {self.vault_dir}: {e}"
            )
            return None

    def _should_process(self, event_path: str) -> bool:
        """Applies debounce logic per file path."""
        current_time = time.time()
        last_time = self.last_processed_event.get(event_path, 0)
        if current_time - last_time < self.debounce_time:
            # print(f"Debounced event for {event_path}") # Optional: Verbose logging
            return False
        self.last_processed_event[event_path] = current_time
        return True

    def _handle_event(self, event_path: str, event_type: str):
        """Common handler for modified and created events."""
        print(f"\n{event_type} detected: {event_path}")
        # Basic check for markdown files
        if not event_path.lower().endswith(".md"):
            # print(f"Ignoring non-markdown file: {event_path}")
            return

        # Check if it's a "daily" note (adjust pattern if needed)
        # This check is simple, might need refinement based on vault structure
        if "daily" not in event_path.lower():
            # print(f"Ignoring non-daily note: {event_path}")
            return

        if not self._should_process(event_path):
            return

        print(f"---> Processing {event_type} for daily note: {event_path}")
        rel_path = self._get_relative_path(event_path)
        if not rel_path:
            print(f"Could not determine relative path for {event_path}. Skipping.")
            return

        print(f"---> Relative path: {rel_path}")
        try:
            note_json = self.task_parser.connect_and_get_content(relative_path=rel_path)
            if note_json:
                print(f"---> Successfully fetched content for {rel_path}")
                self.task_parser.parse_content_and_update_registry(note_json, rel_path)
                print(f"---> Finished processing {rel_path}")
            else:
                print(f"---> Failed to fetch or parse content for {rel_path}")
        except Exception as e:
            # Catch unexpected errors during handling
            print(f"*** CRITICAL ERROR processing {event_path}: {e} ***")
            import traceback

            traceback.print_exc()  # Print full traceback for debugging

    def on_modified(self, event):
        if not event.is_directory:
            self._handle_event(event.src_path, "Modification")

    def on_created(self, event):
        if not event.is_directory:
            self._handle_event(event.src_path, "Creation")

    # Optional: Handle deletion/moves if needed
    # def on_deleted(self, event):
    #     if not event.is_directory and "daily" in event.src_path.lower() and event.src_path.lower().endswith('.md'):
    #         rel_path = self._get_relative_path(event.src_path)
    #         if rel_path:
    #              print(f"Deletion detected for daily note: {rel_path}. Removing associated tasks...")
    #              ids_to_remove = [task_id for task_id, task in Task.registry.items() if task.source_file == rel_path]
    #              for task_id in ids_to_remove:
    #                   Task.remove_task(task_id)
    #              self.task_parser.print_registry_summary()

    # def on_moved(self, event):
    #      if not event.is_directory and "daily" in event.dest_path.lower() and event.dest_path.lower().endswith('.md'):
    #          old_rel_path = self._get_relative_path(event.src_path)
    #          new_rel_path = self._get_relative_path(event.dest_path)
    #          if old_rel_path and new_rel_path:
    #                print(f"Move detected: {old_rel_path} -> {new_rel_path}. Updating source file for associated tasks...")
    #                for task in Task.registry.values():
    #                     if task.source_file == old_rel_path:
    #                          task.source_file = new_rel_path
    #                          print(f"  Updated source for task '{task.name}' (ID: {task.ID})")
    #                # Optional: Trigger a re-parse of the new location?
    #                # self._handle_event(event.dest_path, "Move/Reparse")


# --- Watcher Class ---
class Watcher:
    def __init__(self, directory: str, task_parser: TaskParser):
        if not os.path.isdir(directory):
            raise ValueError(f"Provided directory does not exist: {directory}")
        self.observer = Observer()
        # Make sure vault_dir passed to handler is absolute
        self.directory = os.path.abspath(directory)
        self.handler = TaskWatcher(task_parser, vault_dir=self.directory)

    def run(self):
        self.observer.schedule(self.handler, self.directory, recursive=True)
        self.observer.start()
        print(f"\n--- Watcher Started ---")
        print(f"Monitoring directory: {self.directory}")
        print(f"Looking for changes in '.md' files containing 'daily' in the path.")
        print(f"Press Ctrl+C to stop.")
        print(f"-----------------------\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received. Stopping watcher...")
            self.observer.stop()
        self.observer.join()
        print("\n--- Watcher Terminated ---\n")


# -- Add these save/load functions to your file ---
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
        print(f"Task registry file not found at {file_path}")
        return None

    try:
        with open(file_path, "rb") as f:
            registry = pickle.load(f)
        print(f"Task registry successfully loaded from {file_path}")
        # After loading, update the class registry and next_id_counter
        Task.registry = registry
        if registry:
            # Find the maximum existing ID and set next_id_counter
            max_id = max(registry.keys()) if registry else 0
            Task.next_id_counter = max_id + 1
        else:
            Task.next_id_counter = 1

        print(f"Task.registry updated. Next ID counter set to {Task.next_id_counter}")
        return registry
    except (IOError, pickle.UnpicklingError, EOFError) as e:
        print(f"Error loading task registry from {file_path}: {e}")
        return None
    except Exception as e:  # Catch any other unexpected errors during loading
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return None


# --- Main Execution ---
if __name__ == "__main__":
    # Initialize the task parser
    # Define pat h for the resgisty file
    registry_file_path = "task_registry.pkl"

    # -- Loading the registry on Startup -- #
    print("Attempting to load existing task registry")
    loaded_registry = load_task_registry(registry_file_path)

    if loaded_registry is None:
        print(
            "No existing registry found or failed to load. Starting with an empty registry."
        )
        # Ensure the Task class registry is cleared if loading failed
        Task.registry.clear()
        Task.next_id_counter = 1
    else:
        print("Task registry loaded successfully.")

    task_parser = TaskParser()

    if Task.registry:  # Only build if there are tasks in the registry
        print("\nBuilding the task dependency DAG...")
        task_dag = build_task_dag(Task.registry)
        print("Task DAG built.")
        print(f"Number of nodes: {task_dag.number_of_nodes()}")
        print(f"Number of edges: {task_dag.number_of_edges()}")

        # Add the Visualization code here
        print("Generating DAG visualization...")

        # Create a dictionary to map node IDs to task names for labels
        # This uses your actual Task.registry
        node_labels = {task_id: task.name for task_id, task in Task.registry.items()}

        plt.figure(figsize=(10, 8))  # Adjust figure size as needed
        pos = nx.spring_layout(task_dag)  # Use a layout

        # Draw the nodes
        nx.draw_networkx_nodes(
            task_dag, pos, node_size=3000, node_color="skyblue", alpha=0.9
        )

        # Draw the edges
        nx.draw_networkx_edges(
            task_dag,
            pos,
            edgelist=task_dag.edges(),
            edge_color="gray",
            arrows=True,
            arrowsize=20,
            width=2,
        )

        # Draw the labels
        nx.draw_networkx_labels(
            task_dag, pos, labels=node_labels, font_size=10, font_weight="bold"
        )

        plt.title("Task Dependency DAG")
        plt.axis("off")  # Hide axes
        plt.show()  # Display the plot

        print("DAG visualization displayed.")

        # Example: Perform topological sort again
        try:
            topological_order = list(nx.topological_sort(task_dag))
            print("\nTopological Sort (a valid order of execution):")
            # Map IDs back to task names for readability
            ordered_task_names = [
                Task.get_task_by_id(node_id).name
                for node_id in topological_order
                if Task.get_task_by_id(node_id)
            ]
            print(ordered_task_names)
        except nx.NetworkXUnfeasible:
            print(
                "\nCannot perform topological sort because the graph contains cycles."
            )

    else:
        print("\nNo tasks in registry, skipping DAG building.")

    # Fetch the current daily note first to ensure connection works & potentially populate initial tasks
    print("--- Performing Initial Check/Parse ---")
    initial_content = (
        task_parser.connect_and_get_content()
    )  # Fetches current daily note
    if initial_content and "path" in initial_content:
        initial_rel_path = initial_content["path"]
        print(f"Initial parse target: Current Daily Note ({initial_rel_path})")
        task_parser.parse_content_and_update_registry(initial_content, initial_rel_path)
    else:
        print(
            "Warning: Could not fetch or parse initial daily note. Registry might be empty initially."
        )

    task_parser.print_registry_summary()
    print("--------------------------------------\n")

    # --- Start the Watcher ---
    vault_directory = "D:\\test vault"

    if not os.path.isdir(vault_directory):
        print(
            f"ERROR: Vault directory '{vault_directory}' not found. Please update the path in the script."
        )
        # --- CONSIDER SAVING BEFORE EXITING ON ERROR ---
        print("Saving current registry before exiting due to vault error...")
        save_task_registry(registry_file_path, Task.registry)
        exit(1)

    watcher = Watcher(vault_directory, task_parser)

    try:
        print(f"\n--- Watcher Started ---")
        print(f"Monitoring directory: {watcher.directory}")
        print(f"Looking for changes in '.md' files containing 'daily' in the path.")
        print(f"Press Ctrl+C to stop.")
        watcher.run()
    except KeyboardInterrupt:
        print("\nWatcher interrupted by user. Shutting down gracefully...")
    finally:
        # --- Saving the registry on graceful shutdown ---
        print("Saving the current task registry on shutdown...")
        save_task_registry(registry_file_path, Task.registry)
        print("Shutdown complete.")

    # This line might not be reached if watcher.run() is blocking and not interrupted
    print("Script finished.")
