# dag_builder.py
import networkx as nx
from typing import Dict
from typing import List, Dict, Optional, Set
from datetime import datetime
from task import Task

def build_task_dag(task_registry: Dict[int, Task]) -> nx.DiGraph:
    """
    Builds a Directed Graph (DiGraph) from the task registry,
    representing task dependencies.
    """
    G = nx.DiGraph()

    # Add nodes (tasks)
    for task_id, task in task_registry.items():
        G.add_node(task_id, name=task.name, priority=task.priority,
                   deadline=task.deadline, duration=task.duration,
                   status=task.status, time_spent=task.time_spent,
                   source_file=task.source_file)

    # Add edges (dependencies)
    for task_id, task in task_registry.items():
        for dep_id in task.dependencies:
            if dep_id in task_registry: # Ensure dependency task exists in registry
                G.add_edge(dep_id, task_id)
            else:
                # In a separate module, I might  log this or handle differently
                print(f"Warning: Dependency ID {dep_id} for task {task_id} ('{task.name}') not found in registry when building DAG. Edge skipped.")

    # Check if the graph is a DAG
    if not nx.is_directed_acyclic_graph(G):
        print("\n!!! Warning: The task dependency graph contains a cycle. This is not a valid DAG. !!!")
        # You might want to raise an error or handle this case (e.g., report the cycle)
        try:
            cycles = list(nx.simple_cycles(G))
            print("Cycles found:", cycles)
        except nx.NetworkXNoCycle:
             # Should not happen if is_directed_acyclic_graph was False
             pass
    return G
