t# The agent learning over time
- States
    - List of pending tasks
    - Their Priorities , deadlines, and estimated durations.
    - current date and time
    - Task dependencies from the DAG
    - Past scheduling outcomes (optionally)

- Actions
    - Which task to schedule next
    - How much time to allocate to a task today or in the future.

- Rewards
    - Positive rewards for completing a task before its deadline.
    - Penalty for missing deadline or leaving idle time when tasks are pending.
    