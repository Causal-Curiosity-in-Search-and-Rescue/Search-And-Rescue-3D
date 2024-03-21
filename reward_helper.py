import numpy as np 

def calculate_reward(current_position, current_step, goal_position, wall_positions, visited_states, max_steps,some_threshold=0.5):
    reward = 0  # Reset reward for each calculation

    # Distance to goal reward
    goal_distance = np.linalg.norm(np.array(goal_position) - np.array(current_position))
    max_possible_distance = np.sqrt(11**2 + 11**2 + 11**2)  # Adjust for your grid
    reward += (1 - goal_distance / max_possible_distance) * 10  # Adjust scaling factor as needed

    # Wall proximity penalty
    wall_distances = [np.linalg.norm(np.array(wall_pos) - np.array(current_position)) for wall_pos in wall_positions]
    min_wall_distance = min(wall_distances)
    if min_wall_distance < some_threshold:  # Define 'some_threshold' based on your grid scale
        reward -= (1 - min_wall_distance / some_threshold) * 5  # Adjust scaling factor as needed

    # Exploration incentive
    if tuple(current_position) not in visited_states:
        reward += 1  # Increment for visiting a new state
        visited_states.add(tuple(current_position))  # Add current position to visited states

    # Step penalty and urgency as steps approach max_steps
    step_penalty = -0.1  # Small penalty for each step
    steps_remaining = max_steps - current_step
    urgency_bonus = (1 - steps_remaining / max_steps) * 5  # Increase penalty as fewer steps remain
    reward += step_penalty - urgency_bonus

    return reward
