import heapq
from itertools import count

# Define the guard's state
class State:
    """state represantation"""
    def __init__(self, location, visited, time_used):
        self.location = location            # Where the guard is right now
        self.visited = tuple(sorted(visited))  # Rooms the guard has already been to
        self.time_used = time_used          # How much walking distance used so far

    def is_goal(self, required_rooms):
        """Check if all important rooms are visited"""
        return all(room in self.visited for room in required_rooms)

    def __hash__(self):
        """Make the state usable in dictionaries so we can track visited states."""
        return hash((self.location, self.visited, self.time_used))

    def __eq__(self, other):
        """Two states are the same if location, visited rooms, and distance are the same."""
        return (self.location, self.visited, self.time_used) == \
               (other.location, other.visited, other.time_used)


# Museum patrol problem
class MuseumPatrolProblem:
    """Holds the map of the museum, priority rooms, and the max shift distance."""
    def __init__(self, graph, priority_rooms, shift_limit=15):
        self.graph = graph
        self.priority_rooms = priority_rooms
        self.shift_limit = shift_limit      # Max distance guard can walk

    def initial_state(self):
        """Start at Security, no rooms visited, distance 0."""
        return State("Security", (), 0)

    def successors(self, state):
        """ Make a list of all next possible moves from current room.
        Only include moves that don't go over shift limit. """
        actions = []
        for neighbor, cost in self.graph.get(state.location, []):
            new_time = state.time_used + cost
            if new_time <= self.shift_limit:
                new_visited = set(state.visited)   # Make a copy of visited rooms
                new_visited.add(neighbor)          # Add the new room
                new_state = State(neighbor, tuple(new_visited), new_time)
                actions.append((f"Walk to {neighbor}", new_state, cost))  # Keep track of action
        return actions


# UCS
def uniform_cost_search(problem):
    """Explore paths by lowest total distance first."""
    start = problem.initial_state()
    frontier = []             # List of states to explore
    counter = count()         # Just a tie-breaker for equal-cost states
    heapq.heappush(frontier, (0, next(counter), start))
    reached = {start: 0}      # Keep track of best distance for each state
    nodes_expanded = 0        # Count how many states we looked at

    while frontier:
        cost, _, state = heapq.heappop(frontier)  # Pick the state with lowest total distance
        nodes_expanded += 1

        if state.is_goal(problem.priority_rooms):
            return cost, state, nodes_expanded

        for _, next_state, step_cost in problem.successors(state):
            new_cost = cost + step_cost
            if next_state not in reached or new_cost < reached[next_state]:
                reached[next_state] = new_cost
                heapq.heappush(frontier, (new_cost, next(counter), next_state))

    return None, None, nodes_expanded


# A* Search
def heuristic(state, problem):
    remaining = [r for r in problem.priority_rooms if r not in state.visited]
    if not remaining:
        return 0
    neighbor_costs = [
        cost for neighbor, cost in problem.graph.get(state.location, []) if neighbor in remaining
    ]
    return min(neighbor_costs) if neighbor_costs else 1  # fallback in case no neighbor is priority


def a_star_search(problem):
    """ A* uses g + h to decide which state to explore next.
    g = distance so far, h = estimated remaining distance. """
    start = problem.initial_state()
    frontier = []
    counter = count()
    heapq.heappush(frontier, (0, next(counter), start))
    reached = {start: 0}       # g-cost to reach state
    nodes_expanded = 0

    while frontier:
        f_cost, _, state = heapq.heappop(frontier)
        nodes_expanded += 1

        if state.is_goal(problem.priority_rooms):
            return reached[state], state, nodes_expanded

        for _, next_state, step_cost in problem.successors(state):
            g = reached[state] + step_cost
            f = g + heuristic(next_state, problem)
            if next_state not in reached or g < reached[next_state]:
                reached[next_state] = g
                heapq.heappush(frontier, (f, next(counter), next_state))

    return None, None, nodes_expanded

# Example
if __name__ == "__main__":
    # Museum layout graph
    graph = {
        "Security": [("Gallery1", 2), ("Gallery2", 3)],
        "Gallery1": [("Gallery2", 2), ("ExhibitA", 4), ("Security", 2)],
        "Gallery2": [("Gallery1", 2), ("ExhibitB", 3), ("Security", 3)],
        "ExhibitA": [("Gallery1", 4), ("ExhibitB", 2)],
        "ExhibitB": [("Gallery2", 3), ("ExhibitA", 2), ("Storage", 4)],
        "Storage": [("ExhibitB", 4)]
    }

    # Important rooms the guard must visit
    priority_rooms = ["ExhibitA", "ExhibitB", "Storage"]

    # Create problem instance
    problem = MuseumPatrolProblem(graph, priority_rooms, shift_limit=15)

    # Run UCS
    cost, state, nodes = uniform_cost_search(problem)
    print("UCS total cost:", cost, "- nodes expanded:", nodes)

    # Run A*
    cost, state, nodes = a_star_search(problem)
    print("A* total cost:", cost, "- nodes expanded:", nodes)
