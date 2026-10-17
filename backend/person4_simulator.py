"""Small, explainable deadlock demonstration (not a system-wide detector)."""


def detect_cycle(wait_for_graph):
    """Return a cycle path if the directed wait-for graph contains a cycle."""
    visiting, visited, stack = set(), set(), []

    def dfs(node):
        if node in visiting:
            start = stack.index(node)
            return stack[start:] + [node]
        if node in visited:
            return None
        visiting.add(node)
        stack.append(node)
        for neighbour in wait_for_graph.get(node, []):
            cycle = dfs(neighbour)
            if cycle:
                return cycle
        stack.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    for node in wait_for_graph:
        cycle = dfs(node)
        if cycle:
            return cycle
    return None


def deadlock_scenario(scenario):
    if scenario == "safe":
        holds = [
            {"task": "Task 1", "resource": "Resource A"},
        ]
        requests = [
            {"task": "Task 2", "resource": "Resource A", "waiting_for": "Task 1"},
        ]
        graph = {"Task 2": ["Task 1"], "Task 1": []}
        explanation = "Only Task 2 waits for Task 1. There is no circular wait, so the detector finds no deadlock."
    else:
        holds = [
            {"task": "Task 1", "resource": "Resource A"},
            {"task": "Task 2", "resource": "Resource B"},
        ]
        requests = [
            {"task": "Task 1", "resource": "Resource B", "waiting_for": "Task 2"},
            {"task": "Task 2", "resource": "Resource A", "waiting_for": "Task 1"},
        ]
        graph = {"Task 1": ["Task 2"], "Task 2": ["Task 1"]}
        explanation = "Task 1 waits for Task 2 while Task 2 waits for Task 1. This circular wait is a deadlock."
    cycle = detect_cycle(graph)
    return {
        "scenario": scenario,
        "deadlock_detected": bool(cycle),
        "result": "Deadlock Detected" if cycle else "No Deadlock Detected",
        "holds": holds,
        "requests": requests,
        "cycle": cycle,
        "explanation": explanation,
    }
