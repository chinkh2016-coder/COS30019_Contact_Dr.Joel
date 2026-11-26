"""Directed reachability helpers."""


def nodes_reaching(graph, destination):
    # Reverse every directed connection.
    reverse = {node: [] for node in graph}

    for source, edges in graph.items():
        for target, _ in edges:
            reverse.setdefault(target, []).append(source)

    # Search backwards from the destination.
    seen = {destination}
    pending = [destination]

    while pending:
        current = pending.pop()

        for source in reverse.get(current, []):
            if source not in seen:
                seen.add(source)
                pending.append(source)

    return seen


def can_reach(graph, origin, destination):
    return origin in nodes_reaching(
        graph,
        destination,
    )