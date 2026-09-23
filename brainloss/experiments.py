"""Task lists for the runner."""
FRACS = [0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.85, 0.95]


def curve_tasks(graph, w, mode, conds, fracs=FRACS, masks=20, flies=(1, 2, 3, 4, 5), sd=0.5, T=300.0, trials=3,
                dt=0.2, tag=""):
    """Random removal: every (fraction, mask, fly). Fraction 0 is the intact brain of each fly."""
    tasks = []
    for p in fracs:
        for m in range(masks):
            for fly in flies:
                key = f"{graph}|{w}|{mode}{tag}|{p}|{m}|{fly}"
                tasks.append(dict(key=key, graph=graph, w=w, dt=dt, T=T, trials=trials, fly=fly, sd=sd,
                                  conds=conds, mask=(mode, p, m), seed=m * 101 + fly))
    return tasks


def synapse_tasks(graph, w, conds, fracs=FRACS, masks=20, flies=(1, 2, 3, 4, 5), sd=0.5, T=300.0, trials=3, dt=0.2):
    """Control: the same fractions of synapses removed at random all over the graph, no neuron removed."""
    tasks = []
    for p in fracs:
        for m in range(masks):
            for fly in flies:
                key = f"{graph}|{w}|syn|{p}|{m}|{fly}"
                t = dict(key=key, graph=graph, w=w, dt=dt, T=T, trials=trials, fly=fly, sd=sd, conds=conds,
                         removed=[], seed=m * 101 + fly)
                if p > 0:
                    t["syn"] = (p, 1000 + m)
                tasks.append(t)
    return tasks


def single_tasks(graph, w, removals, conds, flies=(0,), sd=0.5, T=500.0, trials=4, dt=0.2, tag="single"):
    """removals: label -> list of neuron indices (empty list = intact)."""
    tasks = []
    for label, idx in removals.items():
        for fly in flies:
            key = f"{graph}|{w}|{tag}|{label}|{dt}|{fly}|{sd}"
            tasks.append(dict(key=key, graph=graph, w=w, dt=dt, T=T, trials=trials, fly=fly, sd=sd,
                              conds=conds, removed=list(map(int, idx)), seed=fly * 7 + 1))
    return tasks
