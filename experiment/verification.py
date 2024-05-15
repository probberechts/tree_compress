import time
import veritas

def count_ocs(at, timeout):
    config = veritas.Config(veritas.HeuristicType.MAX_OUTPUT)
    config.stop_when_optimal = False
    config.max_memory = 16*1024*1024*1024
    search = config.get_search(at)

    has_timed_out = False

    while search.steps(1000) != veritas.StopReason.NO_MORE_OPEN:
        has_timed_out = search.time_since_start() >= timeout
        if has_timed_out:
            break

    return search.num_solutions(), search.time_since_start(), has_timed_out

def approx_emp_robustness(at, example, target_label):
    if target_label:
        source_at, target_at = None, at
    else:
        source_at, target_at = at, None
    
    start_delta = 0.4
    rob = veritas.VeritasRobustnessSearch(
        example, start_delta, source_at, target_at, silent=True
    )
    delta, delta_lo, delta_hi = rob.search()

    return delta_lo

def exact_emp_robustness(at, example, target_label):
    kan = veritas.KantchelianAttack(at, target_label, example)
    kan.optimize()
    return kan.bounds[-1][0]

def emp_robustness(at, x, y, n, exact):
    delta_lo = 0.0
    count = 0

    if exact:
        f = exact_emp_robustness
    else:
        f = approx_emp_robustness

    t = time.time()
    for i in x.index:
        target_label = not (y.loc[i] > 0.0)
        example = x.loc[i, :].to_numpy()
        pred_label = at.eval(example)[0, 0] > 0.0
        res = f(at, example, target_label)
        delta_lo += res
        if pred_label == target_label:
            print("wrongly classified", res)
        count += 1
        if count >= n:
            break
    t = time.time() - t

    return delta_lo / count, t
