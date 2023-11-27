import pandas as pd
from docplex.cp.model import *

"""
This file contains the stochastic model that can be used to solve PSPlib instances (j30, j90)
"""

def solve_rcpsp_penalty_stochastic(scenarios, durations, num_tasks, num_resources, successors, temporal_relations, demands, capacity,
                              deadlines, activity_translation, product_translation, time_limit=None, write=False,
                              output_location="results.csv", seed=1, penalty=1, analysis=False, analysis_output="analysis.csv",
                                   objectives_output="objectives.csv"):
    # Create model
    mdl = CpoModel()

    # Create task interval variables
    all_tasks = []
    first_stage = [mdl.integer_var(name=f'first_stage_decision{i}') for i in range(num_tasks)]
    makespans = [mdl.integer_var(name=f'makespan_scenarios{omega}') for omega in scenarios]
    deviations = [mdl.integer_var(name=f'deviation_scenarios{omega}') for omega in scenarios]

    mdl.add(first_stage[t] >= 0 for t in range(num_tasks))
    # Make scenario intervals
    for omega in scenarios:
        tasks = [mdl.interval_var(name=f'T{i}_{omega}', size=durations[omega][i]) for i in range(num_tasks)]
        all_tasks.append(tasks)

    # Add constraints
    for omega in scenarios:
        tasks = all_tasks[omega]

        # Add relation between scenario start times and first stage decision
        mdl.add(start_of(tasks[t]) >= first_stage[t] for t in range(num_tasks))

        # Precedence relations
        mdl.add(
            end_before_start(tasks[t], tasks[s - 1], delay=temporal_relations[(t + 1, s)]) for t in range(num_tasks)
            for s in successors[t])

        # Constrain capacity of resources
        mdl.add(
            sum(pulse(tasks[t], demands[t][r]) for t in range(num_tasks) if demands[t][r] > 0) <= capacity[r] for r in
            range(num_resources))

        # Makespan constraint for this scenario
        mdl.add(makespans[omega] >= max(end_of(t) for t in tasks))

    # Solve model
    # Compute deviation from schedule
    for omega in scenarios:
        tasks = all_tasks[omega]

        mdl.add(deviations[omega] == sum(start_of(tasks[t]) - first_stage[t] for t in range(num_tasks)))

    mdl.add(minimize(sum(makespans[omega] for omega in scenarios) + penalty * sum(deviations[omega] for omega in scenarios)))

    res = mdl.solve(TimeLimit=time_limit, Workers=1, LogVerbosity="Quiet", RandomSeed=seed)

    data = []
    if res:

        # To do obtain makespan
        # To do obtain deviation factor
        for i in range(num_tasks):
            start = res.get_var_solution(first_stage[i]).value
            data.append({"task": i, "earliest_start": start, 'product_id': product_translation[i],
                         'activity_id': activity_translation[i]})
        data_df = pd.DataFrame(data)
        if write:
            data_df.to_csv(output_location)

        if analysis:
            data = []
            for omega in scenarios:
                for i in range(num_tasks):
                    tasks = all_tasks[omega]
                    first_stage_start = res.get_var_solution(first_stage[i]).value
                    start_scenario = res.get_var_solution(tasks[i]).start
                    end_scenario = res.get_var_solution(tasks[i]).end
                    data.append({"scenario": omega, "task": i,"product_id": product_translation[i],
                                 "activity_id": activity_translation[i], "first_stage": first_stage_start,
                                 "scenario_start": start_scenario, "scenario_end": end_scenario
                                 })
            analysis_df = pd.DataFrame(data)
            analysis_df.to_csv(analysis_output)

            objectives = []
            for omega in scenarios:
                makespan = res.get_var_solution(makespans[omega]).value
                total_deviation = res.get_var_solution(deviations[omega]).value
                objectives.append({'scenario': omega,
                                   'makespan': makespan,
                                   'total_deviation': total_deviation})

            objectives_df = pd.DataFrame(objectives)
            objectives_df.to_csv(objectives_output)

        return res, data_df

    else:
        return None, None