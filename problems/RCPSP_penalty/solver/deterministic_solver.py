import logging
from docplex.cp.model import *


# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
The RCPSP (Resource-Constrained Project Scheduling Problem) is a generalization
of the production-specific Job-Shop (see job_shop_basic.py), Flow-Shop
(see flow_shop.py) and Open-Shop(see open_shop.py) scheduling problems.

Given:
- a set of q resources with given capacities,
- a network of precedence constraints between the activities, and
- for each activity and each resource the amount of the resource
  required by the activity over its execution,
the goal of the RCPSP is to find a schedule meeting all the
constraints whose makespan (i.e., the time at which all activities are
finished) is minimal.

Please refer to documentation for appropriate setup of solving configuration.
"""
import pandas as pd
from docplex.cp.model import *
import docplex.cp.solver as solver


def convert_instance_RCPSP(instance):
    # convert to RCPSP instance
    capacity = instance.factory.capacity
    durations = []
    deadlines = []
    num_tasks = 0
    num_resources = len(capacity)
    demands = []
    successors = [[] for p in instance.products for a in p.activities]
    temporal_relations = {}

    resource_translation = instance.factory.resource_names
    product_translation = []
    activity_translation = []

    activity_counter = 1
    product_counter = 0

    # loop over all products
    for p in instance.products:
        for (i, j) in p.temporal_relations:
            temporal_relations[(i + activity_counter, j + activity_counter)] = p.temporal_relations[(i, j)]
            successors[i + activity_counter - 1].append(j + activity_counter)

        nr_activities = len(p.activities)
        for act in range(0, nr_activities):
            product_translation.append(product_counter)
            activity_translation.append(act)

        for counter, a in enumerate(p.activities):
            if counter == nr_activities - 1:
                deadlines.append(p.deadline)
            else:
                deadlines.append(1000000000)

            durations.append(a.processing_time[0])
            demands.append(a.needs)

        activity_counter += nr_activities
        product_counter += 1
    num_tasks = len(durations)
    return durations, num_tasks, num_resources, successors, temporal_relations, demands, capacity, deadlines, activity_translation, product_translation


def solve_RCPSP_CP(durations, num_tasks, num_resources, successors, temporal_relations, demands, capacity,
                   deadlines, activity_translation, product_translation, temp_relation="time_lag",
                   time_limit=None, l1=1, l2=0, write=False, output_location="results.csv"):

    # Create model
    mdl = CpoModel()
    # Create task interval variables
    tasks = [interval_var(name='T{}'.format(i+1), size=durations[i]) for i in range(num_tasks)]

    # Add precedence constraints
    if temp_relation == "time_lag":
        mdl.add(start_before_start(tasks[t], tasks[s-1], delay=temporal_relations[(t+1, s)]) for t in range(num_tasks)
                for s in successors[t])
    elif temp_relation == "temporal":
        mdl.add(start_at_start(tasks[t], tasks[s - 1], delay=temporal_relations[(t + 1, s)]) for t in range(num_tasks)
            for s in successors[t])

    # Constrain capacity of resources
    mdl.add(sum(pulse(tasks[t], demands[t][r]) for t in range(num_tasks) if demands[t][r] > 0) <= capacity[r] for r in range(num_resources))

    mdl.add(minimize(l1 * max(end_of(t) for t in tasks) + l2 * sum(max(end_of(tasks[t]) - deadlines[t], 0) for t in range(num_tasks))))


    # Solve model
    res = mdl.solve(TimeLimit=time_limit, Workers=1, LogVerbosity="Quiet")

    data = []
    if res:
        for i in range(len(durations)):
            start = res.get_var_solution(tasks[i]).start
            end = res.get_var_solution(tasks[i]).end
            data.append({"task": i, "earliest_start": start, "end": end, 'product_id': product_translation[i],
                         'activity_id': activity_translation[i]})
        data_df = pd.DataFrame(data)
        if write:
            data_df.to_csv(output_location)

    return res,  data_df




