
import json

"""
This file is to showcase that adjustments are needed to make the simulator work for the benchmark instances
One: there is a mismatch in the deterministic CP makespan and deterministic simulation makespan
Two: while running the stochastic simulation some products do not finish, giving a infinite penalty for start time deviations
"""

benchmark = True # Decide which instance type we run as example

if benchmark:
    from problems.RCPSP_benchmark.simulator.classes import ProductionPlan
    from problems.RCPSP_benchmark.simulator.simulator import Simulator
    from problems.RCPSP_benchmark.simulator.operator import Operator
    from problems.RCPSP_benchmark.solver.deterministic_solver import solve_RCPSP_CP, convert_instance_RCPSP
    # Read instance from CP benchmark set
    instance_name = "j301_1" # This is an instance from the CP benchmark set
    problem_type = "RCPSP_benchmark"

else:
    from problems.RCPSP_penalty.simulator.classes import ProductionPlan
    from problems.RCPSP_penalty.simulator.simulator import Simulator
    from problems.RCPSP_penalty.simulator.operator import Operator
    from problems.RCPSP_penalty.solver.deterministic_solver import solve_RCPSP_CP, convert_instance_RCPSP
    # Read instance from our set
    instance_name = "60_1_factory_1" # This is one of my own instances
    problem_type = "RCPSP_penalty"


productionplan = ProductionPlan(
                    **json.load(open(f'problems/{problem_type}/instances/instance_{instance_name }.json')))

# Solve CP model
durations, num_tasks, num_resources, successors, temporal_relations, demands, capacity, deadlines, activity_translation, \
product_translation = convert_instance_RCPSP(instance=productionplan)
res, earliest_start = solve_RCPSP_CP(durations, num_tasks, num_resources, successors, temporal_relations,
                          demands, capacity,
                          deadlines, activity_translation, product_translation)

# Run deterministic simulation
cp_makespan = max(earliest_start["end"])
print(earliest_start)
earliest_start = earliest_start.to_dict('records')
productionplan.set_earliest_start_times(earliest_start)
operator = Operator(plan=productionplan, policy_type=2, printing=False)
my_simulator = Simulator(plan=productionplan, operator=operator, printing=False)
makespan, lateness, nr_unfinished_products, total_penalty = my_simulator.simulate(sim_time=2000, write=False)

print(f'\nAccording to (deterministic) simulation we had makespan {makespan} and {nr_unfinished_products} unfinished, while CP makespan was {cp_makespan}\n')

# TODO: the makespan does only match the CP makespan when we model a small hack: env.timeout(proc-time-0.00000001)
#  to release machines a bit earlier

# Run stochastic simulation for multiple scenarios
for s in range(1, 1000):
    scenario = productionplan.create_scenario(s)
    operator = Operator(plan=scenario.production_plan, policy_type=2, printing=False)
    my_simulator = Simulator(plan=scenario.production_plan, operator=operator, printing=
                             False, penalty_divide=False, penalty=1)
    makespan, lateness, nr_unfinished, penalty = my_simulator.simulate(sim_time=9999, write=False)
    print(f'scenario {s} with makespan {makespan} and unfinished {nr_unfinished} and penalty is {penalty}')
    if nr_unfinished > 0:
        print(f'WARNING NOT ALL PRODUCTS FINISHED')

# TODO: the time issues haas been resolved by again a small time difference hack (is this common in Discrete-Event?)
