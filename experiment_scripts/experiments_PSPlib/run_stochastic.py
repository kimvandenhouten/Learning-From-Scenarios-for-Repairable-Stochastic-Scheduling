from docplex.cp.model import *
import pandas as pd
from problems.RCPSP_benchmark.solver.deterministic_solver import convert_instance_RCPSP
from problems.RCPSP_benchmark.solver.stochastic_solver import solve_rcpsp_penalty_stochastic
from problems.RCPSP_benchmark.simulator.classes import ProductionPlan
from problems.RCPSP_benchmark.problem_class import Problem
import numpy as np
import time

""" 
This Python script runs the stochastic algorithm for the PSPlib instances.
"""

# Settings
train_size, val_size, test_size = 100, 50, 50
problem_type, loss_type = "RCPSP_benchmark", "post_hoc_regret"
noise_factor, seed = 0, 1
nr_scenarios, budget, instance_folder = 50, 3600, "j90"


for a in [1, 2, 3]:
    for penalty_type in ["small"]:

        for b in range(1, 11):
            instance_name = f'{instance_folder}{a}_{b}'
            print(instance_name)
            my_productionplan = ProductionPlan(
                **json.load(open(f'problems/{problem_type}/instances/instance_' + instance_name + '.json')))
            durations, num_tasks, num_resources, successors, temporal_relations, demands, capacity, deadlines, activity_translation, \
            product_translation = convert_instance_RCPSP(instance=my_productionplan)
            if penalty_type == "small":
                penalty_factor = np.round(1 / num_tasks, 3)
            elif penalty_type == "large":
                penalty_factor = 1
            elif penalty_type == "medium":
                penalty_factor = np.round(10 / num_tasks, 3)
            elif penalty_type == "zero":
                penalty_factor = 0

            print(f'Runtime budget is {budget}, nr of scenarios  {nr_scenarios}, seed is {seed}, penalty {penalty_factor}')
            scenarios = [i for i in range(nr_scenarios)]
            problem = Problem(problem=problem_type, instance_name=instance_name, noise_factor=noise_factor, penalty=penalty_factor, penalty_divide=False)
            y_true_train, y_true_val, y_true_test = problem.generate_data(train_size=train_size, val_size=val_size,
                                                                          test_size=test_size, validation=True, test=True)
            # Generate y data and compute optimal solutions under perfect information
            durations_scenarios = []
            for s in scenarios:
                scenario_1 = my_productionplan.create_scenario(s, noise_factor=noise_factor)
                true_processing_times = []
                for product in scenario_1.production_plan.products:
                    for activity in product.activities:
                        true_processing_times.append(activity.processing_time[0])
                durations_scenarios.append(true_processing_times)

            start = time.time()

            res, sol = solve_rcpsp_penalty_stochastic(scenarios, durations_scenarios, num_tasks, num_resources, successors, temporal_relations, demands, capacity,
                            deadlines, activity_translation, product_translation, time_limit=budget, seed=seed,
                            penalty=penalty_factor, analysis=False, analysis_output=f'analysis/{problem_type}/stochastic/'
                            f'{instance_name}_seed={seed}_train={train_size}_test={test_size}_scen={nr_scenarios}_b={budget}'
                            f'_noise={noise_factor}_pen={penalty_factor}.csv', objectives_output=f'analysis/{problem_type}/stochastic/'
                            f'objectives_{instance_name}_seed={seed}_train={train_size}_test={test_size}_scen={nr_scenarios}_b={budget}'
                            f'_noise={noise_factor}_pen={penalty_factor}.csv')

            sol = sol.to_dict('records')
            finish = time.time()
            print("solving finished")

            # Evaluate
            scenarios_train = [i for i in range(0, train_size)]
            scenarios_val = [i for i in range(train_size, train_size + val_size)]
            scenarios_test = [i for i in range(train_size + val_size, train_size + val_size + test_size)]

            train_objs, val_objs, test_objs = [], [], []
            train_regrets, val_regrets, test_regrets = [], [], []
            train_ph_regrets, val_ph_regrets, test_ph_regrets = [], [], []

            for s in scenarios_train:
                obj, penalty = problem.repair(sol, s)
                regret = obj - problem.tobjs[s]
                post_hoc_regret = regret + penalty
                train_objs.append(obj)
                train_regrets.append(regret)
                train_ph_regrets.append(post_hoc_regret)
                if regret < 0:
                    print(f'WARNING: regret < 0 for scenario {s} of instance {instance_name}')

            for s in scenarios_val:
                obj, penalty = problem.repair(sol, s)
                regret = obj - problem.tobjs[s]
                post_hoc_regret = regret + penalty
                val_objs.append(obj)
                val_regrets.append(regret)
                val_ph_regrets.append(post_hoc_regret)
                if regret < 0:
                    print(f'WARNING: regret < 0 for scenario {s} of instance {instance_name}')

            for s in scenarios_test:
                obj, penalty = problem.repair(sol, s)
                regret = obj - problem.tobjs[s]
                post_hoc_regret = regret + penalty
                test_objs.append(obj)
                test_regrets.append(regret)
                test_ph_regrets.append(post_hoc_regret)
                if regret < 0:
                    print(f'WARNING: regret < 0 for scenario {s} of instance {instance_name}')

            train_obj = np.mean(train_objs)
            val_obj = np.mean(val_objs)
            test_obj = np.mean(test_objs)
            train_regret = np.mean(train_regrets)
            val_regret = np.mean(val_regrets)
            test_regret = np.mean(test_regrets)
            train_ph_regret = np.mean(train_ph_regrets)
            val_ph_regret = np.mean(val_ph_regrets)
            test_ph_regret = np.mean(test_ph_regrets)

            train_obj_std = np.std(train_objs)
            val_obj_std = np.std(val_objs)
            test_obj_std = np.std(test_objs)
            train_regret_std = np.std(train_regrets)
            val_regret_std = np.std(val_regrets)
            test_regret_std = np.std(test_regrets)
            train_ph_regret_std = np.std(train_ph_regrets)
            val_ph_regret_std = np.std(val_ph_regrets)
            test_ph_regret_std = np.std(test_ph_regrets)

            if loss_type == "regret":
                train_loss = train_regret
                val_loss = val_regret
                test_loss = test_regret
                train_loss_std = train_regret_std
                val_loss_std = val_regret_std
                test_loss_std = test_regret_std
            elif loss_type == "post_hoc_regret":
                train_loss = train_ph_regret
                val_loss = val_ph_regret
                test_loss = test_ph_regret
                train_loss_std = train_ph_regret_std
                val_loss_std = val_ph_regret_std
                test_loss_std = test_ph_regret_std

            results = [{"problem": problem_type,
                        "instance": instance_name,
                        "num_tasks": num_tasks,
                        "num_resources": num_resources,
                        "total_capacity": sum(capacity),
                        "seed": seed,
                        "noise_factor": noise_factor,
                        "scenarios": nr_scenarios,
                        "budget": budget,
                        "train_size": train_size,
                        "test_size": test_size,
                        "training_obj": train_obj,
                        "training_regret": train_regret,
                        "training_ph_regret": train_ph_regret,
                        "val_obj": val_obj,
                        "val_regret": val_regret,
                        "val_ph_regret": val_ph_regret,
                        "test_obj": test_obj,
                        "test_regret": test_regret,
                        "test_ph_regret": test_ph_regret,
                        "total_time": finish - start,
                        "training_loss": train_loss,
                        "std_training_loss": train_loss_std,
                        "val_loss": val_loss,
                        "std_val": val_loss_std,
                        "test_loss": test_loss,
                        "std_test_loss": test_loss_std,
                        "penalty_factor": penalty_factor,
                        "algorithm": "stochastic"
                        }]

            print(f'Training post-hoc regret {train_ph_regret}')
            print(f'Validation post-hoc regret {val_ph_regret}')
            print(f'Test post-hoc regret {test_ph_regret}')

            results = pd.DataFrame(results)
            results.to_csv(f'results/j90/{penalty_type}/stoch_{instance_name}_s={seed}_tr={train_size}_'
                           f'v={val_size}_t={test_size}_sc={nr_scenarios}_b={budget}_n={noise_factor}_p={penalty_factor}.csv')

