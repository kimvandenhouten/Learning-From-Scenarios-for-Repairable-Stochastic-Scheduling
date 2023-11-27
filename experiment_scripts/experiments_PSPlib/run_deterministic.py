import numpy as np
import json
import pandas as pd
import time
from problems.RCPSP_benchmark.simulator.classes import ProductionPlan
from problems.RCPSP_benchmark.problem_class import Problem
from problems.RCPSP_benchmark.solver.deterministic_solver import solve_RCPSP_CP, convert_instance_RCPSP

""" 
This Python script runs the deterministic algorithm for the PSPlib instances.
"""


# Settings
train_size, val_size, test_size = 100, 50, 50
validation, test = True, True  # Whether we use the data and need to solve the true optimums
noise_factor, nr_scenarios = 0, train_size
problem_type, loss_type = "RCPSP_benchmark", "post_hoc_regret"
noise_factor, nr_scenarios, budget, seed = 0, 50, 3600, 1
instance_folder = "j90"

# List random seeds for scenarios of training data (these will be used in the simulation evaluation)
scenarios_train_set = [i for i in range(0, train_size)]
scenarios_val_set = [i for i in range(train_size, train_size + val_size)]
scenarios_test_set = [i for i in range(train_size + val_size, train_size + val_size + test_size)]

for a in [1, 2, 3]:
    for penalty_type in ["small"]:
        for b in range(1, 11):
            instance_name = f'{instance_folder}{a}_{b}'
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

            print(f'Runtime budget is {budget}, nr of scenarios  {nr_scenarios}, seed is {seed} penalty factor is {penalty_factor}')

            problem = Problem(problem=problem_type, instance_name=instance_name, noise_factor=noise_factor, penalty=penalty_factor, penalty_divide=False)
            y_true_train, y_true_val, y_true_test = problem.generate_data(train_size=train_size, val_size=val_size, test_size=test_size, validation=True,
                                            test=True)

            # COMPUTE SAMPLE_AVERAGES OF SCENARIOS
            start = time.time()
            data_y_scenarios = y_true_train[:nr_scenarios]
            y_bar = np.mean(data_y_scenarios, axis=0)
            y_bar = (np.rint(y_bar)).astype(int)

            # SOLVE THE SCHEDULING PROBLEM WITH SAMPLE_AVERAGES
            print(y_bar)
            res, sol = solve_RCPSP_CP(y_bar, num_tasks, num_resources, successors, temporal_relations,
                                       demands, capacity,
                                       deadlines, activity_translation, product_translation, time_limit=budget)
            sol = sol.to_dict('records')
            finish = time.time()

            run_time = finish - start

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
                        "total_time": finish - start,
                        "training_loss": train_loss,
                        "std_training_loss": train_loss_std,
                        "val_loss": val_loss,
                        "std_val": val_loss_std,
                        "test_loss": test_loss,
                        "std_test_loss": test_loss_std,
                        "penalty_factor": penalty_factor,
                        "algorithm": "deterministic"
                        }]

            print(f'Training post-hoc regret {train_ph_regret}')
            print(f'Validation post-hoc regret {val_ph_regret}')
            print(f'Test post-hoc regret {test_ph_regret}')

            results = pd.DataFrame(results)
            results.to_csv(f'results/j90/{penalty_type}/det_{instance_name}_tr={train_size}_'
                           f'v={val_size}_t={test_size}_sc={nr_scenarios}_b={budget}_n={noise_factor}_p={penalty_factor}s={seed}.csv')

