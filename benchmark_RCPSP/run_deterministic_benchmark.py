import numpy as np
import json
import pandas as pd
import time
from problems.RCPSP_penalty.simulator.classes import ProductionPlan
from problems.RCPSP_penalty.problem_class import Problem
from problems.RCPSP_penalty.simulator.operator import Operator
from problems.RCPSP_penalty.simulator.simulator import Simulator
from problems.RCPSP_penalty.solver.deterministic_solver import solve_RCPSP_CP, convert_instance_RCPSP
problem_type = "RCPSP_penalty"
loss_type = "post_hoc_regret"

# Settings
train_size, val_size, test_size = 100, 50, 50
validation, test = True, True  # Whether we use the data an need to solve the true optimums
noise_factor, nr_scenarios = 0, train_size

# List random seeds for scenarios of training data (these will be used in the simulation evaluation)
scenarios_train_set = [i for i in range(0, train_size)]
scenarios_val_set = [i for i in range(train_size, train_size + val_size)]
scenarios_test_set = [i for i in range(train_size + val_size, train_size + val_size + test_size)]


# Settings
solver = True
train_size = 100
val_size = 50
test_size = 50
problem_type = "RCPSP_penalty"
id = 1
noise_factor = 0
nr_scenarios = 100
loss_type = "post_hoc_regret"
for seed in [1]:
    for instance_name in ["j301_1"]:
            penalty_factor = 1
            print(penalty_factor)
            budget = 100
            print(f'Runtime budget is {budget} and nr of scenarios is {nr_scenarios} and seed is {seed}')
            # Initialize problem instance
            problem = Problem(problem=problem_type, instance_name=instance_name, noise_factor=noise_factor, penalty=penalty_factor, penalty_divide=False)
            y_true_train, y_true_val, y_true_test = problem.generate_data(train_size=train_size, val_size=val_size, test_size=test_size, validation=True,
                                            test=True)
            my_productionplan = ProductionPlan(
                    **json.load(open(f'problems/{problem_type}/instances/instance_{instance_name }.json')))
            durations, num_tasks, num_resources, successors, temporal_relations, demands, capacity, deadlines, activity_translation,\
                product_translation = convert_instance_RCPSP(instance=my_productionplan)

            print(temporal_relations)

            # COMPUTE SAMPLE_AVERAGES OF SCENARIOS
            start = time.time()
            data_y_scenarios = y_true_train[:nr_scenarios]
            print(np.array(data_y_scenarios).shape)
            y_bar = np.mean(data_y_scenarios, axis=0)
            y_bar = (np.rint(y_bar)).astype(int)

            # SOLVE THE SCHEDULING PROBLEM WITH SAMPLE_AVERAGES
            res, sol = solve_RCPSP_CP(y_bar, num_tasks, num_resources, successors, temporal_relations,
                                       demands, capacity,
                                       deadlines, activity_translation, product_translation, time_limit=budget)
            sol = sol.to_dict('records')
            print(sol)
            finish = time.time()

            run_time = finish - start

            # Evaluate
            scenarios_train = [i for i in range(0, train_size)]
            scenarios_val = [i for i in range(train_size, train_size + val_size)]
            scenarios_test = [i for i in range(train_size + val_size, train_size + val_size + test_size)]

            train_objs, val_objs, test_objs = [], [], []
            train_regret, val_regret, test_regret = [], [], []
            train_ph_regret, val_ph_regret, test_ph_regret = [], [], []

            for s in scenarios_train:
                obj, penalty = problem.repair(sol, s)
                print(f's {s} obj {obj} penalty {penalty}')
                regret = obj - problem.tobjs[s]
                post_hoc_regret = regret + penalty
                train_objs.append(obj)
                train_regret.append(regret)
                train_ph_regret.append(post_hoc_regret)

            for s in scenarios_val:
                obj, penalty = problem.repair(sol, s)
                regret = obj - problem.tobjs[s]
                post_hoc_regret = regret + penalty
                val_objs.append(obj)
                val_regret.append(regret)
                val_ph_regret.append(post_hoc_regret)

            for s in scenarios_test:
                obj, penalty = problem.repair(sol, s)
                regret = obj - problem.tobjs[s]
                post_hoc_regret = regret + penalty
                test_objs.append(obj)
                test_regret.append(regret)
                test_ph_regret.append(post_hoc_regret)

            train_obj = np.mean(train_objs)
            val_obj = np.mean(val_objs)
            test_obj = np.mean(test_objs)
            train_regret = np.mean(train_regret)
            val_regret = np.mean(val_regret)
            test_regret = np.mean(test_regret)
            train_ph_regret = np.mean(train_ph_regret)
            val_ph_regret = np.mean(val_ph_regret)
            test_ph_regret = np.mean(test_ph_regret)

            train_obj_std = np.std(train_objs)
            val_obj_std = np.std(val_objs)
            test_obj_std = np.std(test_objs)
            train_regret_std = np.std(train_regret)
            val_regret_std = np.std(val_regret)
            test_regret_std = np.std(test_regret)
            train_ph_regret_std = np.std(train_ph_regret)
            val_ph_regret_std = np.std(val_ph_regret)
            test_ph_regret_std = np.std(test_ph_regret)

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
                        "penalty_factor": penalty_factor
                        }]

            print(f'Training regret {train_regret}')
            print(f'Training post-hoc regret {train_ph_regret}')
            print(f'Validation regret {val_regret}')
            print(f'Validation post-hoc regret {val_ph_regret}')
            print(f'Test regret {test_regret}')
            print(f'Test post-hoc regret {test_ph_regret}')

            results = pd.DataFrame(results)
            results.to_csv(f'results/benchmark/{instance_name}_tr={train_size}_'
                           f'v={val_size}_t={test_size}_sc={nr_scenarios}_b={budget}_n={noise_factor}_p={penalty_factor}s={seed}.csv')

