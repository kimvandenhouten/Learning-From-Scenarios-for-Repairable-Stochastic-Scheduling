import json
import pandas as pd
import torch
from classes.classes import DataLoader, CustomDataset, Model
import numpy as np
from matplotlib import pyplot as plt
import torch.optim as optim
from problems.RCPSP_benchmark.problem_class import Problem
import time
from problems.RCPSP_benchmark.simulator.classes import ProductionPlan
from problems.RCPSP_benchmark.solver.deterministic_solver import convert_instance_RCPSP

""" 
This Python script runs the decision-focused learning (also known as score function gradient estimation (SFGE) algorithm 
for the PSPlib instances.
"""

# Settings
train_size, val_size, test_size = 100, 50, 50
batch_size, noise_factor, seed, stop_criterium = 10, 0, 1, "time"
loss_type, problem_type = "post_hoc_regret", "RCPSP_benchmark"
plotting = False
nr_scenarios, budget, instance_folder = 50, 3600, "j90"

for a in [1, 2, 3]:
    for penalty_type in ["small", "large"]:
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
            for (lr, method) in [(0.001, "Adam")]:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                # Initialize problem instance
                print(instance_name)
                problem = Problem(problem=problem_type, instance_name=instance_name, noise_factor=noise_factor, penalty=penalty_factor, penalty_divide=False)
                y_true_train, y_true_val, y_true_test = problem.generate_data(train_size=train_size, val_size=val_size, test_size=test_size, validation=True, test=True)

                y_true_train = y_true_train[0:nr_scenarios]

                # Compute y_bar and y_std which will be used later on
                y_true = torch.tensor(y_true_train, dtype=torch.float32)

                y_bar = torch.mean(y_true, dim=0)
                std_bar = torch.std(y_true, dim=0)
                std_bar[std_bar == 0] = 0.00001
                dim = y_true.shape[1]

                # Step 2: Instantiate a DataLoader
                dataset_train = CustomDataset(y_true_train)
                dataset_test = CustomDataset(y_true_test)
                dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False)
                dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=False)

                # Step 3: Set up model and optimizer
                model = Model(dim=dim)
                mu, sigma = model(mu_bar=y_bar, sigma_bar=std_bar)
                if method == "SGD":
                    optimizer = optim.SGD(model.parameters(), lr=lr)
                elif method == "Adam":
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                objs_avg, losses_avg = [], []

                # Initial evaluation, evaluate current yhat distribution
                if plotting:
                    objs, regrets, post_hoc_regrets, sol = problem.evaluate_mu(mu, data="train")
                    objs_avg.append(np.mean(objs))
                    if loss_type == "regret":
                        losses_avg.append(np.mean(regrets))
                    elif loss_type == "post_hoc_regret":
                        losses_avg.append(np.mean(post_hoc_regrets))

                    plt.plot(losses_avg)
                    plt.savefig(
                        f'plots/{problem_type}/{problem_type}_i={instance_name}_seed={seed}_train_size{train_size}_'
                        f'val_size={val_size}_n={noise_factor}_scen={nr_scenarios}_p={penalty_factor}.png')
                    plt.close()

                # Iterate over epochs
                nr_epochs = 0
                start = time.time()
                training_time = 0
                # Iterate over epochs
                training = True
                while training:
                    nr_epochs += 1
                    print(f'Start epoch {nr_epochs}')
                    for batch in dataloader_train:
                        scenarios = batch['idx']
                        y_true_batch = batch['y_true']
                        objs, tobjs, log_probs, penalties = [], [], [], []

                        for s, y_true in zip(scenarios, y_true_batch):
                            mu, sigma = model(mu_bar=y_bar, sigma_bar=std_bar)
                            y_hat = torch.normal(mu, sigma)
                            log_prob = torch.sum(torch.distributions.Normal(mu, sigma).log_prob(y_hat))
                            log_probs.append(log_prob)
                            y_pred = [max(int(round(x)), 1) for x in y_hat.tolist()]
                            pobj, psol = problem.solve(y_pred)
                            obj, penalty = problem.repair(psol, scenario=s)
                            objs.append(obj)
                            tobjs.append(problem.tobjs[s])
                            penalties.append(penalty)

                        objs = np.array(objs)
                        tobjs = np.array(tobjs)
                        if loss_type == "regret":
                            loss_terms = torch.tensor(objs - tobjs)
                        elif loss_type == "post_hoc_regret":
                            loss_terms = torch.tensor(objs - tobjs + penalties)
                        loss_terms = loss_terms.to(torch.float32)

                        # Perform element-wise multiplication
                        loss = torch.mean(loss_terms * torch.stack(log_probs))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if plotting:
                        mu, sigma = model(mu_bar=y_bar, sigma_bar=std_bar)
                        objs, regrets, post_hoc_regrets, sol = problem.evaluate_mu(mu, data="train")
                        objs_avg.append(np.mean(objs))
                        if loss_type == "regret":
                            losses_avg.append(np.mean(regrets))
                        elif loss_type == "post_hoc_regret":
                            losses_avg.append(np.mean(post_hoc_regrets))
                        plt.plot(losses_avg)
                        plt.savefig(f'plots/{problem_type}/problem={problem_type}_instance={instance_name}_seed={seed}_train_size{train_size}_'
                                    f'val_size={val_size}_noise_factor={noise_factor}_scen={nr_scenarios}_p={penalty_factor}.png')
                        plt.close()

                    training_time = time.time() - start

                    if stop_criterium == "epochs":
                        if nr_epochs >= budget:
                            training = False
                    elif stop_criterium == "time":
                        if training_time >= budget:
                            training = False

                # Final evaluation after training
                finish_training = time.time()
                print(f'After training')

                mu, sigma = model(mu_bar=y_bar, sigma_bar=std_bar)
                objs, regrets, post_hoc_regrets, sol = problem.evaluate_mu(mu, data="train")
                finish_inference = time.time()
                train_obj = np.mean(objs)

                if loss_type == "regret":
                    train_loss = np.mean(regrets)
                    train_loss_std = np.std(regrets)
                elif loss_type == "post_hoc_regret":
                    train_loss = np.mean(post_hoc_regrets)
                    train_loss_std = np.std(post_hoc_regrets)
                print(f'Training obj {train_obj}, training regret loss {train_loss}')

                objs, regrets, post_hoc_regrets, sol = problem.evaluate_mu(mu, data="val")
                val_obj = np.mean(objs)
                if loss_type == "regret":
                    val_loss = np.mean(regrets)
                    val_loss_std = np.std(regrets)
                elif loss_type == "post_hoc_regret":
                    val_loss = np.mean(post_hoc_regrets)
                    val_loss_std = np.std(post_hoc_regrets)

                objs, regrets, post_hoc_regrets, sol = problem.evaluate_mu(mu, data="test")
                test_obj = np.mean(objs)

                if loss_type == "regret":
                    test_loss = np.mean(regrets)
                    test_loss_std = np.std(regrets)
                elif loss_type == "post_hoc_regret":
                    test_loss = np.mean(post_hoc_regrets)
                    test_loss_std = np.std(post_hoc_regrets)

                print(f'Test obj {test_obj}, test regret loss {test_loss}')
                training_time = finish_training - start
                inference_time = finish_inference - finish_training
                total_time = finish_inference - start

                results = [{"problem": problem_type,
                           "instance": instance_name,
                            "num_tasks": num_tasks,
                            "num_resources": num_resources,
                            "total_capacity": sum(capacity),
                               "seed": seed,
                               "train_size": train_size,
                               "test_size": test_size,
                               "epochs": nr_epochs,
                               "batch_size": batch_size,
                               "optimizer": method,
                               "learning rate": lr,
                               "training_obj": train_obj,
                               "val_obj": val_obj,
                               "test_obj": test_obj,
                               "training_time": training_time,
                               "inference_time": inference_time,
                               "total_time": total_time,
                               "stop_criterium": stop_criterium,
                               "budget": budget,
                               "noise_factor": noise_factor,
                               "scenarios": nr_scenarios,
                               "training_loss": train_loss,
                                "std_training_loss": train_loss_std,
                                "val_loss": val_loss,
                                "std_val": val_loss_std,
                                "test_loss": test_loss,
                                "std_test_loss": test_loss_std,
                                "penalty_factor": penalty_factor,
                                "algorithm": "decision-focused"
                           }]

                results = pd.DataFrame(results)
                results.to_csv(f'results/sfge_{instance_name}_s={seed}_ts_{train_size}_vs={val_size}'
                               f'_stop={stop_criterium}_b={budget}_bs={batch_size}_opt={method}'
                               f'_lr={lr}_n={noise_factor}_scen={nr_scenarios}_p={penalty_factor}.csv')




