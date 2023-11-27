import json
from problems.RCPSP_benchmark.simulator.operator import Operator
from problems.RCPSP_benchmark.simulator.simulator import Simulator
from problems.RCPSP_benchmark.solver.deterministic_solver import solve_RCPSP_CP, convert_instance_RCPSP
import numpy as np

"""
This problem class can be used as a wrapper in the different algrithms (mainly decision-focused). It is meant as a 
more generic wrapper to be able to run the different algorithms with different problems with different structure, repair
function, data generation and so on.
"""

class Problem:
    def __init__(self, problem, instance_name, noise_factor=0, penalty=0, penalty_divide=True):
        self.problem = problem
        self.instance_name = instance_name
        self.noise_factor = noise_factor
        self.penalty=penalty
        self.penalty_divide=penalty_divide
        print(f'Noise factor is set to {self.noise_factor}')

        if self.problem == "RCPSP_benchmark":
            from problems.RCPSP_benchmark.simulator.classes import ProductionPlan

            self.plan = ProductionPlan(**json.load(open(f'problems/RCPSP_benchmark/instances/instance_{instance_name}.json')))
            _, num_tasks, num_resources, successors, temporal_relations, demands, capacity, deadlines, activity_translation,\
                product_translation = convert_instance_RCPSP(instance=self.plan)
            self.num_tasks = num_tasks
            self.num_resources = num_resources
            self.successors = successors
            self.temporal_relations = temporal_relations
            self.demands = demands
            self.capacity = capacity
            self.deadlines = deadlines
            self.activity_translation = activity_translation
            self.product_translation = product_translation

    def solve(self, y):
        res, data = solve_RCPSP_CP(durations=y, num_tasks=self.num_tasks, num_resources=self.num_resources,
                                 successors=self.successors,
                                 temporal_relations=self.temporal_relations, demands=self.demands,
                                 capacity=self.capacity, deadlines=self.deadlines,
                                 activity_translation=self.activity_translation,
                                 product_translation=self.product_translation)
        obj = res.get_objective_values()[0]
        earliest_start = data.to_dict('records')

        return obj, earliest_start

    def solve_and_write(self, y, output_location):
        res, data = solve_RCPSP_CP(durations=y, num_tasks=self.num_tasks, num_resources=self.num_resources,
                                 successors=self.successors,
                                 temporal_relations=self.temporal_relations, demands=self.demands,
                                 capacity=self.capacity, deadlines=self.deadlines,
                                 activity_translation=self.activity_translation,
                                 product_translation=self.product_translation)
        obj = res.get_objective_values()[0]
        earliest_start = data.to_dict('records')

        #write
        data.to_csv(output_location)

        return obj, earliest_start

    def repair(self, sol, scenario, write=False, output_location="simulation_output.csv"):
        self.plan.set_earliest_start_times(earliest_start=sol)
        scenario_1 = self.plan.create_scenario(scenario, noise_factor=self.noise_factor)
        operator = Operator(plan=scenario_1.production_plan, policy_type=2, printing=False)
        my_simulator = Simulator(plan=scenario_1.production_plan, operator=operator, printing=False, penalty=self.penalty, penalty_divide=self.penalty_divide)
        makespan, lateness, nr_unfinished, penalty = my_simulator.simulate(sim_time=9999, write=write, output_location=output_location)
        obj = makespan
        assert nr_unfinished == 0
        return obj, penalty

    def generate_data(self, train_size, val_size, test_size, validation=False, test=False):
        # Generate random noise samples
        self.scenarios_train = [i for i in range(0, train_size)]
        self.scenarios_val = [i for i in range(train_size, train_size + val_size)]
        self.scenarios_test = [i for i in range(train_size + val_size, train_size + val_size + test_size)]
        self.tobjs = []
        y_true_train, y_true_val, y_true_test = [], [], []

        for s in self.scenarios_train:
            scenario_1 = self.plan.create_scenario(s, noise_factor=self.noise_factor)
            durations = []
            for product in scenario_1.production_plan.products:
                for activity in product.activities:
                    sample = activity.processing_time[0]
                    durations.append(max(0, round(sample)))
            y_true_train.append(durations)
            obj, sol = self.solve(durations)
            self.tobjs.append(obj)

        for s in self.scenarios_val:
            if validation:
                scenario_1 = self.plan.create_scenario(s, noise_factor=self.noise_factor)
                durations = []
                for product in scenario_1.production_plan.products:
                    for activity in product.activities:
                        sample = activity.processing_time[0]
                        durations.append(max(0, round(sample)))
                y_true_val.append(durations)
                obj, sol = self.solve(durations)
                self.tobjs.append(obj)
            else:
                self.tobjs.append(np.inf)

        for s in self.scenarios_test:
            if test:
                scenario_1 = self.plan.create_scenario(s, noise_factor=self.noise_factor)
                durations = []
                for product in scenario_1.production_plan.products:
                    for activity in product.activities:
                        sample = activity.processing_time[0]
                        durations.append(max(0, round(sample)))
                y_true_test.append(durations)
                obj, sol = self.solve(durations)
                self.tobjs.append(obj)
            else:
                self.tobjs.append(np.inf)

        return y_true_train, y_true_val, y_true_test

    def solve_to_optimality(self, train_size, val_size, test_size, validation=False, test=False, output_folder=""):
        # Generate random noise samples
        self.scenarios_train = [i for i in range(0, train_size)]
        self.scenarios_val = [i for i in range(train_size, train_size + val_size)]
        self.scenarios_test = [i for i in range(train_size + val_size, train_size + val_size + test_size)]
        self.tobjs = []
        y_true_train, y_true_val, y_true_test = [], [], []

        for s in self.scenarios_train:
            scenario_1 = self.plan.create_scenario(s, noise_factor=self.noise_factor)
            durations = []
            for product in scenario_1.production_plan.products:
                for activity in product.activities:
                    sample = activity.processing_time[0]
                    durations.append(max(0, round(sample)))
            y_true_train.append(durations)
            output_location = f'{output_folder}_s_{s}.csv'
            obj, sol = self.solve_and_write(durations, output_location)
            self.tobjs.append(obj)

        for s in self.scenarios_val:
            if validation:
                scenario_1 = self.plan.create_scenario(s, noise_factor=self.noise_factor)
                durations = []
                for product in scenario_1.production_plan.products:
                    for activity in product.activities:
                        sample = activity.processing_time[0]
                        durations.append(max(0, round(sample)))
                y_true_val.append(durations)
                output_location = f'{output_folder}_s_{s}.csv'
                obj, sol = self.solve_and_write(durations, output_location)
                self.tobjs.append(obj)
            else:
                self.tobjs.append(np.inf)

        for s in self.scenarios_test:
            if test:
                scenario_1 = self.plan.create_scenario(s, noise_factor=self.noise_factor)
                durations = []
                for product in scenario_1.production_plan.products:
                    for activity in product.activities:
                        sample = activity.processing_time[0]
                        durations.append(max(0, round(sample)))
                y_true_test.append(durations)
                output_location = f'{output_folder}_s_{s}.csv'
                obj, sol = self.solve_and_write(durations, output_location)
                self.tobjs.append(obj)
            else:
                self.tobjs.append(np.inf)

        return y_true_train, y_true_val, y_true_test

    def evaluate_mu(self, mu, data):
        y_hat = [max(int(round(x)), 0) for x in mu.tolist()]

        # Solve optimization problem with y hat set to mu

        obj, sol = self.solve(y_hat)

        self.plan.set_earliest_start_times(earliest_start=sol)

        if data == "train":
            scenarios = self.scenarios_train
        elif data == "val":
            scenarios = self.scenarios_val
        elif data == "test":
            scenarios = self.scenarios_test
        objs, regrets, post_hoc_regrets = [], [], []

        # Use scenarios to evaluate solution
        for s in scenarios:
            scenario_1 = self.plan.create_scenario(s, noise_factor=self.noise_factor)
            operator = Operator(plan=scenario_1.production_plan, policy_type=2, printing=False)
            my_simulator = Simulator(plan=scenario_1.production_plan, operator=operator, printing=False, penalty=self.penalty, penalty_divide=self.penalty_divide)
            makespan, lateness, nr_unfinished, penalty = my_simulator.simulate(sim_time=9999, write=False)
            obj = makespan
            objs.append(obj)
            regret = obj - self.tobjs[s]
            post_hoc_regret = regret + penalty
            regrets.append(regret)
            post_hoc_regrets.append(post_hoc_regret)

        return objs, regrets, post_hoc_regrets, sol



