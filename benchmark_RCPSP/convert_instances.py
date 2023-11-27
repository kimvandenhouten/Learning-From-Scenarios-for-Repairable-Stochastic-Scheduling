import pandas as pd
import numpy as np
from problems.RCPSP_benchmark.simulator.distributions import NormalDistribution
from problems.RCPSP_benchmark.simulator.classes import Factory, Product, Activity, ProductionPlan
from problems.RCPSP_benchmark.simulator.simulator import Simulator
from problems.RCPSP_benchmark.simulator.operator import Operator
from benchmark_RCPSP.process_file import process_file

# Read instance information and set up instance class
instance_folder = "j90"

for a in range(1, 49):
    for b in range(1, 11):
        instance_name = f'{instance_folder}{a}_{b}'
        activities, precedence_relations, resources, durations, capacity, needs = process_file(f"benchmark_RCPSP/instances/{instance_folder}/",
                                                                                               f"{instance_name}.sm")
        # make factory
        print(instance_name, resources, capacity)
        factory = Factory(name=instance_name, resource_names=resources, capacity=capacity, products=None)

        temporal_relations = []
        for (pred, suc) in precedence_relations:
            temporal_relations.append({"predecessor": pred,
                                            "successor": suc,
                                            "rel": 0})

        product = Product(id=0, name=instance_name, temporal_relations=temporal_relations)
        for i, act in enumerate(activities):
            if durations[i] > 0:
                processing_time = durations[i]
                default_variance = np.sqrt(processing_time)
                distribution = NormalDistribution(processing_time, default_variance)
            else:
                distribution = None
            activity = Activity(id=i, processing_time=[durations[i], durations[i]], product=0, product_id=0, needs=needs[i], distribution=distribution)
            product.add_activity(activity)

        product.set_temporal_relations(product.temporal_relations)
        factory.add_product(product)
        productionplan = ProductionPlan(id=1, size=1, name=instance_name, factory=factory, product_ids=[0], deadlines=[0])
        productionplan.list_products()
        productionplan.to_json()

        json_str = productionplan.to_json()

        file_name = f'problems/RCPSP_benchmark/instances/instance_{instance_name}.json'
        print(file_name, ' created')

        with open(file_name, 'w+') as file:
            file.write(json_str)
