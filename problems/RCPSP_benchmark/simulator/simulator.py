import copy
import simpy
import random
import pandas as pd
from collections import namedtuple

"""
This file contains the simulator class, using the SimPy Discrete-Event-Simulation package. This simulator is created for 
a biomanufacturing scheduling application based on our industrial partner. The PSPlib instance do not necessarily need
such a simulator, a different data structure could have been used to better match with the PSPlib instances. However, 
due to time limitations we have chosen to use the industrial simulator with some small adjustments also to evaluate the 
PSPlib instances. For future work a simpler simulation tool could be chosen.
"""

class Simulator:
    def __init__(self, plan, operator, printing=False, penalty=0, penalty_divide=False):
        self.plan = plan
        self.resource_names = plan.factory.resource_names
        self.nr_resources = len(self.resource_names)
        self.capacity = plan.factory.capacity
        self.resources = []
        self.env = simpy.Environment()
        self.resource_usage = {}
        self.printing = printing
        self.nr_clashes = 0
        self.operator = operator
        self.log_start_times = {}
        self.log_end_times = {}
        self.log_earliest_start = {}
        self.penalty = penalty
        self.penalty_divide = penalty_divide

    def activity_processing(self, activity_id, product_id, proc_time, needs):
        """
        :param activity_id: id of the activity (int)
        :param product_id: id of the product (int)
        :param proc_time: processing time of this activity (int)
        :param resources_required: list with SimPy processes for resource requests (list)
        :param resources_names: list with the corresponding resource names (list)
        :return: generator
        """
        # Trace back the moment in time that the resources are requested
        request_time = self.env.now
        if self.printing:
            print(f'At time {self.env.now}: the available resources are {self.factory.items}')

        start_processing = True

        # Check precedence relations (check if minimal difference between start time with predecessors is satisfied)
        predecessors = self.plan.products[product_id].predecessors[activity_id]
        for pred_activity_id in predecessors:
            temp_rel = self.plan.products[product_id].temporal_relations[(pred_activity_id, activity_id)]
            if temp_rel != 0:
                print(f"WARNING: ignoring duration {temp_rel} for temporal relation between activity {activity_id} "
                      f"and predecessor activity {pred_activity_id} in product {product_id}")
            end_pred = self.log_end_times[(product_id, pred_activity_id)]
            if end_pred is None:
                if self.printing:
                    print(f'At time {self.env.now}: product {product_id}, activity {activity_id} cannot start because '
                          f' predecessors {product_id}, {pred_activity_id} did not finish yet')
                start_processing = False


            # FIXME: adding these checks breaks the simulation some how, maybe because of small delays
            #else:

                #if round(end_pred) - round(self.env.now) < 0:
                #    print(f'end pred {end_pred} and now {self.env.now} and {activity_id}')
                # log_end_time should never contain values in the future
                #assert round(end_pred) - round(self.env.now) < 0, "Simulator.log_end_time contains end time in the future?!"

        if start_processing:
            # Check if all resources are available
            for r, need in enumerate(needs):
                if need > 0:
                    resource_name = self.resource_names[r]
                    available_machines = [i.resource_group for i in self.factory.items].count(resource_name)
                    if self.printing:
                        print(f'At time {self.env.now}: we need {need} {resource_name} for product {product_id}, activity {activity_id} and currently '
                              f'in the factory we have {available_machines} available')
                    if available_machines < need:
                        start_processing = False

        # If it is available start the request and processing
        if start_processing:
            self.signal_to_operator = False
            if self.printing:
                print(f'At time {self.env.now}: product {product_id} ACTIVITY {activity_id} requested resources: {needs}')

            # SimPy request
            resources = []
            for r, need in enumerate(needs):
                if need > 0:
                    resource_name = self.resource_names[r]
                    for _ in range(0, need):
                        resource = yield self.factory.get(lambda resource: resource.resource_group == resource_name)
                        resources.append(resource)
            # Trace back the moment in time that the resources are retrieved
            retrieve_time = round(self.env.now)

            if self.printing:
                print(f'At time {self.env.now}: product {product_id} ACTIVITY {activity_id} retrieved resources: {needs}')

            # Trace back the moment in time that the activity starts processing
            start_time = round(self.env.now)
            self.log_start_times[(product_id, activity_id)] = start_time

            # Generator for processing the activity
            # FIXME: here we used a simple hack to facilitate resource availability with a small time difference
            yield self.env.timeout(max(0, proc_time-0.000000001))

            # Trace back the moment in time that the activity ends processing
            end_time = round(self.env.now)
            self.log_end_times[(product_id, activity_id)] = end_time

            # Release the resources that were used during processing the activity
            # For releasing use the SimPy put function from the FilterStore object
            for resource in resources:
                yield self.factory.put(resource)

            if self.printing:
                print(f'At time {self.env.now}: product {product_id} ACTIVITY {activity_id} released resources: {needs}')

            self.resource_usage[(product_id, activity_id)] = \
                                        {"Product": product_id,
                                        "Activity": activity_id,
                                        "Needs": needs,
                                        "resources": resources,
                                        "Request": request_time,
                                        "Retrieve": retrieve_time,
                                        "Start": start_time,
                                        "Finish": end_time}
            #print(self.plan.PRODUCTS[product_ID].ACTIVITIES[activity_ID].start_time)
            #print(self.resource_usage[(product_ID, activity_ID)])

        # If it is not available then we don't process this activity, so we avoid that there starts a queue in the
        # factory
        else:
            if self.printing:
                print(f"At time {round(self.env.now)}: there are no resources available for product {product_id} ACTIVITY {activity_id}, so it cannot start")
            self.operator.signal_failed_activity(product_id=product_id, activity_id=activity_id,
                                                 current_time=round(self.env.now))
            self.nr_clashes += 1

    def activity_generator(self):
        """Generate activities that arrive at the factory based on earliest start times."""

        finish = False

        # Ask operator about next activity
        while not finish:
            send_activity, delay, activity_id, product_id, proc_time, needs, finish = \
                self.operator.send_next_activity(current_time=round(self.env.now))

            if send_activity:
                self.env.process(self.activity_processing(activity_id, product_id, proc_time, needs))

            # Generator object that does a time-out for a time period equal to delay value
            yield self.env.timeout(delay)

    def simulate(self, sim_time=1000, random_seed=1, write=False, output_location="Results.csv"):
        """
        :param SIM_TIME: time allowed for running the discrete-event simulation (int)
        :param random_seed: random seed when used in stochastic mode (int)
        :param write: set to true if you want to write output to a csv file (boolean)
        :param output_location: give location for output file (str)
        :return:
        """
        if self.printing:
            print(f'START factory SIMULATION FOR seed {random_seed}\n')

        # Set random seed
        random.seed(random_seed)

        # Reset environment
        self.env = simpy.Environment()

        for act in self.plan.earliest_start:
            self.resource_usage[(act["product_id"], act["activity_id"])] = {"Product": act["product_id"],
                                        "Activity": act["activity_id"],
                                        "Needs": float("inf"),
                                        "resources": "NOT PROCESSED DUE TO CLASH",
                                        "Request": float("inf"),
                                        "Retrieve": float("inf"),
                                        "Start": float("inf"),
                                        "Finish": float("inf")}

            self.log_start_times[(act["product_id"], act["activity_id"])] = None
            self.log_end_times[(act["product_id"], act["activity_id"])] = None
            self.log_earliest_start[(act["product_id"], act["activity_id"])] = act["earliest_start"]

        # Create the factory that is a SimPy FilterStore object
        self.factory = simpy.FilterStore(self.env, capacity=sum(self.capacity))

        # Create the resources that are present in the SimPy FilterStore
        Resource = namedtuple('Machine', 'resource_group, id')
        items = []
        for r in range(0, self.nr_resources):
            for j in range(0, self.capacity[r]):
                resource = Resource(self.resource_names[r], j)
                items.append(copy.copy(resource))
        self.factory.items = items

        # Execute the activity_generator
        self.env.process(self.activity_generator())
        self.env.run(until=sim_time)

        # Process results
        resource_usage_df = []
        for i in self.resource_usage:
            product_id = self.resource_usage[i]["Product"]
            activity_id = self.resource_usage[i]["Activity"]
            earliest_start = self.log_earliest_start[(product_id, activity_id)]
            self.resource_usage[i]["Earliest_start"] = earliest_start
            resource_usage_df.append(self.resource_usage[i])

        self.resource_usage = pd.DataFrame(resource_usage_df)
        earliest_start = self.resource_usage["Earliest_start"].tolist()
        realized_start = self.resource_usage["Start"].tolist()
        start_difference = [realized_start[i] - earliest_start[i] for i in range(len(earliest_start))]
        total_penalty = (sum(start_difference) * self.penalty)
        if self.penalty_divide:
            total_penalty = total_penalty / len(start_difference)
        if self.printing:
            print(f' \nSIMULATION OUTPUT\n {self.resource_usage}')
        finish_times = self.resource_usage["Finish"].tolist()
        finish_times = [i for i in finish_times if i != float("inf")]
        makespan = max(finish_times)
        lateness = 0

        nr_unfinished_products = 0
        for i, p in enumerate(self.plan.products):
            schedule = self.resource_usage[self.resource_usage["Product"] == i]
            finish = max(schedule["Finish"])
            if finish == float("inf"):
                nr_unfinished_products += 1
                if self.printing:
                    print(f'Product {i} did not finish, while the deadline was {self.plan.products[i].deadline}.')

            else:
                lateness += max(0, finish - self.plan.products[i].deadline)
                if self.printing:
                    print(f'Product {i} finished at time {finish}, while the deadline was {self.plan.products[i].deadline}.')

        if self.printing:
            print(f"The makespan corresponding to this schedule is {makespan}")
            print(f"The lateness corresponding to this schedule is {lateness}")
            print(f"The number of unfinished products is {nr_unfinished_products}")

        if write:
            self.resource_usage.to_csv(output_location)
        return makespan, lateness, nr_unfinished_products, total_penalty

