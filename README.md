# decision_focused_scheduling

This repository contains the code that corresponds to the CPAIOR 2024 submission Learning From Scenarios for Repairable Stochastic Scheduling by Kim van den Houten et al.

## Installation and practical issues
Besides the installation of the requirements.txt, it is needed to install the IBM Cplex Optimization Studio in order to be able to run all experiments in this repository. Furthermore, for all scripts in this repository it holds that the working directory should be set to "LOCATION_PATH/decision_focused_scheduling", where LOCATION_PATH should be replaced with the users location for this repository.

## Instances
The experiments shown in the papers use the following instances:

- PSPlib j30 instances: j301_1 - j303_10 (30 instances). 

- PSPlib j90 instances: j901_1 - j903_10 (30 instances).

- Small industry instances: 5_1_factory 1, 5_2_factory_1, 5_3_factory 1, 5_4_factory_1, 5_5_factory_1, 10_1_factory 1, 10_2_factory_1, 10_3_factory 1, 10_4_factory_1, 10_5_factory_1, 20_1_factory 1, 20_2_factory_1, 20_3_factory 1, 20_4_factory_1, 20_5_factory_1.

- Large industry instances: 40_1_factory 1, 40_2_factory_1, 40_3_factory 1, 40_4_factory_1, 40_5_factory_1, 60_1_factory 1, 60_2_factory_1, 60_3_factory 1, 60_4_factory_1, 60_5_factory_1.

## Hyperparameters

Decision-focused learning: Pytorch ADAM optimizer with default settings, batch size is 10, random shuffling is true. 
Deterministic model: IBM CP Optimizer with default settings.
Stochastic model: IBM CP Optimer with default settings. 

Settings per instance set:

PSPlib j30 instances: time budget = 30 min, the number of scenarios:

- deterministic, 100 scenarios
- stochastic, 50 scenarios
- decision-focused, 50 scenarios

PSPlib j90 instances: time budget = 60 min, the number of scenarios:

- deterministic, 100 scenarios
- stochastic, 50 scenarios
- decision-focused, 50 scenarios

Small industry instances: time budget: 5_i instances 600 sec, 10_i instances 1200 sec, 20_i instances 2400 sec, the number of scenarios:

- deterministic, 100 scenarios
- stochastic, 10 scenarios
- decision-focused, 25 scenarios

Large industry instances: time budget: 2_i instances 4800 sec, 60_i instances 3600 sec, the number of scenarios:

- deterministic, 100 scenarios
- stochastic, 10 scenarios
- decision-focused, 25 scenarios

## Experiments scripts


## Results

The results are provided in the results folder, in which a separate folder exists for each instance set, comprising two folders for the two different penalty settings (small and large). To analyze the significance of the obtained results, one could run experiment_scripts/paper_significance.py. 