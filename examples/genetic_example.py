import json
import logging
import time

import numpy as np
import pandas as pd

from lekin.dashboard.gantt import get_scheduling_res_from_all_jobs, plot_gantt_chart
from lekin.lekin_struct import (
    Job,
    JobCollector,
    Operation,
    OperationCollector,
    Resource,
    ResourceCollector,
    Route,
    RouteCollector,
)
from lekin.solver.construction_heuristics import EPSTScheduler, LPSTScheduler
from lekin.solver.meta_heuristics.genetic import GeneticScheduler

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


class DataReader(object):
    def __init__(self):
        machine_sequence_df = pd.read_excel("./data/JSP_dataset.xlsx", sheet_name="Machines Sequence", index_col=[0])
        processing_time_df = pd.read_excel("./data/JSP_dataset.xlsx", sheet_name="Processing Time", index_col=[0])

        num_jobs = processing_time_df.shape[0]
        num_machines = processing_time_df.shape[1]

        processing_time = [list(map(int, processing_time_df.iloc[i])) for i in range(num_jobs)]
        machine_sequence = [list(map(int, machine_sequence_df.iloc[i])) for i in range(num_jobs)]
        print(processing_time)
        print(machine_sequence)

    def get_job_collector(self):
        return

    def get_resource_collector(self):
        return


class GeneticOPT(object):
    def __init__(self):
        self.genetic = GeneticScheduler(
            popolation_size=30, crossover_rate=0.8, mutation_rate=0.2, mutation_selection_rate=0.2, num_iterations=2000
        )

    def run(self):
        start_time = time.time()
        while True:
            np.random.seed(int(time.time()))


def main():
    data_reader = DataReader()

    # job_collector = data_reader.get_job_collector()
    # resource_collector = data_reader.get_resource_collector()
    #
    # opt = GeneticOPT(job_collector, resource_collector)
    # opt.run()


if __name__ == "__main__":
    main()
