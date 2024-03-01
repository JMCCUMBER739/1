import json
import logging
import time

import numpy as np

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
        pass


class GeneticOPT(object):
    def __init__(self):
        pass

    def run(self):
        start_time = time.time()
        while True:
            np.random.seed(int(time.time()))


def main():
    data_reader = DataReader()

    opt = GeneticOPT()
    opt.run()


if __name__ == "__main__":
    pass
