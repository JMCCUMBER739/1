"""Dispatching rules"""

from lekin.solver.construction_heuristics.atcs import ATCScheduler
from lekin.solver.construction_heuristics.epst import EPSTScheduler
from lekin.solver.construction_heuristics.lpst import LPSTScheduler
from lekin.solver.construction_heuristics.spt import SPTScheduler

__all__ = [ATCScheduler, LPSTScheduler, SPTScheduler]


class RuleScheduler(object):
    def __init__(self):
        pass
