"""优化方法
对关键工序进行优化: 目前支持对单资源的顺序优化.
- 同资源优化: 遗传算法调整permutation, 实现交期延误较少的原则
- 跨资源: 同样的优化耗时会比较久, 因此通过手工调整甘特图来实现.
- 跨资源建模: 基因序列长度 n_jobs x n_machines,
- 关键工序识别优化:


文档:
https://teletraan.feishu.cn/wiki/AxcfwhQ8WiJ42bkUwmAcmpaGnec?appStyle=UI4&domain=www.feishu.cn&locale=zh-CN&refresh=1&tabName=wiki&userId=7073353126295928833
"""
import time
import logging
from copy import copy
from typing import Union
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.survival import Survival
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.util.display.output import Output
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.util.display.single import SingleObjectiveOutput
from pymoo.operators.crossover.binx import mut_binomial
from pymoo.operators.mutation.nom import NoMutation
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.duplicate import DefaultDuplicateElimination, DuplicateElimination
from pymoo.core.selection import Selection


class OPTSolver(object):
    """优化入口: 优化同一个资源上的GroupOP顺序"""

    def __init__(
        self,
        population_size: int = 10,
        num_generation: int = 5,
        time_limit_minutes: int = 5,
        seed: int = 43,
        verbose: int = 1,
    ) -> None:
        self.population_size = population_size
        self.num_generation = num_generation
        self.time_limit_minutes = time_limit_minutes
        self.seed = seed
        self.verbose = verbose

    def run(self, resource) -> None:
        start_time = time.time()
        initial_kpi: dict[str, float] = {}
        initial_kpi = ", ".join(f"{key}: {value}" for key, value in initial_kpi.items())
        logging.info(
            f"资源:  启动遗传算法优化, GroupOP数量{len(resource.assigned_group_op_list)}, 初始指标: {initial_kpi}"
        )
        problem = ResourceProblem(resource=resource)
        algorithm = NSGA2(pop_size=self.population_size, eliminate_duplicates=True)
        algorithm.termination = DefaultSingleObjectiveTermination()
        res = minimize(
            problem,
            algorithm,
            termination=("n_gen", self.num_generation),
            seed=self.seed,
            verbose=self.verbose,
        )

        # 更新结果
        new_order = problem.decoder.get_order(res.X)
        print(new_order)

        if len(np.array(new_order).shape) > 1:
            new_order = new_order[0]

        resource.assigned_group_op_list = [
            problem.decoder.original_group_op_list[i] for i in new_order
        ]
        resource.reload_by_optimized_groupop()

        print([i.id for i in resource.assigned_group_op_list])

        opt_kpi: dict[str, float] = {}
        opt_kpi = ", ".join(f"{key}: {value}" for key, value in opt_kpi.items())
        logging.info(
            f"资源  遗传算法优化完成, 耗时{time.time() - start_time}, 优化后指标: {opt_kpi}"
        )

    def parallel_run(self, resources) -> None:
        cpu_count: int = os.cpu_count() or 1
        Parallel(n_jobs=cpu_count // 2)(
            delayed(self.run)(resource) for resource in resources
        )


class ResourceProblem(ElementwiseProblem):
    """问题定义函数
    ElementwiseProblem: implements a function evaluating a single solution at a time
    """

    def __init__(self, resource) -> None:
        n_var = len(resource.assigned_group_op_list)
        n_obj = 1
        xl: float | np.ndarray = np.zeros(n_var)
        xu: float | np.ndarray = np.ones(n_var)
        self.decoder = CustomDecoder(resource)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)

    def _evaluate(self, x: np.ndarray, out) -> None:
        tardiness = self.decoder.get_tardiness(x)
        # changover = self.resource.get_total_changeover_times()
        out['F'] = tardiness


class CustomDecoder:
    """解码: 问题转化为适合遗传等算法解决的方式(encode) 再转化回来 (decode)"""

    def __init__(self, resource) -> None:
        self.resource = resource
        self.original_group_op_list = copy(self.resource.assigned_group_op_list)
        self.original_order = np.array(range(len(resource.assigned_group_op_list)))
        self.num_freeze_task = 1

    def get_order(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 1:
            idx = np.concatenate([self.original_order[:self.num_freeze_task], self.num_freeze_task + np.argsort(x[self.num_freeze_task:])])
        else:
            x1 = self.original_order[:self.num_freeze_task]
            x1 = np.tile(x1.reshape(-1, self.num_freeze_task), (x.shape[0], 1))
            print(x1.shape, x.shape)
            idx = np.concatenate([x1, self.num_freeze_task + np.argsort(x[:, self.num_freeze_task:])], axis=1)


        new_order = self.original_order[idx]
        return new_order

    def get_tardiness(self, x) -> float:
        """延误指标
        - 注意不同类型权重是不同的
        """
        new_order = self.get_order(x)

        self.resource.assigned_group_op_list = [self.original_group_op_list[i] for i in new_order]

        self.resource.update_working_time_from_groupop_sequence()

        tardiness: float = 0
        for group_op in self.resource.assigned_group_op_list:
            tardiness += group_op.tardiness_hours
        #print(new_order, tardiness)
        return tardiness

    def get_changeover(self):
        return self.resource.get_total_changeover_times()


# ------------------------测试实例------------------------

class GroupOperation(object):
    def __init__(self, id, demand_date, processing_hours):
        self.id = id
        self.processing_hours = processing_hours
        self.demand_date = demand_date
        self.assigned_start_time = None
        self.assigned_end_time = None

    @property
    def tardiness_hours(self):
        return max(0, self.assigned_end_time - self.demand_date)


class ResourceExtend(object):
    def __init__(self):
        self.assigned_group_op_list = []

    def update_working_time_from_groupop_sequence(self, ):
        time = 0

        for item in self.assigned_group_op_list:
            item.assigned_start_time = time
            end = time + item.processing_hours
            item.assigned_end_time = end
            item.tardiness = max(0, end - item.demand_date)
            time = end

    def get_total_changeover_times(self):
        return 0

    def reload_by_optimized_groupop(self):
        pass


groupop1 = GroupOperation(id=10, demand_date=4, processing_hours=3)
groupop2 = GroupOperation(id=11, demand_date=10, processing_hours=7)
groupop3 = GroupOperation(id=12, demand_date=15, processing_hours=6)
groupop4 = GroupOperation(id=13, demand_date=20, processing_hours=5)
groupop5 = GroupOperation(id=14, demand_date=20, processing_hours=3)

resource = ResourceExtend()
resource.assigned_group_op_list = [groupop4, groupop1, groupop2, groupop3, groupop2]

OPTSolver().run(resource)
