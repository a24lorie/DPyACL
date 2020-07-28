import json
import os
import time
import unittest

from dask.distributed import Client
# from sklearn.datasets import load_breast_cancer
from dask_ml.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from dpyacl.core.stop_criteria import MaxIteration
from dpyacl.experiment.context import HoldOutExperiment
from dpyacl.metrics import Accuracy
from dpyacl.oracle import SimulatedOracle
from dpyacl.scenario.scenario import PoolBasedSamplingScenario
from dpyacl.strategies.single_label import QueryMarginSampling


class TestDatasets_1_3_5_7(unittest.TestCase):

    __client = Client('tcp://192.168.2.100:8786')

    _split_count = 10
    _feature_num = 30
    _label_num = 2
    _instance_num = 12000

    _ml_technique = LogisticRegression(solver='sag')
    _query_strategy = QueryMarginSampling()
    _performance_metrics = [Accuracy()]



    def execute_experiment(self, num_iters, file_name):
        for i in range(0, num_iters):
            X, y = make_classification(n_samples=self._instance_num, n_features=self._feature_num,
                                       n_informative=2 * self._label_num,
                                       n_redundant=self._label_num,
                                       n_repeated=0,
                                       n_classes=self._label_num,
                                       n_clusters_per_class=self._label_num,
                                       weights=None,
                                       flip_y=0.01,
                                       class_sep=1.0,
                                       hypercube=True,
                                       shift=0.0,
                                       scale=1.0,
                                       shuffle=True,
                                       random_state=None,
                                       chunks=self._instance_num * 0.10)


            experiment = HoldOutExperiment(
                self.__client,
                X,
                y,
                scenario_type=PoolBasedSamplingScenario,
                ml_technique=self._ml_technique,
                performance_metrics=self._performance_metrics,
                query_strategy=self._query_strategy,
                oracle=SimulatedOracle(labels=y),
                stopping_criteria=MaxIteration(25),
                self_partition=True,
                test_ratio=0.3,
                initial_label_rate=0.05,
                all_class=True,
                batch_size=100,
                rebalance=True
            )

            start_time = time.time()
            experiment.evaluate(client=self.__client, multithread=False, verbose=True)
            end_time = time.time() - start_time
            self.dump_iteration(file_name, {"iter":i+1, "time":end_time})

    def dump_iteration(self, file_name, iteration):
        iterations = []
        if not os.path.isfile(file_name):
            iterations.append(iteration)
            with open(file_name, mode='w') as f:
                f.write(json.dumps(iterations))
        else:
            with open(file_name) as feedsjson:
                iterations = json.load(feedsjson)

            iterations.append(iteration)
            with open(file_name, mode='w') as f:
                f.write(json.dumps(iterations))


    def test_1_worker(self):
        self.execute_experiment(30, "result_1_worker.json")

    def test_3_workers(self):
        self.execute_experiment(30, "result_3_worker.json")

    def test_5_workers(self):
        self.execute_experiment(30, "result_5_worker.json")


if __name__ == '__main__':
    unittest.main()
