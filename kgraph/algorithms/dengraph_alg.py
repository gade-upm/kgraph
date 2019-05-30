import sys
import time

import graph_tool.all as gt
from dengraph.dengraph import DenGraphIO
from dengraph.graphs.distance_graph import DistanceGraph

from kgraph.algorithms import Algorithm


class DengraphAlg(Algorithm):
    def __init__(self, graph, cluster_distance, core_neighbours, output):
        super(DengraphAlg, self).__init__('dengraph', graph, output)
        self._dg_network = self.__translate_graph()
        self._cluster_distance = cluster_distance
        self._core_neighbours = core_neighbours

    def run(self, draw_graph=None):
        sys.stdout.write('Computing clusters ... ')
        sys.stdout.flush()
        init = time.time()
        clusters = DenGraphIO(self._dg_network, cluster_distance=self._cluster_distance,
                              core_neighbours=self._core_neighbours).clusters
        end = time.time()
        sys.stdout.write('Ok!\n')
        sys.stdout.flush()

        communities = [sorted(cluster) for cluster in sorted(clusters, key=lambda clstr: min(clstr))]
        print('Algorithm finished in {0} s.'.format(end - init))
        if draw_graph and len(communities) > 0:
            for idx, community in enumerate(communities):
                self.paint(community, idx)
                print('community {0} saved.'.format(idx + 1))
        return communities

    def __translate_graph(self):
        graph = DistanceGraph(
            nodes=tuple([v for v in self._network.vertices()]),
            distance=lambda node_from, node_to: self.__paper_distance(node_from, node_to)
        )

        return graph

    @staticmethod
    def __paper_distance(node_from, node_to):
        if node_from == node_to:
            return 0
        I_pq = len([(s, t) for s, t in node_from.out_edges() if s == node_from and t == node_to])
        I_qp = len([(s, t) for s, t in node_to.out_edges() if s == node_to and t == node_from])
        if I_pq > 1 and I_qp > 1:
            return 1.0 / min(I_pq, I_qp)
        else:
            return 1

    def __distance(self, node_from, node_to):
        if node_from == node_to:
            return 0
        v_l, e_l = gt.shortest_path(self._network, node_from, node_to)
        if len(v_l) == 0:
            return 1
        else:
            return 1 - (1.0 / len(v_l))
