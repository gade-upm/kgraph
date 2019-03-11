import random
import time
import numpy as np
from kgraph.algorithms import Algorithm
from kgraph.utils.graph import isolation_index, cohesion_index
import graph_tool.all as gt

from kgraph.utils.graph import paint_graph


class GRASP(Algorithm):
    """
    Multiobjetive GRASP with Path Relinking
    Construction: Secuential combined
    Local Search: secuential combined

    Intensification Path Relinking
    """

    def __init__(self, graph, output):

        self._cohesion_factor = 0.8  # For kores_alg
        self.ES = []
        self.nondominatedI = []
        self.nondominatedC = []

        super(GRASP, self).__init__('GRASP', graph, output)

    def run(self, draw_graph=None):
        init = time.time()
        seeds = self._seeds()

        if draw_graph:
            self._paint_seeds(seeds)

        for seed in seeds:
            # TODO: Stopping criterion
            # while GRASP stopping criterion not satisfied:
            count = 0
            while count < 1:
                count = count + 1
                candidates = self._candidateList(seed)
                solution = self._greedy_randomized_solution(seed, candidates)
                solution = self._localSearch(solution)
                self._updateSolution(solution)

        end = time.time()
        print('GRASP finished in {0} s.'.format(end - init))

        from matplotlib import pyplot as plt
        plt.scatter(self.nondominatedI, self.nondominatedC)
        plt.show()


        init = time.time()
        bestCommunity = self._pathRelinking()
        end = time.time()
        print('Path RElinking finished in {0} s.'.format(end - init))

        if draw_graph:
            self._paint_communities(bestCommunity)

        return bestCommunity

    def _paint_communities(self, bestCommunity):
        """
        Paint all Graphs in ES
        :param ES:
        :param nondominated:
        :return:
        """
        from matplotlib import pyplot as plt
        plt.scatter(self.nondominatedI, self.nondominatedC)

        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }

        plt.title('Non Dominated', fontdict=font)
        plt.xlabel('time (s)', fontdict=font)
        plt.ylabel('voltage (mV)', fontdict=font)
        plt.show()
        paint_graph('test/', self._network, bestCommunity)

    def _candidateList(self, seed):
        """
        CREATE Candidates, with the candidate elements that can be added to the solution
        :return: candidates
        """
        # Isolated vertices
        vertices = [self._network.vertex_index[v] for l in seed for v in l]
        all_vertices = [self._network.vertex_index[v] for v in self._network.vertices()]
        isolated = [self._network.vertex(v) for v in all_vertices if v in all_vertices and v not in vertices]

        return isolated

    def _greedy_randomized_solution(self, seed, candidates):
        """
        Greedy
        Probabilistic
        Adaptative

        :param seed: Communities of the graph
        :param candidates: Candidates to add
        :return:
        """
        communities = list(seed)
        # TODO: Define complete
        # while is not complete:
        count = 0
        while count < 2:
            count = count + 1

            # TODO: Restricted for multi-objetive?
            RCL = self._constructRCL(candidates, communities)  # Greedy
            if RCL:
                candidate = random.choice(RCL)  # Probabilistic

                # TODO: Add solution
                communities[:] = communities + [candidate]

                # TODO: Remove candidate
                candidates.remove(candidate)
                #candidates = self._candidateList(communities, candidates)  # Adaptative
        return communities

    def _localSearch(self, seed):
        """
        Local Search algorithm
        :return: local optima solution
        """
        # Find solution that is better
        communities = self._improve_seeds(seed)

        return communities

    # TODO: Path Relinking
    def _pathRelinking(self):
        """
        Intensification
        :return:
        """
        return random.choice(self.ES)

    def _updateSolution(self, solution):
        """
        Return the best communities
        :param communities, communities
        """
        Ia, Ic = self._compute_results(solution)
        dominated = False

        # if ES is empty
        if not self.ES:
            dominated = True

        # TODO: NoN Dominated check
        for idx, e in enumerate(self.ES):
            if self.nondominatedI[idx] < Ia or self.nondominatedC[idx] < Ic:
                dominated = True
                if self.nondominatedI[idx] < Ia and self.nondominatedC[idx] < Ic \
                        or self.nondominatedI[idx] <= Ia and self.nondominatedC[idx] < Ic\
                        or self.nondominatedI[idx] < Ia and self.nondominatedC[idx] <= Ic:
                    print(len(self.ES))
                    self.ES.pop(idx)
                    self.nondominatedI.pop(idx)
                    self.nondominatedC.pop(idx)
                    print(len(self.ES))

        if dominated:
            self.ES.append(solution)
            self.nondominatedI.append(Ia)
            self.nondominatedC.append(Ic)



    # TODO: From candidates, return the ones with good g(c)
    def _constructRCL(self, candidates, communities):
        """
        Construct Restricted candidate list
        :return: RCL
        """
        RCL = []

        for communitie in communities:
            for vertex in communitie:
                vertex

        # for c in candidates:
        # TODO: define function g(c)
        # g =
        # if g(c) has a good value:
        #    RCL.append(c)
        return RCL

    def _seeds(self):
        """
        Seeds generated from the components obtained by the cuts of all the possible k-cores of the graph
        :return: seeds
        """
        network = self._network.copy()
        kcores = gt.kcore_decomposition(network)
        max_core = np.max(kcores.a)
        min_core = 2
        seeds = []
        for n in range(min_core, max_core + 1):
            k_component = self._components(n, kcores, network)
            if len(k_component) > 0:
                seeds.append(k_component)
        return seeds

    def _compute_results(self, communities):
        """
        Compute means of Isolation and Cohesion
        :param communities:
        :return: Ia, Ic
        """
        values = []
        for idx, community in enumerate(communities):
            ia = isolation_index(community)
            ic = cohesion_index(self._network, community)
            values.append((len(community), ia, ic))

        Ia = np.mean([v[1] for v in values])
        Ic = np.mean([v[2] for v in values])

        return Ia, Ic

    def _paint_seeds(self, seeds):
        """
        Paint different cluster seeds
        :param seeds:
        :return:
        """
        for i, seed in enumerate(seeds):
            for idx, s in enumerate(seed):
                self.paint(s, idx + i * 10)

    def _components(self, k, kcores, network):
        network_cpy = network.copy()
        network_cpy.vp['GRASP'] = network_cpy.new_vertex_property('bool')
        for v in network_cpy.vertices():
            if kcores[v] >= k:
                network_cpy.vp['GRASP'][v] = True
            else:
                network_cpy.vp['GRASP'][v] = False
        network_cpy.set_vertex_filter(network_cpy.vp['GRASP'])
        network_cpy.purge_vertices()

        labels, hist = gt.label_components(network_cpy, directed=False)
        components = self.__group_labels(labels, network_cpy)

        mapped = self._map_vertexs(network_cpy, components)

        return mapped

    def _map_vertexs(self, graph, seeds):
        components = [[gt.find_vertex(self._network, self._network.vp['hash'], graph.vp['hash'][v])[0] for v in seed]
                      for seed in seeds]

        return components

    def __group_labels(self, labels, graph):
        labels_arr = labels.a
        uniques = np.unique(labels_arr)
        components = []
        for unique in uniques:
            idxs = np.where(labels_arr == unique)[0]
            seed = [graph.vertex(idx) for idx in idxs]
            components.append(seed)

        return components

    def __expand(self, seed, isolated_vertices, ic0):
        newseed = list(seed)
        for vertex in newseed:
            in_neighbours = list(vertex.in_neighbors())
            out_neighbours = list(vertex.out_neighbors())
            neighbours = set(in_neighbours + out_neighbours)

            p_candidates = [candidate for candidate in list(neighbours.intersection(isolated_vertices)) if
                            candidate not in seed]
            if len(p_candidates) > 0:
                for candidate in p_candidates:
                    isolation_before = isolation_index(seed)
                    new_seed = list(seed) + [candidate]
                    isolation_after = isolation_index(new_seed)
                    cohesion_after = cohesion_index(self._network, new_seed)
                    if isolation_after > isolation_before and cohesion_after >= self._cohesion_factor * ic0 and candidate not in seed:
                        seed[:] = seed + [candidate]

    def _expand_seed(self, seed, isolated_vertices, ic0):
        l_seed = len(seed)
        self.__expand(seed, isolated_vertices, ic0)
        l_newseed = len(seed)
        if l_seed == l_newseed:
            return seed
        else:
            return self._expand_seed(seed, isolated_vertices, ic0)

    def _improve_seeds(self, seeds, draw_seeds=None):
        communities = []
        isolated_vertices = self.__get_isolated_vertices(seeds)
        for idx, seed in enumerate(seeds):
            ic0 = cohesion_index(self._network, seed)
            self._expand_seed(seed, isolated_vertices, ic0)
            if draw_seeds:
                self.paint(seed, idx)
            communities.append(seed)

        return communities

    def __get_isolated_vertices(self, seeds):
        vertices = [self._network.vertex_index[v] for l in seeds for v in l]
        all_vertices = [self._network.vertex_index[v] for v in self._network.vertices()]
        isolated = [self._network.vertex(v) for v in all_vertices if v in all_vertices and v not in vertices]

        return isolated
