import random
import sys
import time
import numpy as np
import randomcolor
import os.path

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
        self.alpha = 0.5  # Alpha for selecting best in Restricted Candidate List
        self.ES = []
        self.nondominated
        self.max_iterations = 10
        super(GRASP, self).__init__('GRASP', graph, output)

    def run(self, draw_graph=None):
        init = time.time()
        seeds = self._seeds()

        if False:
            self._paint_seeds(seeds)

        for seed in seeds:
            # TODO: Stopping criterion
            # Max iterations
            for i in range(self.max_iterations):
                candidates = self._candidateList(seed)
                if not candidates:
                    break
                solution, candidates = self._greedy_randomized_solution(seed, candidates)
                solution = self._localSearch(solution)
                self._updateSolution(solution)

        end = time.time()
        print('GRASP finished in {0} s.'.format(end - init))

        init = time.time()
        # TODO: Path Relinking
        nondominated_communities = self.ES
        # nondominated_communities = self._pathRelinking()
        end = time.time()
        print('Path RElinking finished in {0} s.'.format(end - init))

        if draw_graph:
            self._paint_communities(nondominated_communities)

        return nondominated_communities

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
        while count < 1:
            count = count + 1

            # Restricted for multi-objetive
            rcl, rcl_ids = self._constructRCL(candidates, communities)  # Greedy
            if rcl:
                # Random selection
                random_number = random.randrange(len(rcl))
                candidate = rcl[random_number]
                idc = rcl_ids[random_number]

                # Add solution
                communities[idc].append(candidate)

                # Remove candidate
                candidates.remove(candidate)  # Adaptative, we remove and then recalculate them
                if not candidates:
                    break
            else:
                break
        return communities, candidates

    def _localSearch(self, seed):
        """
        Local Search algorithm
        :return: local optima solution
        """
        # Find solution that is better
        communities = self._improve_seeds(seed)

        return communities

    def _pathRelinking(self):
        """
        Intensification
        :return:
        """
        nondominated = []

        for idx, e in enumerate(self.ES):
            for idx2, e2 in enumerate(self.ES):
                if idx != idx2:
                    # TODO: Path Relinking
                    solution = self._apply_pathRelinking(e, e2)
                    # Add only non dominated solutions
                    nondominated = self._add_non_dominated(nondominated, solution)

        return nondominated

    # TODO: Path Relinking
    def _apply_pathRelinking(self, x, y):

        return x

    def _add_non_dominated(self, nondominated, solution):
        """
        Add non dominated solution and remove dominated
        :param nondominated:
        :param solution:
        :return: ES, nondominated
        """
        isolation, cohesion = self._compute_results(solution)
        non_dominated = True

        for non in nondominated:
            if non[0] > isolation and non[1] > cohesion:
                non_dominated = False
                break

        if non_dominated:
            # Remove dominated elements
            for idx, non in enumerate(nondominated):
                if non[0] < isolation and non[1] < cohesion:
                    nondominated.pop(idx)
            nondominated.append([isolation, cohesion])
        return nondominated

    def _updateSolution(self, solution):
        """
        Add communities
        :param solution
        """
        self.ES.append(solution)
        # self.nondominated.append([self._compute_results(solution)])

    def _constructRCL(self, candidates, communities):
        """
        Construct Restricted candidate list with the best of both objetives(Isolation and cohesion)
        :return: RCL, rcl_ids_community
        """
        rcl = []  # Restricted candidate list
        rcl_ids_community = []  # Id community
        possibles = []
        possibles_id = []
        g_isolation = []
        g_cohesion = []

        for candidate in candidates:
            for id_community, community in enumerate(communities):
                new_community = community + [candidate]
                # Index of the new communities
                g_isolation.append(isolation_index(new_community))
                g_cohesion.append(cohesion_index(self._network, new_community))
                possibles.append(candidate)
                possibles_id.append(id_community)

        # Select min/max for add to RCL the best candidates
        g_min_isolation = min(g_isolation)
        g_max_isolation = min(g_isolation)

        limit_g_isolation = g_min_isolation + self.alpha * (g_max_isolation - g_min_isolation)

        # Select min/max for add to RCL the best candidates
        g_min_cohesion = min(g_cohesion)
        g_max_cohesion = max(g_cohesion)

        # g(c) formula
        limit_g_cohesion = g_min_cohesion + self.alpha * (g_max_cohesion - g_min_cohesion)

        for id_candidate, localg in enumerate(g_cohesion):
            if localg <= limit_g_cohesion:
                rcl.append(possibles[id_candidate])
                rcl_ids_community.append(possibles_id[id_candidate])
            # Can be added two times if is good in the two objetives
            if g_isolation[id_candidate] <= limit_g_isolation:
                rcl.append(possibles[id_candidate])
                rcl_ids_community.append(possibles_id[id_candidate])

        return rcl, rcl_ids_community

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
        :return: isolation, cohesion
        """
        values = []
        for idx, community in enumerate(communities):
            ia = isolation_index(community)
            ic = cohesion_index(self._network, community)
            values.append((len(community), ia, ic))

        isolation = np.mean([v[1] for v in values])
        cohesion = np.mean([v[2] for v in values])

        return isolation, cohesion

    def _paint_communities(self, nondominated_communities):
        """
        Paint all Graphs in ES
        :param ES:
        :param nondominated:
        :return:

        """
        '''
        from matplotlib import pyplot as plt

        x, y = zip(*self.nondominated)
        plt.scatter(x, y)

        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }

        plt.title('Non Dominated', fontdict=font)
        plt.xlabel('Isolation', fontdict=font)
        plt.ylabel('Cohesion', fontdict=font)
        plt.savefig('./test/Non_Dominated.png')
        plt.show()
        '''
        for idx, communities in enumerate(nondominated_communities):
            self.paint_graph('test/', str(idx), self._network, communities)

    def _paint_seeds(self, seeds):
        """
        Paint different communities from seeds
        :param seeds:
        :return:
        """
        for i, communities in enumerate(seeds):
            for idx, community in enumerate(communities):
                self.paint(community, idx + i * 10)

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
        components = self._group_labels(labels, network_cpy)

        mapped = self._map_vertexs(network_cpy, components)

        return mapped

    def _map_vertexs(self, graph, seeds):
        components = [[gt.find_vertex(self._network, self._network.vp['hash'], graph.vp['hash'][v])[0] for v in seed]
                      for seed in seeds]

        return components

    def _group_labels(self, labels, graph):
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

    def paint_graph(self, path, location, graph, communities):
        if path:
            sys.stdout.write('Drawing graph ... ')
            sys.stdout.flush()
            network = gt.Graph(graph, directed=False)
            folder = os.path.abspath(path)
            # colors = random.sample(range(100,151), len(communities))
            r_cols = randomcolor.RandomColor().generate(count=len(communities) + 1)
            colors = [list(int(r_col[1:][i:i + 2], 16) for i in (0, 2, 4)) for r_col in r_cols]

            # color = graph.new_vertex_property('vector<float>')
            color = network.new_vertex_property('int')

            base_color = colors.pop()
            for v in network.vertices():
                color[v] = (base_color[0] << 16) + (base_color[1] << 8) + base_color[2]
            for community in communities:
                c = colors.pop()
                for v in community:
                    color[v] = (c[0] << 16) + (c[1] << 8) + c[2]
            pos = gt.sfdp_layout(network)
            gt.graph_draw(network, pos=pos, vertex_fill_color=color,
                          output=os.path.join(folder, location + 'graph-communities.svg'))
            sys.stdout.write('Ok!\n')
            sys.stdout.flush()
