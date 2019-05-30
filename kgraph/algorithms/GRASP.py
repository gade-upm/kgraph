import math
import os.path
import random
import sys
import time

import graph_tool.all as gt
import numpy as np
import randomcolor
from matplotlib import pyplot as plt

from kgraph.algorithms import Algorithm
from kgraph.utils.graph import isolation_index, cohesion_index


# from joblib import Parallel, delayed

class GRASP(Algorithm):
    """
    Multiobjetive GRASP with Path Relinking
    Construction: Secuential combined
    Local Search: secuential combined

    Intensification Path Relinking
    """

    def __init__(self, graph, output, alpha):

        self._cohesion_factor = 0.8  # For kores_alg
        self.alpha = alpha  # Alpha for selecting best in Restricted Candidate List
        self.ES = []
        self.non_dominated = []
        self.non_dominated_id = []
        self.max_times = 100  # Times to have same seed
        self.percentage_candidates = 0.2
        self.type_greedy = 'percentaje'  # percentaje, both, cohesion, isolation

        super(GRASP, self).__init__('GRASP', graph, output)

    def run(self, draw_graph=None):

        # GRASP
        init = time.time()
        seeds = self._seeds()

        # TODO: PARALLELIZE
        '''
        for seed in seeds:
            candidates = self._candidate_list(seed)
            Parallel(n_jobs=2, require='sharedmem')(
                delayed(self._job_grasp)(list(seed), list(candidates)) for i in range(self.max_times))
        '''
        for seed in seeds:
            candidates = self._candidate_list(seed)
            # Stopping criterion Max iterations
            for i in range(self.max_times):
                '''
                if not candidates:
                    break
                '''
                sol = self._greedy_randomized_solution(list(seed), list(candidates))
                sol = self._local_search(sol)
                self._update_solution(sol)

        end = time.time()
        print('GRASP finished in {0} s.'.format(end - init))

        # Path Relinking
        init = time.time()
        self._path_relinking()
        end = time.time()
        print('Path Relinking finished in {0} s.'.format(end - init))

        # Remove data of ES
        self.ES = list(self.non_dominated)

        # Paint all solutions
        if draw_graph:
            solution_id = []
            for s in self.ES:
                i_a, i_c = self._index_values(s)
                solution_id.append([i_a, i_c])
            self._paint_solutions(solution_id)

        # Get pareto solutions
        init = time.time()
        self.non_dominated, self.non_dominated_id = self._simple_cull(self.non_dominated, self._dominates)
        end = time.time()
        print('Non dominated solutions finished in {0} s.'.format(end - init))
        '''
        index_values, communities = map(list, zip(*self.non_dominated))

        if draw_graph:
            self._paint_communities(communities)
            self._paint_non_dominated(index_values)
        '''
        if draw_graph:
            self._paint_communities(self.non_dominated)
            self._paint_non_dominated(self.non_dominated_id)

        return self.non_dominated

    # TODO: PARALLELIZE PART
    def _job_grasp(self, seed, candidates):
        if candidates:
            sol = self._greedy_randomized_solution(seed, candidates)
            sol = self._local_search(sol)
            self._update_solution(sol)

    def _candidate_list(self, seed):
        """
        CREATE Candidates, with the candidate elements that can be added to the sol
        :return: candidates
        """
        # Isolated vertices
        vertices = [self._network.vertex_index[v] for l in seed for v in l]
        all_vertices = [self._network.vertex_index[v] for v in self._network.vertices()]
        isolated = [self._network.vertex(v) for v in all_vertices if v not in vertices]
        # isolated = [self._network.vertex(v) for v in all_vertices if v in all_vertices and v not in vertices]

        return isolated

    def _greedy_randomized_solution(self, communities, candidates):
        """
        Greedy
        Probabilistic
        Adaptative

        :param communities: Communities of the graph
        :param candidates: Candidates to add
        :return:
        """
        # TODO: Make this in init to choose one type
        communities = self._max_isolation(communities, candidates)

        '''
        if self.type_greedy == 'percentaje':
            communities = self._max_percentaje(communities, candidates)
        elif self.type_greedy == 'both':
            communities = self._max_both(communities, candidates)
        elif self.type_greedy == 'cohesion':
            communities = self._max_cohesion(communities, candidates)
        elif self.type_greedy == 'isolation':
            communities = self._max_isolation(communities, candidates)
        '''

        return communities

    def _max_percentaje(self, communities, candidates):
        """
        Greedy
        Probabilistic
        Adaptative

        Based on a percenjate of sols
        :param communities:
        :param candidates:
        :return:
        """
        # Calculate max_iterarions based on a percentaje of the sols
        max_iterations = math.trunc(len(candidates) * self.percentage_candidates)
        for i in range(max_iterations):
            # Restricted for multi-objetive
            rcl, rcl_ids = self._construct_rcl(candidates, communities)  # Greedy
            if rcl:
                # Random selection
                random_number = random.randrange(len(rcl))
                candidate = rcl[random_number]
                idc = rcl_ids[random_number]
                # # Add sol
                communities[idc].append(candidate)
                # Remove candidate
                candidates.remove(candidate)  # Adaptative, we remove and then recalculate them
                if not candidates:
                    break
            else:
                break
        return communities

    def _max_both(self, communities, candidates):
        """
        Greedy
        Probabilistic
        Adaptative

        Based on maximize isolation and cohesion
        :param communities:
        :param candidates:
        :return:
        """
        old_ia, old_ic = self._index_values(communities)

        while True:
            # Restricted for multi-objetive
            rcl, rcl_ids = self._construct_rcl(candidates, communities)  # Greedy
            if rcl:

                # Random selection
                random_number = random.randrange(len(rcl))
                candidate = rcl[random_number]
                random_id = rcl_ids[random_number]

                # Steady improvement
                communities[random_id].append(candidate)
                new_ia, new_ic = self._index_values(communities)

                if not (old_ia < new_ia and old_ic < new_ic):
                    communities[random_id].remove(candidate)
                    break
                else:
                    candidates.remove(candidate)  # Adaptative, we remove and then recalculate them
                    old_ia = new_ia
                    old_ic = new_ic
                    if not candidates:
                        break
            else:
                break
        return communities

    def _max_cohesion(self, communities, candidates):

        """
        Greedy
        Probabilistic
        Adaptative

        Based on maximize cohesion
        :param communities:
        :param candidates:
        :return:
        """
        old_ic = self._index_cohesion(communities)

        while True:
            # Restricted for multi-objetive
            rcl, rcl_ids = self._construct_rcl(candidates, communities)  # Greedy
            if rcl:

                # Random selection
                random_number = random.randrange(len(rcl))
                candidate = rcl[random_number]
                idc = rcl_ids[random_number]

                # Mejora constante
                communities[idc].append(candidate)
                new_ic = self._index_cohesion(communities)

                if not (old_ic < new_ic):
                    communities[idc].remove(candidate)
                    break
                else:
                    candidates.remove(candidate)  # Adaptative, we remove and then recalculate them
                    old_ic = new_ic
                    if not candidates:
                        break
            else:
                break
        return communities

    def _max_isolation(self, communities, candidates):
        """
        Greedy
        Probabilistic
        Adaptative

        Based on maximize isolation
        :param communities:
        :param candidates:
        :return:
        """
        old_ia = self._index_isolation(communities)

        while True:
            # Restricted for multi-objetive
            rcl, rcl_ids = self._construct_rcl(candidates, communities)  # Greedy
            if rcl:

                # Random selection
                random_number = random.randrange(len(rcl))
                candidate = rcl[random_number]
                idc = rcl_ids[random_number]

                # Mejora constante
                communities[idc].append(candidate)
                new_ia = self._index_isolation(communities)

                if not (old_ia < new_ia):
                    communities[idc].remove(candidate)
                    break
                else:
                    candidates.remove(candidate)  # Adaptative, we remove and then recalculate them
                    if not candidates:
                        break
            else:
                break
        return communities

    def _local_search(self, seed):
        """
        Local Search algorithm
        :return: local optima sol
        """
        # Find sol that is better
        communities = self._improve_seeds(seed)

        return communities

    def _path_relinking(self):
        """
        Intensification, generate pairs and apply path relinking
        These sols generated are the combination off these pairs
        Return all non dominated sols of these combinations
        """
        print(len(self.ES))
        # TODO: PARALELIZE
        for idx, x in enumerate(self.ES):
            # In pairs
            for idy, y in enumerate(self.ES):
                # Path Relinking
                if idx != idy:
                    # If they have the same number of clusters
                    if len(x) == len(y):
                        self._apply_path_relinking(list(x), list(y))


    def _apply_path_relinking(self, x, y):
        """
        Generate all combinations of sols of x and y
        """
        dist_sol = []
        dist_sol_id = []
        dist = []

        # Generate all different alternatives
        for id_community, community in enumerate(x):
            # To add, not in x but is in y
            for node_y in y[id_community]:
                if node_y not in community:
                    dist.append([id_community, node_y])
            # Remove of x that is not in y
            for id_node_y, node_x in enumerate(community):
                if node_x not in y[id_community]:
                    x[id_community].pop(id_node_y)

        # Greeady, better in two objetives
        i_orig, c_orig = self._index_values(x)
        sol = list(x)

        if dist:
            while True:
                for d in dist:
                    new_sol = list(sol)
                    new_sol[d[0]].append(d[1])
                    isolation, cohesion = self._index_values(new_sol)
                    dist_sol_id.append([isolation, cohesion])
                    dist_sol.append(new_sol)

                new = max(list(dist_sol_id))
                if new[0] > i_orig and new[1] > c_orig:
                    i_orig, c_orig = new[0], new[1]
                    id_sol = dist_sol_id.index(new)
                    sol = dist_sol[id_sol]
                    dist_sol = []
                    dist_sol_id = []
                    dist.pop(id_sol)
                else:
                    # If not better
                    break
                # If empty
                if len(dist) == 0:
                    break

        self.non_dominated.append(sol)
        '''
        self.non_dominated_id.append([i_orig, c_orig])
        '''

        '''
        self._add_non_dominated(sol)
        '''

    def _simple_cull(self, dist_sol, dominates):
        """
        Return non dominated solutions(pareto solutions)
        
        :param dist_sol: distinct solutions
        :param dominates: function 
        :return: 
        """
        pareto_communities = [] # Pareto communities
        pareto_id = [] # Pareto points(isolation, cohesion)
        candidate_row_nr = 0
        input_points = []
        for s in dist_sol:
            i_a, i_c = self._index_values(s)
            input_points.append([i_a, i_c])

        while len(input_points):
            candidate_row = input_points[candidate_row_nr]
            candidate_row_sol = dist_sol[candidate_row_nr]
            input_points.remove(candidate_row)
            dist_sol.remove(candidate_row_sol)
            row_nr = 0
            non_dominated = True
            while len(input_points) != 0 and row_nr < len(input_points):
                row = input_points[row_nr]
                if dominates(candidate_row, row):
                    input_points.pop(row_nr)
                    dist_sol.pop(row_nr)
                elif dominates(row, candidate_row):
                    non_dominated = False
                    row_nr += 1
                else:
                    row_nr += 1

            if non_dominated:
                # add the non-dominated point to the Pareto frontier
                pareto_id.append(tuple(candidate_row))
                pareto_communities.append(candidate_row_sol)

        return pareto_communities, pareto_id

    @staticmethod
    def _dominates(row, another_row):
        return sum([row[x] >= another_row[x] for x in range(len(row))]) == len(row)  # maximization domination

    def _add_non_dominated(self, sol):
        """
        Add non dominated sol and remove dominated
        :param sol:
        :return:
        """
        isolation, cohesion = self._index_values(sol)
        non_dominated = True

        for non in self.non_dominated:
            if non[0][0] >= isolation and non[0][1] >= cohesion:
                non_dominated = False
                break

        if non_dominated:
            # Remove dominated elements
            for idx, non in enumerate(self.non_dominated):
                if non[0][0] < isolation and non[0][1] < cohesion:
                    self.non_dominated.pop(idx)
            self.non_dominated.append(([isolation, cohesion], sol))

    def _update_solution(self, sol):
        """
        Add communities
        :param sol
        """
        self.ES.append(sol)

    def _construct_rcl(self, candidates, communities):
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

    def _index_values(self, communities):
        """
        Compute means of Isolation and Cohesion
        :param communities:
        :return: isolation, cohesion
        """
        values = []
        for community in communities:
            ia = isolation_index(community)
            ic = cohesion_index(self._network, community)
            values.append((ia, ic))
        '''
        isolation, cohesion = zip(*values)
        i_a = np.mean(isolation)
        i_c = np.mean(cohesion)
        '''
        i_a = np.mean([v[0] for v in values])
        i_c = np.mean([v[1] for v in values])

        return i_a, i_c

    def _index_cohesion(self, communities):
        """
        Compute means of Cohesion
        :param communities:
        :return: cohesion
        """
        values = []
        for community in communities:
            ic = cohesion_index(self._network, community)
            values.append(ic)
        return np.mean(values)

    def _index_isolation(self, communities):
        """
        Compute means of Isolation
        :param communities:
        :return: isolation
        """
        values = []
        for community in communities:
            ia = isolation_index(community)
            values.append(ia)
        return np.mean(values)

    def _seeds(self):
        """
        Seeds generated from the components obtained by the cuts of all the possible k-cores of the graph
        :return: seeds
        """
        network = self._network.copy()
        k_cores = gt.kcore_decomposition(network)
        max_core = np.max(k_cores.a)
        min_core = 2
        seeds = []
        for n in range(min_core, max_core + 1):
            k_component = self._components(n, k_cores, network)
            if len(k_component) > 0:
                seeds.append(k_component)
        return seeds

    def _paint_communities(self, nondominated_communities):
        """
        Paint all Graphs in ES
        :param nondominated_communities:
        :return:

        """
        for idx, communities in enumerate(nondominated_communities):
            self.paint_graph(str(idx), self._network, communities)

    def _paint_seeds(self, seeds):
        """
        Paint different communities from seeds
        :param seeds:
        :return:
        """
        for i, communities in enumerate(seeds):
            for idx, community in enumerate(communities):
                self.paint(community, idx + i * 10)

    def _components(self, k, k_cores, network):
        network_cpy = network.copy()
        network_cpy.vp['GRASP'] = network_cpy.new_vertex_property('bool')
        for v in network_cpy.vertices():
            if k_cores[v] >= k:
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

    @staticmethod
    def _group_labels(labels, graph):
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
            '''
            neighbours = set(list(vertex.all_neighbors()))
            '''
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

    def _improve_seeds(self, seeds):
        communities = []
        isolated_vertices = self.__get_isolated_vertices(seeds)
        for seed in seeds:
            ic0 = cohesion_index(self._network, seed)
            self._expand_seed(seed, isolated_vertices, ic0)
            communities.append(seed)

        return communities

    def __get_isolated_vertices(self, seeds):
        vertices = [self._network.vertex_index[v] for l in seeds for v in l]
        all_vertices = [self._network.vertex_index[v] for v in self._network.vertices()]
        isolated = [self._network.vertex(v) for v in all_vertices if v not in vertices]
        # isolated = [self._network.vertex(v) for v in all_vertices if v in all_vertices and v not in vertices]
        return isolated

    def paint_graph(self, location, graph, communities):

        sys.stdout.write('Drawing graph ... ')
        sys.stdout.flush()
        network = gt.Graph(graph, directed=False)
        folder = os.path.abspath(self._output_dir)

        r_cols = randomcolor.RandomColor().generate(count=len(communities) + 1)
        colors = [list(int(r_col[1:][i:i + 2], 16) for i in (0, 2, 4)) for r_col in r_cols]

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
                      output=os.path.join(folder, str(len(communities)) + '_' + location + '_graph-communities.svg'))
        sys.stdout.write('Ok!\n')
        sys.stdout.flush()

    def _paint_non_dominated(self, non_dominated):
        """
        Paint non dominated(pareto) solutions
        :param non_dominated:
        :return:
        """
        x, y = zip(*non_dominated)

        plt.scatter(x, y)

        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }

        plt.title('Non Dominated', fontdict=font)
        plt.xlabel('Isolation', fontdict=font)
        plt.ylabel('Cohesion', fontdict=font)
        plt.savefig('./' + self._output_dir + '/Non_Dominated.png')
        plt.close()

    def _paint_solutions(self, solution_index):
        """
        Paint isolation and cohesion solutions
        :param non_dominated:
        :return:
        """
        x, y = zip(*solution_index)

        plt.scatter(x, y)

        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }

        plt.title('All solutions', fontdict=font)
        plt.xlabel('Isolation', fontdict=font)
        plt.ylabel('Cohesion', fontdict=font)
        plt.savefig('./' + self._output_dir + '/solutions.png')
        plt.close()
