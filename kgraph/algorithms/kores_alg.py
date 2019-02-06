from kgraph.algorithms import Algorithm
from kgraph.utils.graph import isolation_index, cohesion_index, paint_graph
import graph_tool.all as gt
import numpy as np
import sys, time

class KoresAlg(Algorithm):
    def __init__(self, graph, k, output):
        self._k = k
        self._cohesion_factor = 0.8
        super(KoresAlg,self).__init__('kores',graph, output)
        
    def run(self, draw_graph=None):
        init = time.time()
        seeds = self._seeds()
        communities = self._improve_seeds(seeds, draw_seeds=draw_graph)
#        if draw_graph:
#            sys.stdout.write('Drawing communities in graph ... ')
#            sys.stdout.flush()
#            t0 = time.time()
#            paint_graph(self._output_dir, self._network,communities)
#            t = time.time()
#            sys.stdout.write('Ok! ({0} s.)\n'.format(t-t0))
#            sys.stdout.flush()
        end = time.time()
        print('Algorithm finished in {0} s.'.format(end-init))
        
        return communities
        
    def _select_best_k(self, components):
        return self._k, components[self._k]
    
    def _seeds(self):
        t0 = time.time()
        network = self._network.copy()
        kcores = gt.kcore_decomposition(network)
        max_core = np.max(kcores.a)
        min_core = 2
        components = {}
        for n in range(min_core, max_core+1):
            k_component = self._components(n, kcores, network)
            if len(k_component) > 0:
                components[n] =  k_component
                sys.stdout.write('\rAnalyzing the {0}-support ... '.format(n))
                sys.stdout.flush()
        t = time.time()
        sys.stdout.write(' Ok! ({0} s.)\n'.format(t-t0))
        
        k, comps = self._select_best_k(components)        
        
        print('Selecting {0}-cut'.format(k))
        
        return comps
    
    def _map_vertexs(self, graph, seeds):
        components = [[gt.find_vertex(self._network, self._network.vp['hash'], graph.vp['hash'][v])[0] for v in seed] for seed in seeds]
        
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
    
    def _components(self, k, kcores, network):
        network_cpy = network.copy()
        network_cpy.vp['kcores'] = network_cpy.new_vertex_property('bool')
        for v in network_cpy.vertices():
             if kcores[v] >= k:
                network_cpy.vp['kcores'][v] = True
             else:
                network_cpy.vp['kcores'][v] = False
        network_cpy.set_vertex_filter(network_cpy.vp['kcores'])
        network_cpy.purge_vertices()
        
        labels, hist = gt.label_components(network_cpy, directed=False)
        components = self.__group_labels(labels, network_cpy)
        
        mapped = self._map_vertexs(network_cpy, components)
        
        return mapped
    
    def __get_isolated_vertices(self, seeds):
        vertices = [self._network.vertex_index[v] for l in seeds for v in l]
        all_vertices = [self._network.vertex_index[v] for v in self._network.vertices()]
        isolated = [self._network.vertex(v) for v in all_vertices if v in all_vertices and v not in vertices]
    
        return isolated
    
    def __expand(self, seed, isolated_vertices, ic0):
        newseed = list(seed)
        for vertex in newseed:
            in_neighbours = list(vertex.in_neighbors())
            out_neighbours = list(vertex.out_neighbors())
            neighbours = set(in_neighbours+out_neighbours)
            
            p_candidates = [candidate for candidate in list(neighbours.intersection(isolated_vertices)) if candidate not in seed]
            if len(p_candidates) > 0:
                for candidate in p_candidates:
                    isolation_before = isolation_index(seed)
                    new_seed =list(seed)+[candidate]
                    isolation_after = isolation_index(new_seed)
                    cohesion_after = cohesion_index(self._network,new_seed)
                    if isolation_after > isolation_before and cohesion_after >= self._cohesion_factor*ic0 and candidate not in seed:
                        seed[:] = seed+[candidate]
            
    def _expand_seed(self, seed, isolated_vertices, ic0):
        l_seed = len(seed)
        self.__expand(seed, isolated_vertices, ic0)
        l_newseed = len(seed)
        ia = isolation_index(seed)
        ic = cohesion_index(self._network, seed)
        print('Expanding community ... {0} vertexs -> {1} vertexs (Ia = {2} | Ic = {3})'.format(l_seed, l_newseed, ia, ic))
        if l_seed == l_newseed:
            sys.stdout.flush()
            return seed
        else:
            return self._expand_seed(seed, isolated_vertices, ic0)
    
    def _improve_seeds(self,seeds, draw_seeds=None):
        communities = []
        isolated_vertices = self.__get_isolated_vertices(seeds)
        print("{0} seeds detected to improve.".format(len(seeds)))
        for idx,seed in enumerate(seeds):
            print("=== Improving seed {0} ===".format(idx+1))
            ic0 = cohesion_index(self._network, seed)
            ia0 = isolation_index(seed)
            print('Ia0 = {0} | Ic0 = {1}'.format(ia0, ic0))
            t0 = time.time()
            self._expand_seed(seed, isolated_vertices, ic0)
            t = time.time()
            ic = cohesion_index(self._network, seed)
            ia = isolation_index(seed)
            print('Ia = {0} | Ic = {1}'.format(ia, ic))
            print('Time until converge: {0} s.'.format(t-t0))
            if draw_seeds:
                self.paint(seed, idx)
                print('community {0} saved.'.format(idx+1))
            communities.append(seed)
        
        return communities
        
        
        
        
        
