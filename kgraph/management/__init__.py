import argparse
import os
import sys
import uuid

import numpy as np

from kgraph.algorithms.GRASP import GRASP
from kgraph.algorithms.dengraph_alg import DengraphAlg
from kgraph.algorithms.kores_alg import KoresAlg
from kgraph.utils.graph import load_graph, community_structure_test, cohesion_index, isolation_index, paint_graph

algorithms = {'kores': KoresAlg,
              'dengraph': DengraphAlg,
              'GRASP': GRASP}


class ManagementUtility:
    def __init__(self, argv=None):
        self.argv = argv or sys.argv[:]
        self.prog_name = os.path.basename(self.argv[0])

    @staticmethod
    def execute():
        parser = argparse.ArgumentParser(description='Run and compare some community detection algorithms.')
        parser.add_argument('-a', '--algorithms', nargs='+', help='Algorithm(s) to take into account', required=True)
        parser.add_argument('-k', '--kore', type=int, help='k argument in k-core algorithm')
        parser.add_argument('-eps', '--epsilon', type=float,
                            help='the mu distance between clusters in dengraph algorithm')
        parser.add_argument('-mu', '--mu', type=float, help='the epsilon-neighborhood of a node in dengraph algorithm')
        parser.add_argument('-c', '--compare-to', nargs='+',
                            help='Algorithm(s) which main algorithm(s) is(are) compared')
        parser.add_argument('-comp', '--component', action='store_true', default=False)
        parser.add_argument('-i', '--input', help='GraphML graph file', required=True)
        parser.add_argument('-o', '--output', help='Output folder where communities will be saved')

        args = parser.parse_args()
        if 'kores' in args.algorithms and args.kore is None:
            print('K param is needed if you run kores algorithm')
            parser.print_usage()
            sys.exit(-1)
        elif 'dengraph' in args.algorithms and (args.epsilon is None or args.mu is None):
            print('epsilon param and mu param are needed if you run dengraph algorithm')
            parser.print_usage()
            sys.exit(-1)
        g = hash_graph(load_graph(args.input, args.algorithms, component=args.component))
        if args.component:
            print('INPUT: execute {0} algorithm(s) | Network: {1} vertices and {2} edges (largest component)'.format(
                ' '.join(args.algorithms), g.num_vertices(), g.num_edges()))
        else:
            print('INPUT: execute {0} algorithm(s) | Network: {1} vertices and {2} edges'.format(
                ' '.join(args.algorithms), g.num_vertices(), g.num_edges()))

        value = community_structure_test(g)
        print('Q = {0}'.format(value))

        for alg in args.algorithms:
            if alg in algorithms:
                if alg == 'kores':
                    algorithm = algorithms[alg](g, args.kore, args.output)
                elif alg == 'dengraph':
                    algorithm = algorithms[alg](g, args.epsilon, args.mu, args.output)
                elif alg == 'GRASP':
                    algorithm = algorithms[alg](g, args.output)
                print('algorithm {0} selected'.format(algorithm.name))
                communities = algorithm.run(draw_graph=args.output)
                if len(communities) > 0:
                    compute_results(g, communities)
                    if args.output is not None:
                        paint_graph(args.output, g, communities)
                else:
                    print('Algorithm run and no results reported')
            else:
                print('{0} is not a valid algorithm'.format(alg))


def compute_results(graph, communities):
    values = []
    print('{0} communities detected'.format(len(communities)))
    for idx, community in enumerate(communities):
        ia = isolation_index(community)
        ic = cohesion_index(graph, community)
        print('community {0} has Ia = {1} and Ic = {2}'.format(idx, ia, ic))
        values.append((len(community), ia, ic))
    length = np.mean([v[0] for v in values])
    Ia = np.mean([v[1] for v in values])
    Ic = np.mean([v[2] for v in values])
    print('Detected {0} communities with mean length = {1} and mean Ia = {2} and mean Ic = {3}'.format(
        int(len(communities)), length, Ia, Ic))


def execute_from_command_line(argv=None):
    utility = ManagementUtility(argv)
    utility.execute()


def hash_graph(graph):
    sys.stdout.write('Hashing graph ... ')
    sys.stdout.flush()

    graph.vp['hash'] = graph.new_vertex_property('string')

    for vertex in graph.vertices():
        hsh = str(uuid.uuid4()).replace('-', '')
        graph.vp['hash'][vertex] = hsh

    sys.stdout.write('Ok!\n')
    sys.stdout.flush()

    return graph
