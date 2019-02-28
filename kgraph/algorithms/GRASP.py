import random
import sys
import time

from kgraph.algorithms import Algorithm
from kgraph.algorithms.kores_alg import


class GRASP(Algorithm):
    """
    Multiobjetive GRASP with Path Relinking
    Construction: Secuential combined
    Local Search: secuential combined

    Intensification Path Relinking
    """
    def __init__(self, graph, output):
        super(GRASP, self).__init__('GRASP', graph, output)

    def run(self):
        init = time.time()
        seeds = self._seeds()
        while GRASP stopping criterion not satisfied:
            communities = self._greedy_randomized_solution(seeds)
            communities = self._localsearch(communities)
            bestCommunity = self._updateSolution(communities, bestCommunity)
        end = time.time()
        print('GRASP finished in {0} s.'.format(end - init))

        bestCommunity = self._pathRelinking()


        return bestCommunity


    def _greedy_randomized_solution(self):
        t0 = time.time()
        while is not complete:
            RCL = self._constructRCL() # Greedy
            s = random.choice(RCL)# Probabilistic
            add s to solution
            Reevaluate the incremental cost # Adaptative aspect

        t = time.time()
        sys.stdout.write(' Ok! ({0} s.)\n'.format(t - t0))
        return solution

    def _localSearch(self, seeds):
        '''
        Local Search algorithm
        :return: local optima solution
        '''

        while is not localltOptimal:
            # Find solution that is better
            solution = super._improve_seeds(seeds)

        return solution


    def _pathRelinking(self, seeds):
        '''
        Intensification
        :return:
        '''

    def _updateSolution(self, communities, bestCommunity):
        '''
        Return the best communities
        :param communities, communities
        :return: bestCommunity
        '''
        return bestCommunity


    def _constructRCL(self):
        '''
        Construct Restricted candidate list
        :return: RCL
        '''

        return RCL

    def _seeds(self):
        '''
        Generate Seeds for GRASP
        :return:
        '''
        return initialSeed
