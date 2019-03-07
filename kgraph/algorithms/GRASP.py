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
        ES = []
        # TODO: Stopping criterion
        while GRASP stopping criterion not satisfied:
            Candidates = self._createCandidateList()
            solution = self._greedy_randomized_solution(Candidates)
            solution = self._localSearch(solution)
            ES = self._updateSolution(solution, ES)
        end = time.time()
        print('GRASP finished in {0} s.'.format(end - init))

        bestCommunity = self._pathRelinking(ES)


        return bestCommunity

    # TODO: Define Candidates
    def _createCandidateList(self):
        """
        CREATE C, with the candidate elements that can be added to the solution
        :return:
        """
        Candidates = []
        # TODO: Generate Candidates with the graph
        return Candidates


    def _greedy_randomized_solution(self, Candidates):

        # TODO: Initialize solution

        # TODO: Define complete
        while is not complete:
            # TODO: Restricted for multi-objetive?
            RCL = self._constructRCL(Candidates) # Greedy
            s = random.choice(RCL)# Probabilistic

            # TODO: Add solution
            Candidates = self._updateCandidateList(Candidates) #Adaptative
        return solution


    # TODO: Define Candidates
    def _updateCandidateList(self, Candidates):
        """
        UPDATE C, with the candidate elements that can be added to the solution
        :return:
        """
        Candidates = []
        # TODO: Generate Candidates with the graph
        return Candidates

    # TODO: Local search global _improve_seeds
    def _localSearch(self, seeds):
        """
        Local Search algorithm
        :return: local optima solution
        """
        # TODO: Define localOptimal
        while is not localltOptimal:
            # Find solution that is better
            solution = super._improve_seeds(seeds)

        return solution


    # TODO: Path Relinking
    def _pathRelinking(self, seeds):
        """
        Intensification
        :return:
        """

    # TODO: Solution that is non dominated
    def _updateSolution(self, solution, ES):
        """
        Return the best communities
        :param communities, communities
        :return: bestCommunity
        """
        return ES

    # TODO: From candidates, return the ones with good g(c)
    def _constructRCL(self, Candidates):
        """
        Construct Restricted candidate list
        :return: RCL
        """
        RCL = []
        for c in Candidates:
            # TODO: define function g(c)
            if g(c) has a good value:
                RCL.append(c)

        return RCL
