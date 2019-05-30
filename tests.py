from kgraph.management import hash_graph, compute_results
from kgraph.utils.graph import load_graph, paint_full_graph, paint_kcore
from kgraph.algorithms.GRASP import GRASP


def main():
    path = "data/netscience.graphml"

    g = hash_graph(load_graph(path, 'kores', component=True))
    # Pintar el grafo entero
    paint_full_graph('test', g, 'real')
    paint_kcore('test', g, 'real')

    alpha = [0.4, 0.5, 0.6, 0.7, 0.8]
    for a in alpha:

        print("-------------------------------")
        print("-------------------------------")
        print("-------------------------------")
        print("-------------------------------")
        print("-------------------------------")
        print("\t " + str(a))
        print("-------------------------------")
        print("-------------------------------")
        print("-------------------------------")
        print("-------------------------------")
        print("-------------------------------")
        print("-------------------------------")
        algorithm = GRASP(g, '_max_isolation/' + str(str(a-int(a))[2:]), a)
        solutions = algorithm.run(draw_graph=True)

        for communities in solutions:
            compute_results(g, communities)


if __name__ == "__main__":
    main()
