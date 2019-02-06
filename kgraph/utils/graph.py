
from graph_tool import all as gt
import os.path
import sys
import time
import randomcolor
import numpy as np

def load_graph(path, algorithms, format='graphml', component=False):
    sys.stdout.write('Loading network ...')
    sys.stdout.flush()
    t0 = time.time()
    g = gt.load_graph(path,fmt=format)
    if 'kores' in algorithms:
        gt.remove_parallel_edges(g)
    gt.remove_self_loops(g)
    if component:
        largest_component = gt.label_largest_component(g, directed=False)
        g.set_vertex_filter(largest_component)
        g.purge_vertices()        
    t = time.time()
    sys.stdout.write('Ok! ({0} s.)\n'.format(t-t0))
    
    return g
    
def community_structure_test(graph):
    sys.stdout.write('Getting community structure ...')
    sys.stdout.flush()
    
    t0 = time.time()
    state = gt.minimize_blockmodel_dl(graph)  
    Q = gt.modularity(graph, state.get_blocks())
    t = time.time()  
    sys.stdout.write('Ok! ({0} s.)\n'.format(t-t0))
    sys.stdout.flush()
    
    return Q

def isolation_index(seed):
        t_edges = set()
        intern_edges = set()
        for vertex in seed:
            in_edges = list(vertex.in_edges())
            intern_inedges = [edge for edge in in_edges if edge.source() in seed]
            out_edges = list(vertex.out_edges())
            intern_outedges = [edge for edge in out_edges if edge.target() in seed]
            partial_edges = set(in_edges+out_edges)
            partial_internedges = set(intern_inedges+intern_outedges)
            t_edges.update(partial_edges)
            intern_edges.update(partial_internedges)
        
        ia = float(len(intern_edges))/float(len(t_edges))
        
        return ia    
    
def cohesion_index(graph, seed):
    g = graph.copy()        
    
    filt = graph.new_vertex_property('bool')
    for vertex in seed:
        filt[vertex] = True
    g.set_vertex_filter(filt)
    g.purge_vertices()
    
    clust = gt.local_clustering(g)
    
    ic = np.mean(clust.a)
    
    return ic

def paint_graph(path, graph, communities):
    if path:
        sys.stdout.write('Drawing graph ... ')
        sys.stdout.flush()
        network =gt.Graph(graph, directed=False)
        folder = os.path.abspath(path)
        # colors = random.sample(range(100,151), len(communities))
        r_cols = randomcolor.RandomColor().generate(count=len(communities)+1)
        colors = [list(int(r_col[1:][i:i+2], 16) for i in (0, 2 ,4)) for r_col in r_cols]
        
        # color = graph.new_vertex_property('vector<float>')
        color = graph.new_vertex_property('int')
        base_color = colors.pop()
        for v in network.vertices():
            color[v] = (base_color[0]<<16) + (base_color[1]<<8) + base_color[2]
        for community in communities:
            c = colors.pop()
            for v in community:
                color[v] = (c[0]<<16) + (c[1]<<8) + c[2]
        pos = gt.sfdp_layout(network)
        gt.graph_draw(network, pos=pos, vertex_fill_color=color, output=os.path.join(folder, 'grafo.png'))
        sys.stdout.write('Ok!\n')
        sys.stdout.flush()
