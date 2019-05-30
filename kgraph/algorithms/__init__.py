import os.path

import graph_tool.all as gt


class Algorithm(object):
    def __init__(self, name, graph, output):
        self._network = graph
        self._name = name
        self._output_dir = output

    def run(self):
        raise NotImplementedError('Subclass must implement abstract method')

    def paint(self, community, community_id):
        """
        Paint Community
        :param community:
        :param community_id:
        :return:
        """
        network = gt.Graph(self._network, directed=False)
        folder = os.path.abspath(self._output_dir)

        v_filt = network.new_vertex_property('bool')
        for vertex in community:
            v_filt[vertex] = True

        gv = gt.GraphView(network, vfilt=v_filt)
        # pos = gt.arf_layout(gv)
        gt.graph_draw(gv, pos=gv.vp['pos'], output=os.path.join(folder, '{0}.png'.format(community_id)))

    @property
    def name(self):
        return self._name
