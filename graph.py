import pandas as pd
import networkx as nx
import sumolib

from shapely.geometry import LineString

class SUMONetworkGraph(nx.DiGraph):
    def __init__(self, net=None, netAddress=None):
        """
        Initializes the SUMO network graph.

        Args:
            net (sumolib.net.Net): SUMO network object.
        """
        super().__init__()
        if net!=None:
            self.net = net
        elif netAddress!=None:
            self.net = sumolib.net.readNet(netAddress, withInternal=True)


    def net2edgedf(self):
        """
        Converts the SUMO network edges to a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing edge information.
        """
        def getShape(edge):
            _shape = list()        
            for point in edge.getShape():  
                #_shape.append(net.convertXY2LonLat(point[0], point[1]))
                _shape.append((point[0], point[1]))

            return LineString(_shape)

        def getRawShape(edge):
            _shape = list()        
            for point in edge.getRawShape():
                #_shape.append(net.convertXY2LonLat(point[0], point[1]))
                _shape.append((point[0], point[1]))
            return LineString(_shape)

        edgelist = []
        for edge in self.net.getEdges():
            box = edge.getBoundingBox()
            xx = (box[0]+box[2])/2
            yy = (box[1]+box[3])/2
            #_pt = net.convertXY2LonLat(xx,yy)
            _pt = (xx,yy)

            _id = edge.getID()
            if(edge.isSpecial()):
                _from = ""
                _to = ""
            else:
                _from = edge.getFromNode().getID()
                _to = edge.getToNode().getID()

            _type = edge.getType()
            _function = edge.getFunction()
            if(_function!="internal"):
                _function ="real"

            _laneNumber = edge.getLaneNumber()

            _length = edge.getLength()
            _speed = edge.getSpeed()
            _name = edge.getName()
            _shape = getShape(edge)
            _rawshape =getRawShape(edge)
            _bicycle = edge.allows("bicycle")
            _vehicle = edge.allows("passenger")
            _pedestrian = edge.allows("pedestrian")
            _bus = edge.allows("bus")
            edgelist.append({"id":_id, "from":_from, "to":_to, "laneNumber":_laneNumber,
                              "pedestrian_allow":_pedestrian,"vehicle_allow":_vehicle, "bicyle_allow":_bicycle,
                             "bus_allow":_bus,"speed":_speed,"function":_function, "shape":_shape,"rawshape":_rawshape,
                             "length":_length, "name":_name, "type":_type})

        edge_df = pd.DataFrame.from_dict(edgelist)
        edge_df["weight"] = edge_df.apply(lambda row: row.length/row.speed, axis=1)
        return edge_df


    def net2graph(self):
        """
        Converts the SUMO network to a graph representation.

        Returns:
            networkx.DiGraph: Directed graph representing the SUMO network.
        """
        EPSILON = 0.001  # Small constant for length of virtual edges

        def getInteralEdgeid(fromedgeid , toedgeid):
            fromedge = self.net.getEdge(fromedgeid)
            toedge = self.net.getEdge(toedgeid)
            conlist = fromedge.getOutgoing()[toedge]
            return self.net.getLane(conlist[0].getViaLaneID()).getEdge().getID()

        def outgoingInternalEdge(fromedgeid):
            outgoinglist = []
            fromedge = self.net.getEdge(fromedgeid)
            conlist = fromedge.getOutgoing()
            for edge in conlist:
                toedgeid = edge.getID()
                internaledge = getInteralEdgeid(fromedgeid , toedgeid)
                outgoinglist.append({"fromedge":fromedgeid, "toedge":toedgeid, "internaledge":internaledge})
            return outgoinglist

        edges = self.net2edgedf()
        edges = edges[edges["function"]=="real"]

        #edges = edges[edges["vehicle_allow"]==True]
        nodes = set(edges["from"]).union(set(edges["to"]))
        if "" in nodes:
            nodes.remove("")

        realedge=edges[["id"]].copy()
        realedge["from"] = realedge["id"].apply(lambda x: "from_"+x)
        realedge["to"] = realedge["id"].apply(lambda x: "to_"+x)
        realedge["type"] = "real"
        realedge["speed"] = realedge["id"].apply(lambda x: self.net.getEdge(x).getSpeed())
        realedge["length"] = realedge["id"].apply(lambda x: self.net.getEdge(x).getLength())

        internaledgelist = []
        templist = []
        for fromedgeid in list(edges["id"]):
            templist = outgoingInternalEdge(fromedgeid)
            for item in templist:
                internaledgelist.append(item)   
        internaledge_df = pd.DataFrame(internaledgelist)

        internaledge = internaledge_df.copy()
        internaledge.rename(columns={"fromedge":"from", "toedge":"to", "internaledge":"id"}, inplace=True)
        internaledge["from"] = internaledge["from"].apply(lambda x: "to_"+x)
        internaledge["to"] = internaledge["to"].apply(lambda x: "from_"+x)
        internaledge["type"] = "internal"
        internaledge["speed"] = internaledge["id"].apply(lambda x: self.net.getEdge(x).getSpeed())
        internaledge["length"] = internaledge["id"].apply(lambda x: self.net.getEdge(x).getLength())

        edge2nodelist = []
        for nodeid in nodes:
            node = self.net.getNode(nodeid)
            for edge in node.getIncoming():
                if not edge.isSpecial():
                    edge2nodelist.append({"fromedge":edge.getID(), "tonode":nodeid})
        edge2node_df = pd.DataFrame(edge2nodelist)

        incomingnode = edge2node_df.copy()
        incomingnode.rename(columns={"fromedge":"from", "tonode":"to"}, inplace=True)
        incomingnode["from"] = incomingnode["from"].apply(lambda x: "to_"+x)
        incomingnode["to"] = incomingnode["to"].apply(lambda x: "d_"+x)
        incomingnode["id"] = incomingnode.apply(lambda row: row["from"] +"_"+row["to"], axis=1)
        incomingnode["type"] = "destination_node"
        incomingnode["speed"] = 1
        incomingnode["length"] = EPSILON


        node2edgelist = []
        for nodeid in nodes:
            node = self.net.getNode(nodeid)
            for edge in node.getOutgoing():
                if not edge.isSpecial():
                    node2edgelist.append({ "fromnode":nodeid, "toedge":edge.getID()})
        node2edge_df = pd.DataFrame(node2edgelist)


        outgoingnode = node2edge_df.copy()
        outgoingnode.rename(columns={"fromnode":"from", "toedge":"to"}, inplace=True)
        outgoingnode["from"] = outgoingnode["from"].apply(lambda x: "o_"+x)
        outgoingnode["to"]   = outgoingnode["to"].apply(lambda x: "from_"+x)
        outgoingnode["id"]   = outgoingnode.apply(lambda row: row["from"] +"_"+row["to"], axis=1)

        outgoingnode["type"] = "origin_node"
        outgoingnode["speed"] = 1
        outgoingnode["length"] = EPSILON

        new_edge_data = pd.concat([realedge, internaledge,incomingnode,outgoingnode])
        new_edge_data["cost"] = new_edge_data["length"]/new_edge_data["speed"]

        G = nx.from_pandas_edgelist(new_edge_data, source="from", target="to", edge_attr=True, create_using=nx.DiGraph)        
        return G
    
def nodepath2edgepath(path, G):
    path_ = path.copy()
    edgelist = list()
    x = path_[0]
    path_.remove(x)
    for node in path_:
        edgelist.append(G[x][node]["id"])
        x = node
    return " ".join(edgelist)

def fixpath(path, G):
    id2type = {}
    for item in G.edges.data():
        id2type[item[2]["id"]] = item[2]["type"]
        
    path = path.split(" ")
    for item in path:
        if id2type[item]!="real":
            path.remove(item)
    path = " ".join(path)
    return path

def trips2paths(od_list, G):
    id2type = {}
    for item in G.edges.data():
        id2type[item[2]["id"]] = item[2]["type"]
    
    paths = list()
    for from_node,to_node in od_list:
        if from_node!=to_node:
            try:
                path = nx.shortest_path(G, source="o_"+from_node, target="d_"+to_node, weight="cost")
            except Exception as e:
                path = ""
            paths.append({"from_node":from_node, "to_node":to_node, "path":path})
    
    paths_df = pd.DataFrame(paths)
    paths_df = paths_df[paths_df["path"].apply(lambda x: len(x)>1)]
    paths_df["cost"] = paths_df["path"].apply(lambda x: nx.path_weight(G, x, weight="cost"))
    paths_df["length"] = paths_df["path"].apply(lambda x: nx.path_weight(G, x, weight="length"))
    paths_df["path"] = paths_df["path"].apply(lambda x: nodepath2edgepath(x,G))
    paths_df["path"] = paths_df["path"].apply(lambda x: fixpath(x, G))
    paths_df = paths_df.reset_index().rename(columns={"index":"route_id"})
    return paths_df




def sample():
    # Load SUMO network
    net = sumolib.net.readNet("grid_interval_4h/net.xml", withInternal=True)
    
    # Create SUMO network graph
    sumo_graph = SUMONetworkGraph(net)
    #or
    sumo_graph = SUMONetworkGraph(netAddress="grid_interval_4h/net.xml")


    # Convert SUMO network to graph representation
    df = sumo_graph.net2edgedf()

    # Convert SUMO network to graph representation
    graph = sumo_graph.net2graph()


