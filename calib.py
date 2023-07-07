import logging
import os
import sys
import sumolib
import traci
import pandas as pd
import numpy as np
from scipy.sparse import bsr_matrix
from scipy.optimize import lsq_linear
from typing import Dict, Optional, NoReturn, Union, Tuple
import random
from argparse import ArgumentParser
import json
from collections import Counter
import shutil
from shapely.geometry import LineString, Polygon
from collections import OrderedDict
import csv
import multiprocessing 
import glob
import networkx as nx
from graph import SUMONetworkGraph,trips2paths




def get_logger(log, name: str, log_dir):
    log.setLevel(logging.INFO)
    if (not os.path.isdir(log_dir)):
        os.mkdir(log_dir,mode=0o777) 
    file_handler = logging.FileHandler(os.path.join(log_dir,f'{name}.log'))
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    return log


log = logging.getLogger("calibration")
report = dict()

def get_options(args=None):
    argParser = sumolib.options.ArgumentParser()
    required = argParser.add_argument_group('required arguments')
    argParser.add_argument("-n", "--net-file", help="the SUMO net filename", required=True)
    argParser.add_argument("-m", "--measurment-file", help="the measurment filename contains sensor data",required=True) 
    argParser.add_argument("-dod", "--dod-file", help="the init distributed origin destination matrix filename",required=True)
    argParser.add_argument("-is", "--interval-size", help="the size of each interval in seconds", required=True)

    argParser.add_argument("-ni", "--number-iteration", help="number of iteration for each interval. default=10", default=10)
    argParser.add_argument("-r", "--route-file", help="the XML routes filename") 
    argParser.add_argument("-ib", "--interval-begin", help="the number of beginnig interval")
    argParser.add_argument("-ie", "--interval-end", help="the number of ending interval")
    argParser.add_argument("-l", "--output-location", help="the location of output files. default=output/", default="output")
    argParser.add_argument("-se", "--sensor-extra", help="the sensor data extra file")
    argParser.add_argument("-teta", "--teta", help="teta for logit. default=-0.001", default=-0.001)
    argParser.add_argument("-nsl", "--netstate-loading", help="load net-state from the previous interval, default=true",  default=True)
    argParser.add_argument("-scalenumber", "--scale-number", help="it is one over expectation sensor crossing for a random trip, default is automated calculation ")
    argParser.add_argument("-wc", "--weight-calibration", help="weight in optimization formula in calibration. default is automated calculation")
    argParser.add_argument("-bw", "--best-waiting", help="the iteration is finished when number of waiting vehicle is less than this number \
                                                           or the number of iteration is more than iteration number. default=0",default=0)
    argParser.add_argument("-mr", "--max-routes", help="maximum number of routes that saved in the database fo each trips, default=10",default=10)
    argParser.add_argument("-ttf", "--traveltime-factor", help="show how much the travel time can be far away the best time, default=20",default=20)
    argParser.add_argument("-sample-iteration", "--sample-iteration", help="number of sampling for getting travel time average for each edge, default=5",default=5)
    argParser.add_argument("-sumo-home", "--sumo-home", help="SUMO_HOME")
    argParser.add_argument("-sumo-binary", "--sumo-binary", help="sumo binary")
    argParser.add_argument("-edge-penalty", "--edge-penalty", help="add time (in second) to edge travel time")
    argParser.add_argument("-edgedata-nx", "--edgedata-nx", help="the edge data csv file for creating networkx")


   
    options = argParser.parse_args()

    if  options.net_file is None or options.measurment_file is None or \
        options.interval_size is None or  options.dod_file is None:
       
        argParser.print_help()
        sys.exit()

    return options

def clean_measurment(option):
    measurment = pd.read_csv(option.measurment_file)[["edge", "interval","count"]]
    m_dict = measurment.set_index(["edge","interval"])["count"].to_dict()

    sensor_edge = measurment.groupby("edge").first().reset_index().reset_index().drop(columns=["interval", "count"])
    sensor_edge = sensor_edge.rename(columns={"index":"id"})
    sensor_edge["edge"] = sensor_edge["edge"].astype(str)

    mylist = list()
    for index, row in sensor_edge.iterrows():
        for interval in range(int(option.interval_begin), int(option.interval_end)+1):
            if (row.edge, interval) in m_dict.keys():
                mylist.append({"edge":row.edge, "id":row.id, "interval":interval, "count":m_dict[(row.edge, interval)]})
            else:
                mylist.append({"edge":row.edge, "id":row.id, "interval":interval, "count":0})

    cleand_measurment = pd.DataFrame(mylist)
    cleand_measurment.to_csv(os.path.join(option.output_location,"real_data.csv"), index=False)
    return cleand_measurment

def update_option(option):

    if not(option.sumo_home is None):
        sumolib.os.environ["SUMO_HOME"]=option.sumo_home 
    if 'SUMO_HOME' in sumolib.os.environ:
        option.tools = sumolib.os.path.join(sumolib.os.environ['SUMO_HOME'], 'tools')
        sumolib.sys.path.append(option.tools)
    else:   
        sumolib.sys.exit("please declare environment variable 'SUMO_HOME'")
    if option.sumo_binary is None:
        option.sumo_binary = "sumo"
   
    if (not os.path.isdir(option.output_location)):
        os.mkdir(option.output_location,mode=0o777) 
   
    if option.sample_iteration is None:
        option.sample_iteration = "5"
    
    if option.edge_penalty is None:
        option.edge_penalty = "0"
        
    if option.traveltime_factor is None:
        option.traveltime_factor = "20"
    measurment = pd.read_csv(option.measurment_file)

    if option.max_routes is None:
        option.max_routes = "10"
    if option.best_waiting is None:
        option.best_waiting = "0"
    if option.netstate_loading is None:
        option.netstate_loading = "true"
    if option.teta is None:
        option.teta=str(-0.001)
    if option.interval_begin is None:
        option.interval_begin = str(measurment["interval"].min())

    if option.interval_end is None:
        option.interval_end = str(measurment["interval"].max())
        
    if option.number_iteration is None:
        option.number_iteration = str(20)
    if option.sensor_extra is None:
        sensor_extra = measurment.groupby(["edge"]).first().reset_index()
        sensor_extra["count"] = 0
        sensor_extra["interval"] = int(option.interval_begin)
        sensor_extra.to_csv(os.path.join(option.output_location,"sensor_extra.csv"), index=False)
        option.sensor_extra = os.path.join(option.output_location,"sensor_extra.csv")
    
    if option.edgedata_nx is None:
        sumo_graph = SUMONetworkGraph(netAddress=option.net_file)
        graph = sumo_graph.net2graph()
        nx.to_pandas_edgelist(graph).to_csv(os.path.join(option.output_location,"edge_data_nx.csv"),index=False)
        option.edgedata_nx = os.path.join(option.output_location,"edge_data_nx.csv")

    if option.route_file is None:
        edges = pd.read_csv(option.edgedata_nx)
        G = nx.from_pandas_edgelist(edges, source="source", target="target", edge_attr=True, create_using=nx.DiGraph)
        od_df = pd.read_csv(option.dod_file)[["from_node","to_node"]].drop_duplicates()
        od_list = list(od_df.to_records(index=False))
        t2p = trips2paths(od_list, G)
        option.route_file = os.path.join(option.output_location, "route_poi.csv")
        t2p.to_csv(option.route_file, index=False)
           
    clean_measurment(option)
    with open(os.path.join(option.output_location , 'option.txt'), 'w') as f:
        json.dump(option.__dict__, f, indent=2)  
        
    return option

def read_init_dod(option, interval):
    dod = pd.read_csv(option.dod_file)
    if "interval" in set(dod.columns):
        dod_temp = dod[dod["interval"]==interval]
        if len(dod_temp)!=0:
            return dod_temp
    return dod

def getInteralEdgeid(net, fromedgeid , toedgeid):
    fromedge = net.getEdge(fromedgeid)
    toedge = net.getEdge(toedgeid)
    conlist = fromedge.getOutgoing()[toedge]
    return net.getLane(conlist[0].getViaLaneID()).getEdge().getID()

def createRouteInternal(net, route):
    mylist = list()
    route_list = route.split(" ")
    for i in range(len(route_list)-1):
        mylist.append(route_list[i])
        mylist.append(getInteralEdgeid(net, route_list[i] , route_list[i+1]))
    mylist.append(route_list[-1])
    return " ".join(mylist)

def set_sensor_list_in_route(route_edges, measurment_edge2id):
    edges = set(route_edges.split(" "))
    sensoronedges = list(edges.intersection(set(measurment_edge2id.keys())))
    sensor_ids = [str(measurment_edge2id[i]) for i in sensoronedges]
    if len(sensor_ids)==0:
        return None
    else:
        return " ".join(sensor_ids)

def sensorTimeCalc(edges, measurment_edge2id, edge2time):
    edges = edges.split(" ")
    _time = 0
    time_list = list()
    for edge in edges:
        _time += edge2time[edge]
        if edge in measurment_edge2id.keys():
            time_list.append(str(_time))
    if(len(time_list)==0):
        return None
    else:
        return " ".join(time_list)



def update_sensorOnRoute(routes, interval_size):

    sensorOnRoute = list()
    for index, row in routes.iterrows():
        if str(row.sensors)!="nan" and not(row.sensors is None):
            _sensors = str(row.sensors).split(" ")
            times = str(row.sensors_time).split(" ")
            for i in range(len(_sensors)):
                    sensor = int(_sensors[i])
                    time = int(float(times[i]))
                    if time > interval_size :
                        prob = 0
                    else:
                        prob = (row.weight_route *(interval_size-time))/interval_size
                    sensorOnRoute.append({"route_id":row.route_id,  
                             "sensor":sensor,"trip_id":row.trip_id,
                            "prob":prob})
    return pd.DataFrame(sensorOnRoute)


def update_spp(sensorOnRoute_df, trips_len, sensors_len):

    index = pd.MultiIndex.from_product([range(trips_len),range(sensors_len)], names=["trip_id", "sensor"])
    spp_0 = pd.DataFrame(index=index)
    spp_0 = spp_0.reset_index()
    spp_df = sensorOnRoute_df.groupby(["trip_id", "sensor"])["prob"].apply(sum).reset_index()
    spp = spp_0.merge(spp_df, on=["trip_id","sensor"], how="left")
    spp["prob"] = spp["prob"].fillna(0)
    return spp

def update_init_od(option, dod, spp, sensor_count, od_waiting):
    if sensor_count < 0:
        sensor_count = 0 
    init_od = dod.copy()
    spp_1 = spp.groupby("trip_id")["prob"].apply(sum).reset_index()

    test = init_od.merge(spp_1, on="trip_id")
    if option.scale_number is None:
        scale = 1/((test["prob"]*test["weight_trip"]).sum())
        log.info("        scale_number = " + str(scale))
    else:
        scale = float(option.scale_number)/((test["prob"]*test["weight_trip"]).sum())
        log.info("        scale_number = " + str(scale))

    init_od["number_of_trips"] = init_od["weight_trip"].apply(lambda x: x*scale*sensor_count)
    if (len(init_od[init_od["number_of_trips"] >= 0]) < len(init_od)):
        log.info("error: negative or None value in init OD.")
        sys.exit("error: negative or None value in init OD.")
    init_od = init_od.drop(columns="weight_trip")

    log.info("        number of vehicles from init OD : " + str(int(init_od["number_of_trips"].sum())))
    if len(od_waiting)!=0:
        init_od = init_od.merge(od_waiting, on=["from_node","to_node"], how="left").fillna(0)
        init_od["number_of_trips"] = abs(init_od["number_of_trips_x"] - init_od["number_of_trips_y"])
        init_od = init_od.drop(columns={"number_of_trips_x","number_of_trips_y"})

    return init_od


def initializing(option):
    routes = pd.read_csv(option.route_file, low_memory=False)
    routes = routes.rename(columns={"path":"route_edges"})[["from_node", "to_node","route_edges"]]
    net = sumolib.net.readNet(option.net_file , withInternal=True)
    routes["route_edges_internal"] = routes["route_edges"].apply(lambda x: createRouteInternal(net, x))
    routelen = len(routes)
    dod = pd.read_csv(option.dod_file)
    dod = dod.groupby(["from_node", "to_node", "trip_id"]).first().reset_index()
    dodlen = len(dod)
    routes = routes.merge(dod, on=["from_node", "to_node"], how="left")
    routes = routes[routes["trip_id"].notnull()]
    routelen2 = len(routes)
    log.info("    remove " + str(routelen - routelen2) + " routes without trip weight")
    dod = routes.groupby(["from_node", "to_node"]).first().reset_index()[["trip_id", "from_node", "to_node", "weight_trip"]]
    dodlen2 = len(dod)
    log.info("    remove " + str(dodlen - dodlen2) + " trips without any route")

    real_data = pd.read_csv(os.path.join(option.output_location,"real_data.csv"))
    measurment_edge2id = real_data.set_index("edge")["id"].to_dict()
    trip2n_routes = routes.groupby(["from_node","to_node"]).apply(len)
    routes["weight_route"] = routes.apply(lambda row: 1/trip2n_routes[row["from_node"], row["to_node"]], axis=1)  
    routes["sensors"] = routes['route_edges_internal'].apply(lambda x: set_sensor_list_in_route(x, measurment_edge2id))  
    routes = routes.reset_index().rename(columns={"index":"route_id"})  
    routes.to_csv(os.path.join(option.output_location,"route_file.csv"), index=False)    
    trips_len = len(dod["trip_id"].unique())
    sensors_len = len(real_data["id"].unique())
     

def calibration(
                spp:pd.DataFrame,
                init_od:pd.DataFrame,
                real_data:pd.DataFrame,
                _sensor_data_extra,
                option,
                no_sensors, no_trips,
                interval=0, iteration=0)->Tuple[pd.DataFrame, pd.DataFrame]:
   
    temp_df = pd.DataFrame(index=list(range(no_sensors))).reset_index()
    temp_df = temp_df.rename(columns={"index":"id"})
    real_data_temp = temp_df.merge(real_data[real_data["interval"]==interval], on=["id"], how="left").fillna(0)

    sensor_data_extra = _sensor_data_extra.set_index("edge")["count"].to_dict()

    real_data_temp["count"] = real_data_temp.apply(lambda row: max(0, row["count"] - sensor_data_extra[row.edge]), axis=1)
    real_data_temp = real_data_temp.set_index("id")["count"]
    spp_df = spp.copy()
    A_x = np.zeros((no_sensors, no_trips))
    for sensor in range(no_sensors):
        A_x[sensor,:] = np.array(spp_df[spp_df["sensor"]==sensor].sort_values(["trip_id"]).reset_index(drop=True)["prob"])
    weight = float(option.weight_calibration) #np.sqrt(no_sensors/no_trips) * 0.1

    row_list = list()
    collumn_list = list()
    data_list = list()

    for i in range(A_x.shape[0]):
        for j in range(A_x.shape[1]):
            if A_x[i][j]!=0:
                row_list.append(i)
                collumn_list.append(j)
                data_list.append(A_x[i][j])

    for i in range(A_x.shape[1]):
        row_list.append(i+A_x.shape[0])
        collumn_list.append(i)
        data_list.append(weight)

    shapeA = (A_x.shape[0]+A_x.shape[1], A_x.shape[1])
    A= bsr_matrix((np.array(data_list), (np.array(row_list), np.array(collumn_list))), shape=shapeA)  
    lb = np.zeros(A_x.shape[1])
    ub = np.ones(A_x.shape[1])*np.inf 


    init_od_temp = init_od.sort_values("trip_id").reset_index(drop=True)
    b = np.concatenate((np.array(real_data_temp), weight*np.array(init_od_temp["number_of_trips"])), axis=0)
    log.info("        starting lsq_linear ...")

    
    res = lsq_linear(A, b, bounds=(lb, ub),  verbose=1, lsmr_tol='auto', max_iter=100)#max_iter=200, tol=1e-4)#tol=1e-35, max_iter=200)  #lsmr_tol='auto',
    log.info("        ended lsq_linear ...")
             
    np.savez_compressed(os.path.join(option.output_location , str(interval), str(iteration)+"_calib.npz"), A=A_x, x=res.x, b=b)
    x_df = pd.DataFrame(res.x).rename(columns={0:"number_of_trips"}).reset_index().rename(columns={"index":"trip_id"})
    sensor_error = (np.dot(A_x, res.x)-b[:A_x.shape[0]])
    mysensor_error_dict = dict()
    for i in range(no_sensors):
        mysensor_error_dict[(i,interval)] =sensor_error[i]     
    index_e = pd.MultiIndex.from_tuples(mysensor_error_dict.keys(), name=["index","interval"])
    e_df = pd.DataFrame(index = index_e, data=mysensor_error_dict.values()).rename(columns={0:"error"}).reset_index()
    vehicle_number = int(x_df["number_of_trips"].sum())

    return e_df, x_df, vehicle_number

###########################################################################################3

def route_sampling(od, _routes, option, interval,  iteration, sample_iteration):
    interval_size = int(option.interval_size)

    routes = _routes.copy()
    od2tripsNum = od.set_index(["trip_id"])["number_of_trips"].to_dict()
    id2path = routes.set_index("route_id")["route_edges"]
    all_routes = list()
    
    routes["number_of_trips_float"] = routes.apply(lambda row: row.weight_route * od2tripsNum[row.trip_id], axis=1)
    routes["number_of_trips_int"] = routes["number_of_trips_float"].apply(lambda x:
                                        #                                     int(np.round(x)))
                                    int(random.choices([np.ceil(x), np.floor(x)],
                                                    weights=(1-np.ceil(x)+x, np.ceil(x)-x), k=1)[0])) 
    routes = routes[routes["number_of_trips_int"] > 0]
    for index, row in routes.iterrows():
        depart_list = random.choices(list(range(interval*interval_size, (interval+1)*interval_size)), k=row.number_of_trips_int)
        for depart in depart_list:
            all_routes.append({"depart":depart, "route_id":row.route_id, "trip_id":row.trip_id})

            
    
    
    all_routes_df = pd.DataFrame.from_dict(all_routes).sort_values("depart").reset_index(drop=True).reset_index()
    all_routes_df["departLane"]="free"
    all_routes_df["departSpeed"]="max"

    all_routes_df["edges"]=all_routes_df["route_id"].apply(lambda x:id2path[x])
    all_routes_df.rename(columns={"index":"id"}, inplace=True)
    all_routes_df["id"] = all_routes_df["id"].apply(lambda x: str(x) + "_" + str(interval))
    vehid2pathid = all_routes_df.set_index("id")["route_id"].to_dict()
    def convert_row(row):
        return """
        <vehicle id="%s" depart="%s" departLane="%s" departSpeed="%s" departPos="0" arrivalPos="max">
            <route edges="%s"/>
        </vehicle>""" % (row.id, row.depart, row.departLane, row.departSpeed, row.edges)

    text0 = """<?xml version="1.0" encoding="UTF-8"?>\n\n\n"""
    text1 = """<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">"""
    text2 = '\n'.join(all_routes_df.apply(convert_row, axis=1))
    text3 = """\n</routes>"""
    with open(os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(sample_iteration)+"_routes_sim.xml"), 'w') as myfile: 
        myfile.write(text0+text1+text2+text3)
    return vehid2pathid, all_routes_df,len(all_routes_df)



###########################################################################################3



def create_edge_data_add(option, interval, iteration, sample_iteration):
        
    text0 = """<?xml version="1.0" encoding="UTF-8"?>\n\n\n"""
    text1 = """<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
"""
    text2 = """
        <edgeData id="%s" freq="%s" file="%s" excludeEmpty="%s" withInternal="%s"/>""" % ("dump_", option.interval_size, str(iteration)+"_"+str(sample_iteration)+"_aggregated.xml", "false","true")

    text3 = """\n</additional>"""
    with open(os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(sample_iteration)+"_edge_data_add.xml"), 'w') as myfile: 
        myfile.write(text0+text1+text2+text3)



###########################################################################################3

def sumoCommand(option, interval, iteration, sample_iteration):
    diroutput = os.path.join(option.output_location,str(interval))
    if (not os.path.isdir(diroutput)):
        os.mkdir(diroutput,mode=0o777) 
    sc_dict = dict()
    sc_dict["--net-file"] = option.net_file
    sc_dict["--route-files"] = os.path.join(diroutput, str(iteration)+"_"+str(sample_iteration)+"_routes_sim.xml")
    sc_dict["--additional-files"] = os.path.join(diroutput , str(iteration) +"_"+str(sample_iteration)+ "_edge_data_add.xml")
    sc_dict["--tripinfo-output"] = os.path.join(diroutput , str(iteration)+"_"+str(sample_iteration)+"_trip_info.xml")
    sc_dict["--statistic-output"] = os.path.join(diroutput , str(iteration)+"_"+str(sample_iteration)+"_statistic_output.xml")
    sc_dict["--vehroute-output"] =  os.path.join(diroutput , str(iteration)+"_"+str(sample_iteration)+"_vehroute_output.xml")
    sc_dict["--tripinfo-output.write-undeparted"] = "true"
    sc_dict["--vehroute-output.write-unfinished"] = "true"

    sc_dict["--vehroute-output.exit-times"] = "true"
    sc_dict["--no-warnings"] = "true"
    sc_dict["--no-step-log"] = "true"
    sc_dict["--begin"] = str(interval * int(option.interval_size))

    sumoCmd = list()
    sumoCmd.append(option.sumo_binary)
    for key in sc_dict.keys():
        sumoCmd.append(key)
        sumoCmd.append(sc_dict[key])
    return sumoCmd



def simulation(option, interval, iteration, dod, last_best_iteration, last_best_sample_iteration, net, sample_iteration):
    
    sumoCmd = sumoCommand(option, interval, iteration, sample_iteration)
    traci.start(sumoCmd)
    savedstate = os.path.join(option.output_location,str(interval-1),str(last_best_iteration),str(last_best_sample_iteration)+"_savedstate.xml")
    if os.path.isfile(savedstate) and (eval(option.netstate_loading.title())):
        traci.simulation.loadState(savedstate)
    step = 0
    while step < int(option.interval_size):
        traci.simulationStep()
        step += 1
    state = traci.simulation.saveState(os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(sample_iteration)+"_savedstate.xml"))

    traci.close()


###########################################################################################3

def fun_edge2time(net, option, interval, iteration, sample_iteration):
    sumolib.subprocess.call(["python3", os.path.join(option.tools,"xml","xml2csv.py"),
                              os.path.join(option.output_location, str(interval),str(iteration)+"_"+str(sample_iteration)+"_aggregated.xml"),
                            "--xsd",os.path.join(option.sumo_home,"data","xsd","meandata_file.xsd")])
    
    aggregated = pd.read_csv(os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(sample_iteration)+"_aggregated.csv"), sep=";")
    alledges = set(aggregated["edge_id"])
    aggregated = aggregated[aggregated["edge_traveltime"].notna()]

    _edge2time = aggregated.set_index("edge_id")["edge_traveltime"].to_dict()

    for key in alledges:
        if not(key in _edge2time.keys()):
            edge = net.getEdge(key)
            _edge2time[key] = edge.getLength()/edge.getSpeed()
    for key in alledges:
        edge = net.getEdge(key)
        _edge2time[key] = min(_edge2time[key]+int(option.edge_penalty), (edge.getLength()*int(option.traveltime_factor))/edge.getSpeed())
    return _edge2time

###########################################################################################3
def shortest_path_OD(option, edge2time, interval, iteration, best_sample):
    
    def nodepath2edgepath(path):
        path_ = path.copy()
        edgelist = list()
        x = path_[0]
        path_.remove(x)
        for node in path_:
            edgelist.append(G[x][node]["id"])
            x = node
        return " ".join(edgelist)

    def fixpath(path, id2type):
        path = path.split(" ")
        for item in path:
            if id2type[item]!="real":
                path.remove(item)
        path = " ".join(path)  
        return path

    def dist(a,b):
        if a.startswith("from_"):
            firstnode = net.getEdge(a.split("from_")[1]).getFromNode()
        elif a.startswith("to_"):
            firstnode = net.getEdge(a.split("to_")[1]).getToNode()
        elif a.startswith("o_"):
            firstnode = net.getNode(a.split("o_")[1])
        elif a.startswith("d_"):
            firstnode = net.getNode(a.split("d_")[1])  

        if b.startswith("from_"):
            secondnode = net.getEdge(b.split("from_")[1]).getFromNode()
        elif b.startswith("to_"):
            secondnode = net.getEdge(b.split("to_")[1]).getToNode()
        elif b.startswith("o_"):
            secondnode = net.getNode(b.split("o_")[1])
        elif b.startswith("d_"):
            secondnode = net.getNode(b.split("d_")[1]) 
        (x1,y1) = firstnode.getCoord()
        (x2,y2) = secondnode.getCoord()
        return (((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)/30
        
    def shortest_path(G, source, target, trip_id, weight_trip, return_dict):
        path = nx.shortest_path(G, source="o_"+source, target="d_"+target, weight="cost")
        path = nodepath2edgepath(path)
        path = fixpath(path, id2type)
        return_dict[(source, target, trip_id, weight_trip)] = path
  
    net = sumolib.net.readNet(option.net_file , withInternal=True)
    edges_nx = pd.read_csv(option.edgedata_nx)
    edges_nx["cost"] =  edges_nx.apply(lambda row: edge2time[row["id"]] if row["type"]  in {"real", "internal"} else 0,axis=1) 
    G = nx.from_pandas_edgelist(edges_nx, source="source", target="target", edge_attr=True, create_using=nx.DiGraph)
    id2type = edges_nx.set_index("id")["type"] 
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    sumolib.subprocess.call(["python3", os.path.join(option.tools ,"xml","xml2csv.py"),
                              os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(best_sample)+"_routes_sim.xml")]
                              )
    routes_w = pd.read_csv(os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(best_sample)+
                        "_routes_sim.csv"), sep=";")[["route_edges"]]
    routes_w["from_edge"] = routes_w["route_edges"].apply(lambda x: x.split(" ")[0])
    routes_w["to_edge"] = routes_w["route_edges"].apply(lambda x: x.split(" ")[-1])
    routes_w["from_node"] = routes_w["from_edge"].apply(lambda x: net.getEdge(x).getFromNode().getID())
    routes_w["to_node"] = routes_w["to_edge"].apply(lambda x: net.getEdge(x).getToNode().getID())
    routes_w = routes_w.groupby(["from_node","to_node"]).first().reset_index()[["from_node", "to_node"]]
    od = pd.read_csv(option.dod_file)
    route_temp = routes_w.merge(od, on=["from_node", "to_node"], how="left")
    for index, row in route_temp.iterrows():
            p = multiprocessing.Process(target = shortest_path, args=(G, row.from_node, row.to_node, row.trip_id, 
                                                                         row.weight_trip ,return_dict))
            p.start()
            processes.append(p)

    for p in processes:
            p.join()   

    routeList = []
    for key in return_dict.keys():
        routeList.append({"from_node":key[0], "to_node":key[1],  "trip_id":key[2], 
                           "weight_trip":key[3], "route_edges":return_dict[key]})
    return pd.DataFrame(routeList)

def route_update(_routes, option, edge2id, edge2time, interval=-1, iteration=-1, best_sample=-1):

    teta = float(option.teta)
   
    net = sumolib.net.readNet(option.net_file , withInternal=True)

    _interval = interval
    #######################################################
    #_interval = -1    # for statics route table !!!!!!!!!!
    #######################################################
    if _interval!=-1:
        log.info("        starting shortest path calculation . . . ")
        new_routes = shortest_path_OD(option, edge2time, interval, iteration, best_sample)
        log.info("        shortest path calculation is ended!")

        #new_routes.to_csv("test_shortest_path.csv", index=False)
        new_routes["route_edges_internal"] = new_routes["route_edges"].apply(lambda x: createRouteInternal(net, x))


        new_routes["sensors"] = new_routes['route_edges'].apply(lambda x: set_sensor_list_in_route(x, edge2id))
        _routes = _routes[list(new_routes.columns)]
        r1 = pd.concat([_routes,new_routes])
        _routes = r1.groupby("route_edges").first().reset_index()
        _routes = _routes.reset_index().rename(columns={"index":"route_id"})
    #############################################

    _routes["travel_time"] = _routes["route_edges_internal"].apply(lambda x: sum([edge2time[i] for i in x.split(" ")]))
    route_collumns = ["route_id","route_edges","from_node","to_node","weight_trip","route_edges_internal","sensors","trip_id","travel_time"]
    df1 = _routes[route_collumns]
    df1 = df1.set_index(["route_id","route_edges","from_node","to_node","weight_trip","route_edges_internal","sensors"])


    _routes = df1.groupby('trip_id')['travel_time'].nsmallest(int(option.max_routes)).reset_index() 
    _routes["sensors_time"] = _routes["route_edges_internal"].apply(lambda x: sensorTimeCalc(x, edge2id, edge2time))
    
    ### routes weight updating ###

    _routes["travel_time_exp"] = _routes["travel_time"].apply(lambda x: np.exp(x*teta))
    trip_id2weight = _routes.groupby("trip_id")["travel_time_exp"].apply(sum)   


    _routes["weight_route"] = _routes.apply(lambda row: row.travel_time_exp/trip_id2weight[row.trip_id], axis=1)
    _routes = _routes.drop(columns={"travel_time_exp"})
    

    _routes = _routes.drop(columns=["route_id"])
    _routes = _routes.reset_index(drop=True).reset_index().rename(columns={"index":"route_id"})


    
    _routes.to_csv(os.path.join(option.output_location , "last_routes.csv"), index = False)
    return _routes

###########################################################################################3


def update_sensor_extra(all_routes, sensor_set, option, interval, iteration, sample_iteration):
    
    sumolib.subprocess.call(["python3", os.path.join(option.tools,"xml","xml2csv.py"),
                              os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(sample_iteration)+"_vehroute_output.xml")])
    
    sumolib.subprocess.call(["python3", os.path.join(option.tools,"xml","xml2csv.py"),
                              os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(sample_iteration)+"_trip_info.xml")])

    vehroute = pd.read_csv(os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(sample_iteration)+"_vehroute_output.csv"),
                            sep=";")
    vehroute["vehicle_arrival"] = vehroute["vehicle_arrival"].fillna(-1)
    tripinfo = pd.read_csv(os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(sample_iteration)+"_trip_info.csv"),
                            sep=";")
    undepart = set(tripinfo[tripinfo["tripinfo_depart"]==-1]["tripinfo_id"])
    all_routes_undepart = all_routes[all_routes["id"].apply(lambda x: x in undepart)]
    
    sensor_data_extra_list = list()
    for index,row in all_routes_undepart.iterrows():
        edgelist = row.edges.split(" ")
        for i in range(len(edgelist)):
            if edgelist[i] in sensor_set:
                    sensor_data_extra_list.append(edgelist[i])

    vehroute["vehicle_arrival"] = vehroute["vehicle_arrival"].fillna(-1)
    vehroute_remined = vehroute[vehroute["vehicle_arrival"]==-1]

    for index,row in vehroute_remined.iterrows():
        edgelist = row.route_edges.split(" ")
        timelist = [int(i.split(".")[0]) for i in row.route_exitTimes.split(" ")]
        for i in range(len(edgelist)):
            if edgelist[i] in sensor_set:
                if (timelist[i]== -1):
                    sensor_data_extra_list.append(edgelist[i])
    sensor_data_extra = dict(Counter(sensor_data_extra_list)) 
    se = list()
    for sensor in sensor_set:
        if sensor not in sensor_data_extra.keys():
            se.append({"edge":sensor , "interval":interval+1 , "count":0})
        else:
            se.append({"edge":sensor , "interval":interval+1 , "count":sensor_data_extra[sensor]})
    return pd.DataFrame(se)

###########################################################################################3


# importing element tree
import xml.etree.ElementTree as ET 

def logging_statistics(option, interval, iteration, sample_iteration):
    sumolib.subprocess.call(["python3", os.path.join(option.tools,"xml","xml2csv.py"),
                              os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(sample_iteration)+"_trip_info.xml")])

    trip_info = pd.read_csv(os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(sample_iteration)+"_trip_info.csv"),
                             sep=";")
    loaded = len(trip_info)
    inserted = len(trip_info[trip_info["tripinfo_depart"]!=-1])
    running = inserted - len(trip_info[trip_info["tripinfo_arrival"]!=-1])
    waiting = loaded - inserted
    vehicles_now = dict()
    vehicles_now["loaded"] = loaded
    vehicles_now["inserted"] = inserted
    vehicles_now["running"] = running
    vehicles_now["waiting"] = waiting
    log.info("        vehicles_now :      " + str(vehicles_now))
   

    tree = ET.parse(os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(sample_iteration)+"_statistic_output.xml")) 

    root = tree.getroot() 
    for key in root[1].attrib.keys():
        root[1].attrib[key] = int(float(root[1].attrib[key]))

    for key in root[2].attrib.keys():
        root[2].attrib[key] = int(float(root[2].attrib[key]))

    for key in root[6].attrib.keys():
        root[6].attrib[key] = float(root[6].attrib[key])

    vehicleTripStatistics = dict()
    vehicleTripStatistics["speed"] = root[6].attrib["speed"] * 3.6
    vehicleTripStatistics["teleport"] = root[2].attrib["total"]

    log.info("        " + root[1].tag + "_till_now:  " + str(root[1].attrib))
    log.info("        " + root[2].tag + "_till_now: " + str(root[2].attrib))
    log.info("        " + root[6].tag + "_till_now:   " + str(vehicleTripStatistics))
    return running , waiting , vehicleTripStatistics["speed"],vehicleTripStatistics["teleport"] 

###########################################################################################3

def getShape(net, edge, geo=True):
   
    _shape = list()

    for point in edge.getShape():  
        if geo :
            _shape.append(net.convertXY2LonLat(point[0], point[1]))
        else:
            _shape.append((point[0], point[1]))

    
    return LineString(_shape)


def viz_edge_count(routes, od, net, option, interval, iteration, sample_iteration, geo=True):
    net = sumolib.net.readNet(option.net_file)
    aggregated = pd.read_csv(os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(sample_iteration)+"_aggregated.csv"), 
                             sep=";")

    od2tripsNum = od.set_index(["trip_id"])["number_of_trips"].to_dict()
    te = dict()
    for index, row in routes.iterrows():
        edges = row.route_edges.split(" ")
        w = row.weight_route * od2tripsNum[row.trip_id]
        trip = row.trip_id
        for edge in edges:
            if edge not in te.keys():
                te[edge] = w
            else:
                te[edge] += w
   
    aggregated["sim_count"] = ((aggregated["edge_arrived"] + aggregated["edge_departed"] +
                                aggregated["edge_entered"] + aggregated["edge_left"])/2).apply(int)
    id2count_sim = aggregated.set_index("edge_id")["sim_count"].to_dict() 
    
    mylist = list()
    for edge in net.getEdges():
        if edge.getID() not in te.keys():
            te[edge.getID()] = 0
        mylist.append({"id":edge.getID(), "from":edge.getFromNode().getID(), "to":edge.getToNode().getID(),
                      "number_lane":edge.getLaneNumber(), "shape":getShape(net, edge, geo),
                       "expected_count": te[edge.getID()], "sim_count":id2count_sim[edge.getID()]})

    edges_count_data = pd.DataFrame(mylist)
    edges_count_data.to_csv(os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(sample_iteration)+"_edge_count_viz.csv"), index=False)

###########################################################################################

def fixedpoint_error(edge2time_old, edge2time_new):
 
    list1 = np.array(list(OrderedDict(sorted(edge2time_old.items())).values()))
    list2 = np.array(list(OrderedDict(sorted(edge2time_new.items())).values()))

    return error(list1, list2)

def update_edge2time(option, init_edge2time, newedge2time, edge2time_list, method="del2"):  #Steffensen's method del2  or  Secant method
    #https://shannon-lab.github.io/zapdos/syntax/Executioner/FixedPointAlgorithms/index.html

    error = 0
    for key in edge2time_list[-1].keys():
        error += np.square(edge2time_list[-1][key] - edge2time_list[-2][key])
    error = np.sqrt(error)/len(edge2time_list[-1])
    log.info("        fixed point error : " + str(error))
    if len(edge2time_list)<3:
        return newedge2time, edge2time_list,error
    else:
        p0 = edge2time_list[0]
        p1 = edge2time_list[1]
        p2 = edge2time_list[2]
        p = dict()
        if method=="del2":
            for key in p0.keys():
                d = p2[key] - 2*p1[key] + p0[key]
                if d<=0:
                    d=p2[key]
                p[key] = min(max(init_edge2time[key], p0[key] - np.square(p1[key] - p0[key]) / d), int(option.traveltime_factor)*init_edge2time[key])
            return p,[p],error
        else:
            for key in p0.keys():
                d = p2[key] - 2*p1[key] + p0[key]
                if d<=0:
                    d=p2[key]
                p[key] = min(max(init_edge2time[key], p0[key] - np.square(p1[key] - p0[key]) / d), int(option.traveltime_factor)*init_edge2time[key])
                edge2time_list = list()
                edge2time_list.append(p2)
                edge2time_list.append(p)
            return p,edge2time_list,error

###########################################################################################3
def norm(A):
    return np.sqrt(np.sum(np.square(A)))

def error(A,B, type_error=0):
    if type_error==0:
        return norm(A-B)/norm(A)
    elif type_error==1:
        return norm(A-B)/len(A)
    else:
        return np.NaN

def fun_sensor_error(option, interval, iteration, sample_iteration):
    
    def vehroute2sensor(vehroute, sensor_set, interval, interval_size):
        vehroute["vehicle_arrival"] = vehroute["vehicle_arrival"].fillna(-1)
        vehroute_remined = vehroute[vehroute["vehicle_depart"]!=-1]

        sensor_count_list = list()

        for index,row in vehroute_remined.iterrows():
            edgelist = row.route_edges.split(" ")
            timelist = [int(i.split(".")[0]) for i in row.route_exitTimes.split(" ")]
            for i in range(len(edgelist)):
                if edgelist[i] in sensor_set:
                    if (timelist[i]!= -1) and (timelist[i] >= (interval*interval_size) ):
                        sensor_count_list.append(edgelist[i])
        sensor_count = dict(Counter(sensor_count_list))  
        for s in sensor_set:
            if s not in sensor_count.keys():
                sensor_count[s] = 0
        return sensor_count
    
    output_dir = option.output_location
    interval_size=int(option.interval_size)
    sumolib.subprocess.call(["python3", os.path.join(option.tools,"xml","xml2csv.py"),
                              os.path.join(output_dir,str(interval),str(iteration)+"_"+str(sample_iteration)+"_vehroute_output.xml")])

    vehroute = pd.read_csv(os.path.join(output_dir,str(interval),str(iteration)+"_"+str(sample_iteration)+"_vehroute_output.csv"), sep=";")
    sensor = pd.read_csv(output_dir + "real_data.csv")

    compare = sensor[sensor["interval"]==interval].drop(columns=["interval"])
    no_sensor = len(compare)

    sensor_set = set(sensor["edge"])
    veh_sensor = vehroute2sensor(vehroute, sensor_set, interval, interval_size)
    compare["sim_count"] = compare["edge"].apply(lambda x: veh_sensor[x])
    
    data = np.load(os.path.join(output_dir,str(interval),str(iteration)+"_calib.npz"))
    b = data["b"]
    A = data["A"]
    x = data["x"]
    compare["calib"] = list(np.dot(A,x)) + compare["count"] - b[0:no_sensor]
    
    return error(compare["count"], compare["sim_count"], 0),error(compare["count"], compare["calib"], 0),error(compare["calib"], compare["sim_count"],0)

###########################################################################################

def parallel_sampling(report, od_estimation, routes, option, interval, iteration, sample_iteration, dod,
                      last_best_iteration, last_best_sample_iteration, net, oldedge2time,return_dict):

    report["sample_number"] = sample_iteration
    vehid2pathid, all_routes, report["vehicle_number_sampling"] = route_sampling(od_estimation, routes, option, interval, iteration, sample_iteration)  
    create_edge_data_add(option, interval, iteration, sample_iteration)
    simulation(option, interval, iteration, dod, last_best_iteration, last_best_sample_iteration, net, sample_iteration)
    newedge2time = fun_edge2time(net, option, interval, iteration, sample_iteration)
    error_sensor, report["sensor_error_calibration"], report["sensor_error_sim2calib"] = \
                                         fun_sensor_error(option, interval, iteration, sample_iteration)
    report["sensor_error_simulation"] = error_sensor
    viz_edge_count(routes, od_estimation, net, option, interval, iteration, sample_iteration, geo=False)
    running, waiting, average_speed, teleport = logging_statistics(option, interval, iteration, sample_iteration)
    report["running"] = running
    report["waiting"] = waiting
    report["speed"] = average_speed
    report["fixed_point_error"] = fixedpoint_error(oldedge2time, newedge2time)
    report["teleport"] = teleport
    report["newedge2time"] = newedge2time
    report["all_routes"] = all_routes
    return_dict[sample_iteration] = report
###########################################################################################
def clean_samples(option, interval, iteration, sample):
    allfiles = []
    path = os.path.join(option.output_location,str(interval))
    for i in range(int(option.sample_iteration)):
        if i!=sample:
            required_files = glob.glob(os.path.join(path,str(iteration)+"_"+str(i)+"_*")) # This gives all the files that matches the pattern
            for x in required_files:    
                allfiles.append(x)
    results = [os.remove(x) for x in allfiles]


###########################################################################################

def clean_iteration(option, interval, iteration):
    path = os.path.join(option.output_location,str(interval))
    required_files = glob.glob(os.path.join(path,str(iteration)+"_*")) # This gives all the files that matches the pattern
    results = [os.remove(x) for x in required_files]


###########################################################################################
def calculate_OD_waiting(option, interval, iteration, sample_iteration):
    interval = int(interval)
    iteration = int(iteration)
    sample_iteration = int(sample_iteration)
    tripinfo_w = pd.read_csv(os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(sample_iteration)+
                            "_trip_info.csv"), sep=";")

    sumolib.subprocess.call(["python3", os.path.join(option.tools,"xml","xml2csv.py"),
                              os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(sample_iteration)+"_routes_sim.xml")])

    routes_w = pd.read_csv(os.path.join(option.output_location,str(interval),str(iteration)+"_"+str(sample_iteration)+
                        "_routes_sim.csv"), sep=";")
    tripinfo_w = tripinfo_w[tripinfo_w["tripinfo_depart"]==-1]
    tripinfo_w = tripinfo_w.rename(columns={"tripinfo_id":"id"})[["id"]]
    routes_w = routes_w.rename(columns={"vehicle_id":"id"})[["id","route_edges"]]
    waiting  = tripinfo_w.merge(routes_w, on="id", how="inner")
    net = sumolib.net.readNet(option.net_file , withInternal=True)

    waiting["from_node"] = waiting["route_edges"].apply(lambda x:net.getEdge(x.split(" ")[0]).getFromNode().getID())
    waiting["to_node"] = waiting["route_edges"].apply(lambda x:net.getEdge(x.split(" ")[-1]).getFromNode().getID())
    if len(waiting)!=0:
        waiting = waiting.groupby(["from_node", "to_node"]).apply(len).reset_index().rename(columns={0:"number_of_trips"})
    return waiting
###########################################################################################

def main():
    option = get_options()
    option = update_option(option)
    get_logger(log, "run", option.output_location)
    log.info('-------------------------------------------------------------------------')
    log.info("initializing")
    initializing(option)
    log.info("reading data")
    report = dict()
    report_fieldnames = ["interval", "iteration", "vehicle_nummber_calib", "sample_number",
                         "vehicle_number_sampling","sensor_error_calibration", "sensor_error_simulation",
                         "sensor_error_sim2calib","running", "waiting","speed","fixed_point_error","teleport"]
    
    with open(os.path.join(option.output_location,'report.csv'), 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=report_fieldnames)
        writer.writeheader()

        
    real_data = pd.read_csv(os.path.join(option.output_location,"real_data.csv"))
    routes = pd.read_csv(os.path.join(option.output_location,"route_file.csv"), low_memory=False)
    sensor_extra = pd.read_csv(option.sensor_extra)

    edge2id = real_data.set_index("edge")["id"].to_dict()
    net = sumolib.net.readNet(option.net_file, withInternal=True)

    init_edge2time = dict()
    for edge in net.getEdges():
        init_edge2time[edge.getID()] = edge.getLength()/edge.getSpeed()

    best_edge2time = init_edge2time.copy()
 
    
    vfli = 0
    waiting = 6
    last_best_iteration = -1
    last_best_sample_iteration = -1
    MAX_WAITING = np.inf
    od_waiting = pd.DataFrame()

    for interval in range(int(option.interval_begin), int(option.interval_end)+1):
        dod = read_init_dod(option, interval)
        no_trips = len(dod["trip_id"].unique())

        sensor_set = set(real_data["edge"].unique())
        no_sensors = len(real_data[real_data["interval"]==interval]["id"].unique())

        if option.weight_calibration is None:
            option.weight_calibration = str(np.sqrt(no_sensors/no_trips)) #* 10
        report["interval"] = interval
        best_fixed_point_error = 10
        best_error_sensor = 1
        routes = route_update(routes, option, edge2id, best_edge2time)
        sensorOnRoute_df = update_sensorOnRoute(routes, int(option.interval_size))
        spp = update_spp(sensorOnRoute_df, trips_len=no_trips, sensors_len=no_sensors)
        best_waiting = MAX_WAITING
        best_average_speed = 0
        best_iteration = -1
        best_sample_iteration = -1
        best_all_routes = None
        best_vehicle_from_last_interval = 0
        sensors_count = real_data[real_data["interval"]== interval]["count"].sum()
        _dir = option.output_location+str(interval)
        if (not os.path.isdir(_dir)):
            os.mkdir(_dir,mode=0o777) 
        iteration = 0
        waiting = MAX_WAITING
        edge2time_list = list()
        edge2time_list.append(init_edge2time)
        while  ( iteration < int(option.number_iteration)):
            report["iteration"] = iteration
            log.info("interval = " + str(interval) + "    " +"iteration = " + str(iteration) + "     " + 
                     "    " + "sensor_count = " + str(sensors_count) +
                     "    " + "sensor_count_supported_last_interval = " + str(sensor_extra["count"].sum()) +
                     "    " + "vehicle_from_last_interval = " + str(vfli))
            log.info("    init OD updating")
            init_od = update_init_od(option, dod, spp, sensors_count - sensor_extra["count"].sum(), od_waiting)

            log.info("    calibration")

            sensor_error, od_estimation, report["vehicle_nummber_calib"] = calibration( 
                                                  spp,
                                                  init_od,
                                                  real_data,
                                                  sensor_extra,
                                                  option,
                                                  no_sensors=no_sensors, no_trips = no_trips,
                                                  interval=interval, iteration=iteration)
            
            log.info("        number of vehicles in calibration : " + str(report["vehicle_nummber_calib"]))

            average_edge2time = dict()
            for edge in init_edge2time.keys():
                average_edge2time[edge] = 0
            best_sample_sensor_error = 10
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            oldedge2time = edge2time_list[-1]
            processes = []

            for sample_iteration in range(int(option.sample_iteration)):
                    p = multiprocessing.Process(target = parallel_sampling, args=(report.copy(), od_estimation, routes, option, interval, iteration,
                                                                                  sample_iteration, dod,last_best_iteration,
                                                                                  last_best_sample_iteration, net, oldedge2time, return_dict))
                    p.start()
                    processes.append(p)

            for p in processes:
                    p.join()    

            
                        
            best_sample = -1
            best_sample_sensor_error = 1000

            for i in range(len(return_dict)):
                if return_dict[i]["sensor_error_simulation"] < best_sample_sensor_error:
                    best_sample_sensor_error = return_dict[i]["sensor_error_simulation"]
                    best_sample = i

            best_report = return_dict[best_sample]
            average_edge2time = best_report["newedge2time"]
            fixed_point_error = best_report["fixed_point_error"]

            log.info("        best sample : " + str(best_sample))


            with open(os.path.join(option.output_location,"report.csv"), 'a', encoding='UTF8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=report_fieldnames)
                        for i in range(len(return_dict)):
                            myreport = dict()
                            for key in report_fieldnames:
                                myreport[key] = return_dict[i][key]
                            writer.writerow(myreport)  
          

            edge2time_list.append(average_edge2time)
            average_edge2time, edge2time_list, fixed_point_error = update_edge2time(option, init_edge2time, average_edge2time, edge2time_list)#, method="secant")
            if fixed_point_error < best_fixed_point_error:
                best_fixed_point_error = fixed_point_error

            
            ##############################################   

            log.info("    routes updating")
            routes = route_update(routes, option, edge2id, average_edge2time, interval, iteration, best_sample)
            log.info("    sensorOnRoute updating")
            sensorOnRoute_df = update_sensorOnRoute(routes, int(option.interval_size))
            log.info("    spp updating")
            spp = update_spp(sensorOnRoute_df, trips_len=no_trips, sensors_len=no_sensors)

            #### remove non necessary samples data  ###########
            clean_samples(option, interval, iteration, best_sample)
         
            #########  update best iteration  ############

            if best_error_sensor > best_sample_sensor_error:
                best_error_sensor = best_sample_sensor_error
                best_waiting = best_report["waiting"]
                best_average_speed = best_report["speed"]
                best_iteration = iteration
                best_sample_iteration = best_sample
                best_all_routes = best_report["all_routes"].copy()
                best_vehicle_from_last_interval = best_report["running"] + best_report["waiting"]
            #else:
            #    clean_iteration(option, interval, iteration)

            ##############################################    
            iteration += 1

        iteration -= 1
        log.info("sensor extra updating")
        sensor_extra = update_sensor_extra(best_all_routes, sensor_set, option, interval, best_iteration, best_sample_iteration)
        vfli = best_vehicle_from_last_interval

        log.info("{ interval : " +str(interval)+ "  best iteration  : " + str(best_iteration) +  "  best sampling : "+ str(best_sample_iteration)+
                 "  best waiting  : " + str(best_waiting) +
                 "  best average speed : " + str(best_average_speed) + " }")
      
        last_best_iteration = best_iteration
        last_best_sample_iteration = best_sample_iteration
        od_waiting = calculate_OD_waiting(option=option, interval=interval, iteration=last_best_iteration, sample_iteration=last_best_sample_iteration)
        od_waiting.to_csv(os.path.join(option.output_location,str(interval) +"_od_waiting.csv"), index=False)



if __name__=="__main__":
    main()
