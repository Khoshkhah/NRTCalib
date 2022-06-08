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

from collections import Counter
import shutil
from shapely.geometry import LineString, Polygon


sumolib.os.environ["SUMO_HOME"]="/home/kaveh/build/sumo"

if 'SUMO_HOME' in sumolib.os.environ:
    tools = sumolib.os.path.join(sumolib.os.environ['SUMO_HOME'], 'tools')
    sumolib.sys.path.append(tools)
else:   
    sumolib.sys.exit("please declare environment variable 'SUMO_HOME'")

#sumoBinary = "/home/kaveh/notebook/jupyterenv/bin/sumo"
sumoBinary = "/home/kaveh/virenv/bin/sumo"





def get_logger(log, name: str, log_dir):
    #log = logging.getLogger(name)
    log.setLevel(logging.INFO)

    # define file handler and set formatter
    #log_dir = "log"
    #log_dir = os.environ.get('MODSPLIT_HOME', '')+"/log"

    if (not os.path.isdir(log_dir)):
        os.mkdir(log_dir,mode=0o777) 

    file_handler = logging.FileHandler(log_dir+ f'/{name}.log')
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    return log

#########################################
log = logging.getLogger("calibration")

#log = get_logger("test")
#log.info("----------------------------------------------------------------------------")




def get_options(args=None):
    argParser = sumolib.options.ArgumentParser()
    argParser.add_argument("-n", "--net-file", help="the SUMO net filename")
    argParser.add_argument("-m", "--measurment-file", help="the measurment filename contains sensor data") 
    argParser.add_argument("-r", "--route-file", help="the XML routes filename") 
    argParser.add_argument("-dod", "--dod-file", help="the init distributed origin destination matrix filename")
    argParser.add_argument("-ni", "--number-iteration", help="number of iteration for each interval, default is 20")
    argParser.add_argument("-is", "--interval-size", help="the size of each interval")
    argParser.add_argument("-ib", "--interval-begin", help="the number of beginnig interval")
    argParser.add_argument("-ie", "--interval-end", help="the number of of ending interval")
    argParser.add_argument("-l", "--output-location", help="the location of output files")
    argParser.add_argument("-se", "--sensor-extra", help="the sensor data extra file")
    argParser.add_argument("-teta", "--teta", help="teta for logit, default is -0.00001 ")
    argParser.add_argument("-nsl", "--netstate-loading", help="load net-state from the previous interval, default is true")
    argParser.add_argument("-scalenumber", "--scale-number", help="it is one over expectation sensor crossing for a random trip, default is automated calculation ")
    argParser.add_argument("-wc", "--weight-calibration", help="weight in optimization formula in calibration. default is automated calculation")
    argParser.add_argument("-bw", "--best-waiting", help="the iteration is finished when number of waiting vehicle is less than this number \
                                                           or the number of iteration is more than iteration number. default is 0")

    argParser.add_argument("-mr", "--max-routes", help="maximum number of routes that saved in the database fo each trips, default is 10")

    options = argParser.parse_args(args=args)

    if options.net_file is None or options.measurment_file is None or options.route_file is None or \
       options.interval_size is None or options.route_file is None:
        argParser.print_help()
        sys.exit()
    return options

def clean_measurment(option):
    measurment = pd.read_csv(option.measurment_file)
    m_dict = measurment.set_index(["edge","interval"])["count"].to_dict()

    sensor_edge = measurment[["edge"]].drop_duplicates().reset_index(drop=True).reset_index()
    sensor_edge = sensor_edge.rename(columns={"index":"id"})
    sensor_edge["edge"] = sensor_edge["edge"].astype(str)
    mylist = list()
    for index, row in sensor_edge.iterrows():
        for interval in range(int(option.interval_begin), int(option.interval_end)):
            if (row.edge, interval) in m_dict.keys():
                mylist.append({"edge":row.edge, "id":row.id, "interval":interval, "count":m_dict[(row.edge, interval)]})
            else:
                mylist.append({"edge":row.edge, "id":row.id, "interval":interval, "count":0})

    cleand_measurment = pd.DataFrame(mylist)
    cleand_measurment.to_csv(option.output_location+"real_data.csv", index=False)
    return cleand_measurment

def update_option(option):
    
    if (not os.path.isdir(option.output_location)):
        os.mkdir(option.output_location,mode=0o777) 
    
    real_data = pd.read_csv(option.output_location+"real_data.csv")
    if option.max_routes is None:
        option.max_routes = "10"
    if option.best_waiting is None:
        option.best_waiting = "0"
    if option.netstate_loading is None:
        option.netstate_loading = "true"
    if option.teta is None:
        option.teta=str(-0.00001)
    if option.interval_begin is None:
        option.interval_begin = str(real_data["interval"].min())

    if option.interval_end is None:
        option.interval_end = str(real_data["interval"].max())
        
    if option.number_iteration is None:
        option.number_iteration = str(20)
    if option.sensor_extra is None:
        sensor_extra = real_data.groupby(["edge"]).first().reset_index()
        sensor_extra["count"] = 0
        sensor_extra["interval"] = int(option.interval_begin)
        sensor_extra.to_csv(option.output_location+"sensor_extra.csv", index=False)
        option.sensor_extra = option.output_location+"sensor_extra.csv"
        
    
    clean_measurment(option)
    
    return option



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
        if edge in measurment_edge2id.keys():
            time_list.append(str(_time))
        #_time += edge_id2time[edge]
        _time += edge2time[edge]
    if(len(time_list)==0):
        return None
    else:
        return " ".join(time_list)



def update_sensorOnRoute(routes, interval_size):

    sensorOnRoute = list()
    for index, row in routes.iterrows():
        #print(row.sensors)
        #if not (row.sensors is None):
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

def update_init_od(dod, spp, sensor_count):
    if sensor_count < 0:
        sensor_count = 0 
    init_od = dod.copy()
    spp_1 = spp.groupby("trip_id")["prob"].apply(sum).reset_index()
    test = init_od.merge(spp_1, on="trip_id")
    if option.scale_number is None:
        scale = 1/((test["prob"]*test["weight_trip"]).sum())
    else:
        scale = float(option.scale_number)
    init_od["number_of_trips"] = init_od["weight_trip"].apply(lambda x: x*scale*sensor_count)
    if (len(init_od[init_od["number_of_trips"] >= 0]) < len(init_od)):
        log.info("error: negative or None value in init OD.")
        sys.exit("error: negative or None value in init OD.")
    init_od = init_od.drop(columns="weight_trip")
    log.info("        number of vehicles from init OD : " + str(int(init_od["number_of_trips"].sum())))

    return init_od


def initializing(option):
    sumolib.subprocess.call(["python3", tools+"/xml/xml2csv.py", option.route_file, "-o", option.output_location+"route_file.csv"])
    routes = pd.read_csv(option.output_location+"route_file.csv", sep=";")
    routes["from_edge"] = routes["route_edges"].apply(lambda x: x.split(" ")[0])
    routes["to_edge"] = routes["route_edges"].apply(lambda x: x.split(" ")[-1])
    routelen = len(routes)
    dod = pd.read_csv(option.dod_file)
    dodlen = len(dod)
    routes = routes.merge(dod, on=["from_edge", "to_edge"], how="left")
    routes = routes[routes["trip_id"].notnull()]
    routelen2 = len(routes)
    log.info("    remove " + str(routelen - routelen2) + " routes without trip weight")
    dod = routes.groupby(["from_edge", "to_edge"]).first().reset_index()[["trip_id", "from_edge", "to_edge", "weight_trip"]]
    dodlen2 = len(dod)
    log.info("    remove " + str(dodlen - dodlen2) + " trips without any route")
    real_data = pd.read_csv(option.output_location+"real_data.csv")
    measurment_edge2id = real_data.groupby("edge").first().reset_index().set_index("edge")["id"].to_dict()
    trip2n_routes = routes.groupby(["from_edge","to_edge"]).apply(len)
    routes["weight_route"] = routes.apply(lambda row: 1/trip2n_routes[row["from_edge"], row["to_edge"]], axis=1)
    
    routes["sensors"] = routes['route_edges'].apply(lambda x: set_sensor_list_in_route(x, measurment_edge2id))
    
    edges_set = set.union(*(routes["route_edges"].apply(lambda x: set(x.split(" ")))))
    edge2time = dict()
    net = sumolib.net.readNet(option.net_file)

    for edge_id in edges_set:
        edge = net.getEdge(edge_id)
        edge2time[edge_id] = edge.getLength()/edge.getSpeed()

   
    
    routes["travel_time"] = routes["route_edges"].apply(lambda x: sum([edge2time[i] for i in x.split(" ")]))

    routes["sensors_time"] = routes["route_edges"].apply(lambda x: sensorTimeCalc(x, measurment_edge2id, edge2time))
    routes.to_csv(option.output_location+"route_file.csv", index=False)
    
    sensorOnRoute_df = update_sensorOnRoute(routes, int(option.interval_size))
    sensorOnRoute_df.to_csv(option.output_location+"sensorOnRoute_micro.csv", index=False)

    trips_len = len(dod["trip_id"].unique())
    sensors_len = len(real_data["id"].unique())
                     
    spp = update_spp(sensorOnRoute_df, trips_len=trips_len, sensors_len=sensors_len)
    spp.to_csv(option.output_location+"spp_micro.csv", index=False)

    #sensors_count = real_data[real_data["interval"]==int(option.interval_begin)]["count"].sum()
    #init_od = update_init_od(dod, spp, sensors_count)
    #init_od.to_csv(option.output_location+"init_od.csv", index=False)
                     

def calibration(
                spp:pd.DataFrame,
                init_od:pd.DataFrame,
                real_data:pd.DataFrame,
                _sensor_data_extra,
                option,
                no_sensors=46, no_trips = 10005,
                interval=0, iteration=0)->Tuple[pd.DataFrame, pd.DataFrame]:
   
   # temp_df = pd.DataFrame(index=list(range(no_sensors))).reset_index()
   # temp_df = temp_df.rename(columns={"index":"id"})
   # real_data_temp = temp_df.merge(real_data[real_data["interval"]==interval], on=["id"], how="left").fillna(0)
   
    # sensor_data_extra = _sensor_data_extra.set_index("edge")["count"].to_dict()
    #sensor_data_extra['0'] = 0 
    #sensor_data_extra[0] = 0 
    #print(len(sensor_data_extra))
    edge2realcount = real_data[real_data["interval"]==interval].set_index("edge")["count"].to_dict()
    real_data_temp = _sensor_data_extra.copy()
    #print(real_data_temp)
    real_data_temp["real_count"] = real_data_temp["edge"].apply(lambda x: edge2realcount[x] if x in edge2realcount.keys() else 0)
    real_data_temp["count"] = real_data_temp.apply(lambda row: max(0, row["real_count"] - row["count"]), axis = 1)
    real_data_temp = real_data_temp.rename(columns={"index":"id"}).sort_values("id")
    #real_data_temp["count"] = real_data_temp.apply(lambda row: row["count"] - sensor_data_extra[row.edge] if
    #                          (row["count"] - sensor_data_extra[row.edge])>0 else 0, axis=1)
    real_data_temp = real_data_temp.set_index("id")["count"]
    #print(real_data_temp)

    spp_df = spp.copy()
    A_x = np.zeros((no_sensors, no_trips))
    for sensor in range(no_sensors):
        A_x[sensor,:] = np.array(spp_df[spp_df["sensor"]==sensor].sort_values(["trip_id"]).reset_index(drop=True)["prob"])
    weight = option.weight_calibration #np.sqrt(no_sensors/no_trips) * 0.1

    """
    AA = np.concatenate((A_x, np.identity(no_trips)*weight), axis=0)
    row_list = list()
    collumn_list = list()
    data_list = list()
    log.info("          step 2")

    
    for i in range(AA.shape[0]):
        for j in range(AA.shape[1]):
            if AA[i][j]!=0:
                row_list.append(i)
                collumn_list.append(j)
                data_list.append(AA[i][j])
    """

    ###################

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

    ###################


    shapeA = (A_x.shape[0]+A_x.shape[1], A_x.shape[1])
    A= bsr_matrix((np.array(data_list), (np.array(row_list), np.array(collumn_list))), shape=shapeA)  
    lb = np.zeros(A_x.shape[1])
    ub = np.ones(A_x.shape[1])*np.inf 


    init_od_temp = init_od.sort_values("trip_id").reset_index(drop=True)
    #s = np.concatenate((np.array(init_od_temp["number_of_trips"]), np.zeros(no_sensors)))
    b = np.concatenate((np.array(real_data_temp), weight*np.array(init_od_temp["number_of_trips"])), axis=0)
    log.info("        starting lsq_linear ...")

    res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto', verbose=1) 
    log.info("        ended lsq_linear ...")
             
    np.savez_compressed(option.output_location + str(interval)+ "/"+str(iteration)+"_calib.npz", A=A_x, x=res.x, b=b)
    x_df = pd.DataFrame(res.x).rename(columns={0:"number_of_trips"}).reset_index().rename(columns={"index":"trip_id"})
    sensor_error = (np.dot(A_x, res.x)-b[:A_x.shape[0]])
    mysensor_error_dict = dict()
    for i in range(no_sensors):
        mysensor_error_dict[(i,interval)] =sensor_error[i]     
    index_e = pd.MultiIndex.from_tuples(mysensor_error_dict.keys(), name=["index","interval"])
    e_df = pd.DataFrame(index = index_e, data=mysensor_error_dict.values()).rename(columns={0:"error"}).reset_index()
    log.info("        number of vehicles in calibration : " + str(int(x_df["number_of_trips"].sum())))

    return e_df, x_df

###########################################################################################3

def route_sampling(od, _routes, option, interval,  iteration, deterministic=True):
    interval_size = int(option.interval_size)

    routes = _routes.copy()
    od2tripsNum = od.set_index(["trip_id"])["number_of_trips"].to_dict()
    id2path = routes.set_index("route_id")["route_edges"]
    all_routes = list()

    if(deterministic):
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

                
    else:
        resolution = int(interval_size/60)
        for index, row in routes.iterrows():
            x=np.random.random(int(interval_size/resolution))
            for depart in range(int(interval_size/resolution)):
                if (row.weight_route*od2tripsNum[row.trip_id]*resolution/interval_size) > 1:
                    log.info("Warning: the resolution is too low.")
                if x[depart] <= (row.weight_route*od2tripsNum[row.trip_id]*resolution/interval_size):
                    all_routes.append({"depart":interval*interval_size + depart*resolution, "route_id":row.route_id, "trip_id":row.trip_id})

    
    all_routes_df = pd.DataFrame.from_dict(all_routes).sort_values("depart").reset_index(drop=True).reset_index()
    all_routes_df["departLane"]="free"
    all_routes_df["departSpeed"]="max"

    all_routes_df["edges"]=all_routes_df["route_id"].apply(lambda x:id2path[x])
    all_routes_df.rename(columns={"index":"id"}, inplace=True)
    all_routes_df["id"] = all_routes_df["id"].apply(lambda x: str(x) + "_" + str(interval))
    vehid2pathid = all_routes_df.set_index("id")["route_id"].to_dict()
    log.info("        number of vehicles in sampling : " + str(len(all_routes_df)))
    def convert_row(row):
        return """
        <vehicle id="%s" depart="%s" departLane="%s" departSpeed="%s">
            <route edges="%s"/>
        </vehicle>""" % (row.id, row.depart, row.departLane, row.departSpeed, row.edges)

    text0 = """<?xml version="1.0" encoding="UTF-8"?>\n\n\n"""
    text1 = """<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">"""
    text2 = '\n'.join(all_routes_df.apply(convert_row, axis=1))
    text3 = """\n</routes>"""
    with open(option.output_location+str(interval)+"/"+str(iteration)+"_routes_sim.xml", 'w') as myfile: 
        myfile.write(text0+text1+text2+text3)
    return vehid2pathid, all_routes_df

#vehid2pathid, all_routes_df = route_sampling(od_estimation, routes, hour)



def calibration_2(
                routes:pd.DataFrame,
                spp:pd.DataFrame,
                init_od:pd.DataFrame,
                real_data:pd.DataFrame,
                _sensor_data_extra,
                option,
                no_sensors, no_trips,
                interval=0, iteration=0)->Tuple[pd.DataFrame, pd.DataFrame]:
   
    
    ########### prepare matrix for  edge capacity  ##########
    te = dict()
    edge_set = set()
    for index, row in routes.iterrows():
        edges = row.route_edges.split(" ")
        w = row.weight_route
        trip = row.trip_id
        for edge in edges:
            edge_set.add(edge)
            if (edge, trip) not in te.keys():
                te[(edge, trip)] = w
            else:
                te[(edge, trip)] += w

    edge_list = list(edge_set) 

    net = sumolib.net.readNet(option.net_file)
    lane_number_list = list()

    edge2index = dict()
    for index in range(len(edge_list)):
        edge2index[edge_list[index]] = index
        lane_number_list.append(net.getEdge(edge_list[index]).getLaneNumber() * int(option.interval_size))



    #####################################################    
    
    temp_df = pd.DataFrame(index=list(range(no_sensors))).reset_index()
    temp_df = temp_df.rename(columns={"index":"id"})
    real_data_temp = temp_df.merge(real_data[real_data["interval"]==interval], on=["id"], how="left").fillna(0)
    #print(real_data_temp)
    #print(sensor_data_extra)
    sensor_data_extra = _sensor_data_extra.set_index("edge")["count"].to_dict()
    #sensor_data_extra['0'] = 0 
    #sensor_data_extra[0] = 0 
    #print(len(sensor_data_extra))
    real_data_temp["count"] = real_data_temp.apply(lambda row: row["count"] - sensor_data_extra[row.edge] if
                              (row["count"] - sensor_data_extra[row.edge])>0 else 0, axis=1)
    real_data_temp = real_data_temp.set_index("id")["count"]
    spp_df = spp.copy()
    A_x = np.zeros((no_sensors, no_trips))
    for sensor in range(no_sensors):
        A_x[sensor,:] = np.array(spp_df[spp_df["sensor"]==sensor].sort_values(["trip_id"]).reset_index(drop=True)["prob"])
    weight = option.weight_calibration #np.sqrt(no_sensors/no_trips) * 0.1

    """
    AA = np.concatenate((A_x, np.identity(no_trips)*weight), axis=0)
    row_list = list()
    collumn_list = list()
    data_list = list()
    log.info("          step 2")

    
    for i in range(AA.shape[0]):
        for j in range(AA.shape[1]):
            if AA[i][j]!=0:
                row_list.append(i)
                collumn_list.append(j)
                data_list.append(AA[i][j])
    """

    ###################

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
        
    for item in te.keys():
        row_list.append(edge2index[item[0]] + A_x.shape[0] + A_x.shape[1])
        collumn_list.append(item[1])
        data_list.append(te[item])
                
    for i in range(len(edge_list)):
        row_list.append(i+A_x.shape[0] + A_x.shape[1])
        collumn_list.append(i+A_x.shape[1])
        data_list.append(1)
        

    ###################


    shapeA = (A_x.shape[0]+A_x.shape[1] + len(edge_list), A_x.shape[1]+len(edge_list))
    A= bsr_matrix((np.array(data_list), (np.array(row_list), np.array(collumn_list))), shape=shapeA)  
    lb = np.zeros(A_x.shape[1] + len(edge_list))
    ub = np.ones(A_x.shape[1] + len(edge_list))*np.inf 


    init_od_temp = init_od.sort_values("trip_id").reset_index(drop=True)
    #s = np.concatenate((np.array(init_od_temp["number_of_trips"]), np.zeros(no_sensors)))
    b = np.concatenate((np.array(real_data_temp), weight*np.array(init_od_temp["number_of_trips"]), 
                       np.array(lane_number_list)), axis=0)
    
    print(b.shape)
    print(A.shape)
    
    log.info("        starting lsq_linear ...")

    res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto', verbose=2) 
    log.info("        ended lsq_linear ...")
             
    np.savez_compressed(option.output_location + str(interval)+ "/"+str(iteration)+"_calib.npz", A=A_x, x=res.x, b=b)
    x_df = pd.DataFrame(res.x[0:A_x.shape[1]]).rename(columns={0:"number_of_trips"}).reset_index().rename(columns={"index":"trip_id"})
    sensor_error = (np.dot(A_x, res.x[0:A_x.shape[1]])-b[:A_x.shape[0]])
    mysensor_error_dict = dict()
    for i in range(no_sensors):
        mysensor_error_dict[(i,interval)] =sensor_error[i]     
    index_e = pd.MultiIndex.from_tuples(mysensor_error_dict.keys(), name=["index","interval"])
    e_df = pd.DataFrame(index = index_e, data=mysensor_error_dict.values()).rename(columns={0:"error"}).reset_index()
    log.info("        number of vehicles in calibration : " + str(int(x_df["number_of_trips"].sum())))

    return e_df, x_df

###########################################################################################3



def create_edge_data_add(option, interval, iteration):
        
    text0 = """<?xml version="1.0" encoding="UTF-8"?>\n\n\n"""
    text1 = """<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
"""
    text2 = """
        <edgeData id="%s" freq="%s" file="%s" excludeEmpty="%s"/>""" % ("dump_", option.interval_size, str(iteration)+"_aggregated.xml", "false")

    text3 = """\n</additional>"""
    with open(option.output_location+str(interval)+"/"+str(iteration)+"_edge_data_add.xml", 'w') as myfile: 
        myfile.write(text0+text1+text2+text3)
#################################################
#interval = int(option.interval_begin)
#create_edge_data_add(option, interval)



def sumoCommand(sumoBinary, option, interval, iteration):
    diroutput = option.output_location+str(interval)+"/"
    if (not os.path.isdir(diroutput)):
        os.mkdir(diroutput,mode=0o777) 
    sc_dict = dict()
    sc_dict["--net-file"] = option.net_file
    sc_dict["--route-files"] = diroutput+ str(iteration)+"_routes_sim.xml"
    sc_dict["--additional-files"] = diroutput + str(iteration) + "_edge_data_add.xml"
    sc_dict["--tripinfo-output"] = diroutput + str(iteration)+"_trip_info.xml"
    sc_dict["--statistic-output"] = diroutput + str(iteration)+"_statistic_output.xml"
    sc_dict["--vehroute-output"] =  diroutput + str(iteration)+"_vehroute_output.xml"
    #sc_dict["--tripinfo-output.write-unfinished"] = "true"
    sc_dict["--tripinfo-output.write-undeparted"] = "true"
    sc_dict["--vehroute-output.write-unfinished"] = "true"
    sc_dict["--vehroute-output.exit-times"] = "true"
    sc_dict["--no-warnings"] = "true"
    sc_dict["--no-step-log"] = "true"
    #sc_dict["--save-state.rng"] ="true"
    #sc_dict["--threads"] ="16"
    sc_dict["--seed"] = "0"
    sc_dict["--random"] = "false"


    #sc_dict["--vehroute-output.sorted"] = "true"
    #if(not eval(option.netstate_loading.title())):
    sc_dict["--begin"] = str(interval * int(option.interval_size))

    sumoCmd = list()
    sumoCmd.append(sumoBinary)
    for key in sc_dict.keys():
        sumoCmd.append(key)
        sumoCmd.append(sc_dict[key])
    return sumoCmd


#import traci.constants as tc

def simulation(sumoBinary, option, interval, iteration, dod, last_best_iteration, net):
    
    ft2id_trips = dod.set_index(["from_edge", "to_edge"])["trip_id"].to_dict()
    ft2weight_trips = dod.set_index(["from_edge", "to_edge"])["weight_trip"].to_dict()
    sumoCmd = sumoCommand(sumoBinary, option, interval, iteration)
    traci.start(sumoCmd, port=8813, traceFile="traci.log")
    #departed = 0
    #arrived = 0
    savedstate = option.output_location+str(interval-1)+"/"+str(last_best_iteration)+"_savedstate.xml"
    if os.path.isfile(savedstate) and (eval(option.netstate_loading.title())):
        traci.simulation.loadState(savedstate)
        log.info("        loading "+ str(len(traci.vehicle.getIDList())) +" vehicles from the last interval  ") 

      #  departed += traci.simulation.getDepartedNumber()
      #  arrived += traci.simulation.getArrivedNumber()
    step = 0
    
    tripset = set()
    count = 0
    edge2time = dict()
    for edge in net.getEdges():
        edge2time[edge.getID()] = 0

    while step < int(option.interval_size):
        idlist = traci.simulation.getLoadedIDList()
        for _id in idlist:
            _route =  traci.vehicle.getRoute(_id)
            _from = _route[0]
            _to = _route[-1]
            tripset.add((_from, _to))

        if(step % 100 == 0):
            count+= 1
            for edge in net.getEdges():
                edge2time[edge.getID()] += traci.edge.getTraveltime(edge.getID())
        
        traci.simulationStep()
        #departed += traci.simulation.getDepartedNumber()
        #arrived += traci.simulation.getArrivedNumber()
        step += 1
    

    for edge in net.getEdges():
        edge2time[edge.getID()] /= count
    #traci.simulationStep()
    state = traci.simulation.saveState(option.output_location+str(interval)+"/"+str(iteration)+"_savedstate.xml")
    #log.info("        vehicle inserted  = " + str(departed))
    #log.info("        vehicle running   = " + str(departed - arrived))
    new_route = list()
    log.info("        tripset length = " + str(len(tripset)))
    for item in tripset:
        route_edges = " ".join(traci.simulation.findRoute(item[0], item[1], routingMode=traci.constants.ROUTING_MODE_AGGREGATED).edges)
        new_route.append({"trip_id":ft2id_trips[item], "from_edge":item[0], "to_edge":item[1],
                         "weight_trip":ft2weight_trips[item], "route_edges":route_edges})
        
    traci.close()
    return pd.DataFrame(new_route), edge2time




def route_update(_routes, option, interval, edge2id, edge2time, net, iteration):
  
    teta = float(option.teta)
    #sumolib.subprocess.call(["python3", tools+"/xml/xml2csv.py", option.output_location+str(interval)+"/"+str(iteration)+"_aggregated.xml"])
    #aggregated = pd.read_csv(option.output_location+str(interval)+"/"+str(iteration)+"_aggregated.csv", sep=";")
    #alledges = set(aggregated["edge_id"])
    #aggregated = aggregated[aggregated["edge_traveltime"].notna()]

    #edge2time = aggregated.set_index("edge_id")["edge_traveltime"].to_dict()

    #for key in alledges:
    #    if not(key in edge2time.keys()):
    #        edge = net.getEdge(key)
    #        edge2time[key] = edge.getLength()/edge.getSpeed()

    _routes["travel_time"] = _routes["route_edges"].apply(lambda x: sum([edge2time[i] for i in x.split(" ")]))
    _routes["sensors_time"] = _routes["route_edges"].apply(lambda x: sensorTimeCalc(x, edge2id, edge2time))
    
    ### routes weight updating ###
    _routes["travel_time_exp"] = _routes["travel_time"].apply(lambda x: np.exp(x*teta))
    trip_id2weight = _routes.groupby("trip_id")["travel_time_exp"].apply(sum)
    _routes["weight_route"] = _routes.apply(lambda row: row.travel_time_exp/trip_id2weight[row.trip_id], axis=1)
    _routes = _routes.drop(columns={"travel_time_exp"})
    
    trip_n = _routes.groupby("trip_id").apply(len).reset_index().rename(columns={0:"count"})
    trip_0 = _routes.groupby("trip_id").apply(lambda row: -1 if len(row)<int(option.max_routes) else row.iloc[np.argmin(row.weight_route)]["route_id"])
    remove_route_set = set(trip_0[trip_0!=-1])
    _routes = _routes[_routes["route_id"].apply(lambda x: x not in remove_route_set)]
    _routes = _routes.drop(columns=["route_id"])
    _routes = _routes.reset_index(drop=True).reset_index().rename(columns={"index":"route_id"})
    
    _routes.to_csv(option.output_location + "last_routes.csv", index = False)
    return _routes



def update_sensor_extra(all_routes, sensor_extra, option, interval, iteration):
    
    sumolib.subprocess.call(["python3", tools+"/xml/xml2csv.py", option.output_location+str(interval)+"/"+str(iteration)+"_vehroute_output.xml"])
    sumolib.subprocess.call(["python3", tools+"/xml/xml2csv.py", option.output_location+str(interval)+"/"+str(iteration)+"_trip_info.xml"])
    sensor_set = set(sensor_extra["edge"])
    vehroute = pd.read_csv(option.output_location+str(interval)+"/"+str(iteration)+"_vehroute_output.csv", sep=";")
    vehroute["vehicle_arrival"] = vehroute["vehicle_arrival"].fillna(-1)
    tripinfo = pd.read_csv(option.output_location+str(interval)+"/"+str(iteration)+"_trip_info.csv", sep=";")
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

    sensor_extra["count"] = sensor_extra["edge"].apply(lambda x: sensor_data_extra[x] if x in sensor_data_extra.keys() else 0)
    sensor_extra["interval"] = interval + 1
    #for sensor in sensor_set:
    #    if sensor not in sensor_data_extra.keys():
    #        se.append({"edge":sensor , "interval":interval+1 , "count":0})
    #    else:
    #       se.append({"edge":sensor , "interval":interval+1 , "count":sensor_data_extra[sensor]})
    return  sensor_extra#pd.DataFrame(se)



# importing element tree
import xml.etree.ElementTree as ET 

def logging_statistics(option, interval, iteration):
    sumolib.subprocess.call(["python3", tools+"/xml/xml2csv.py", option.output_location+str(interval)+"/"+str(iteration)+"_trip_info.xml"])
    trip_info = pd.read_csv(option.output_location+str(interval)+"/"+str(iteration)+"_trip_info.csv", sep=";")
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
    #log.info("        loaded = " +str(loaded) + "  inserted = " + str(inserted) +
    #         "   running = " + str(running) + "   waiting = " + str(waiting))


    tree = ET.parse(option.output_location+str(interval)+"/"+str(iteration)+"_statistic_output.xml") 

    root = tree.getroot() 
    for key in root[0].attrib.keys():
        root[0].attrib[key] = int(root[0].attrib[key])

    for key in root[1].attrib.keys():
        root[1].attrib[key] = int(root[1].attrib[key])

    for key in root[4].attrib.keys():
        root[4].attrib[key] = float(root[4].attrib[key])

    vehicleTripStatistics = dict()
    vehicleTripStatistics["speed"] = root[4].attrib["speed"] * 3.6

    log.info("        " + root[0].tag + "_till_now:  " + str(root[0].attrib))
    log.info("        " + root[1].tag + "_till_now: " + str(root[1].attrib))
    log.info("        " + root[4].tag + "_till_now:   " + str(vehicleTripStatistics))
    return running , waiting , vehicleTripStatistics["speed"]


def getShape(edge, net):
    nodefrom = edge.getFromNode()
    nodeto = edge.getToNode()
    coord_from = net.convertXY2LonLat(nodefrom.getCoord()[0], nodefrom.getCoord()[1])
    coord_to = net.convertXY2LonLat(nodeto.getCoord()[0], nodeto.getCoord()[1])
    _shape = list()
    if not edge.isSpecial():
        _shape.append(coord_from)
    for point in edge.getRawShape():  #getShape()
        _shape.append(net.convertXY2LonLat(point[0], point[1]))
    
    if not edge.isSpecial():
        _shape.append(coord_to)
    return LineString(_shape)


def viz_edge_count(od, routes, aggregated, net, option, interval, iteration):
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
                      "number_lane":edge.getLaneNumber(), "shape":getShape(edge, net),
                       "expected_count": te[edge.getID()], "sim_count":id2count_sim[edge.getID()]})

    edges_count_data = pd.DataFrame(mylist)
    edges_count_data.to_csv(option.output_location+str(interval)+"/" +str(iteration)+"_edge_count_viz.csv", index=False)



def main(option):
    option = update_option(option)
    get_logger(log, "run", option.output_location)
    log.info('-------------------------------------------------------------------------')
    log.info("initializing")
    initializing(option)
    log.info("reading data")
    dod = pd.read_csv(option.dod_file)
    spp = pd.read_csv(option.output_location+"spp_micro.csv")
    #init_od = pd.read_csv(option.output_location+"init_od.csv")
    real_data = pd.read_csv(option.output_location+"real_data.csv")
    routes = pd.read_csv(option.output_location+"route_file.csv")
    sensor_extra = pd.read_csv(option.sensor_extra)

    edge2id = real_data.groupby("edge").first().reset_index().set_index("edge")["id"].to_dict()
    sensor_set = set(real_data["edge"].unique())
    no_sensors = len(real_data["id"].unique())
    no_trips = len(dod["trip_id"].unique())
    net = sumolib.net.readNet(option.net_file)
    vfli = 0
    waiting = 6
    last_best_iteration = -1
    MAX_WAITING = np.inf


    if option.weight_calibration is None:
        option.weight_calibration = np.sqrt(no_sensors/no_trips)*100 # * 0.1 #100

    for interval in range(int(option.interval_begin), int(option.interval_end)): #range(int(option.interval_begin), int(option.interval_end)+1):
        best_waiting = MAX_WAITING
        best_average_speed = 0
        best_iteration = -1
        best_all_routes = None
        best_vehicle_from_last_interval = 0
        sensors_count = real_data[real_data["interval"]== interval]["count"].sum()
        _dir = option.output_location+str(interval)
        if (not os.path.isdir(_dir)):
            os.mkdir(_dir,mode=0o777) 
        iteration = 0
        waiting = MAX_WAITING
        edge2time_list = list()
        while (waiting > 0) and ( iteration < int(option.number_iteration)):
            #waiting = 0
            log.info("interval = " + str(interval) + "    " +"iteration = " + str(iteration) +
                     "    " + "sensor_count = " + str(sensors_count) +
                     "    " + "sensor_count_supported_last_interval = " + str(sensor_extra["count"].sum()) +
                     "    " + "vehicle_from_last_interval = " + str(vfli))
            log.info("    init OD updating")
            init_od = update_init_od(dod, spp, sensors_count - sensor_extra["count"].sum())

            log.info("    calibration")

            sensor_error, od_estimation = calibration(  #  calibration_2(routes,
                                                  spp,
                                                  init_od,
                                                  real_data,
                                                  sensor_extra,
                                                  option,
                                                  no_sensors=no_sensors, no_trips = no_trips,
                                                  interval=interval, iteration=iteration)
            log.info("    sampling")
            vehid2pathid, all_routes = route_sampling(od_estimation, routes, option, interval, iteration)  

            log.info("    simulation")
            create_edge_data_add(option, interval, iteration)
            new_routes, edge2time = simulation(sumoBinary, option, interval, iteration, dod, last_best_iteration, net)
            edge2time_list.append(edge2time)
            #sumolib.subprocess.call(["python3", tools+"/xml/xml2csv.py", option.output_location+ str(interval)+"/"+str(iteration)+"_aggregated.xml"])
            #aggregated = pd.read_csv(option.output_location+str(interval)+"/" +str(iteration)+"_aggregated.csv", sep=";")
            #edge2time = aggregated.set_index("edge_id")["edge_traveltime"].to_dict()

            #viz_edge_count(od_estimation,routes, aggregated, net, option, interval, iteration)

            #########  update best iteration  ############
            running, waiting, average_speed = logging_statistics(option, interval, iteration)

            if best_average_speed < average_speed:
                best_waiting = waiting
                best_average_speed = average_speed
                best_iteration = iteration
                best_all_routes = all_routes.copy()
                best_vehicle_from_last_interval = running + waiting

            ##############################################   


            ############   adding new routes   ###########
            new_routes["sensors"] = new_routes['route_edges'].apply(lambda x: set_sensor_list_in_route(x, edge2id))
            routes = routes[list(new_routes.columns)]
            r1 = pd.concat([routes,new_routes])
            routes = r1.groupby("route_edges").first().reset_index()
            routes = routes.reset_index().rename(columns={"index":"route_id"})
            ##############################################   

            log.info("    routes updating")
            routes = route_update(routes, option, interval, edge2id, edge2time, net, iteration)
            log.info("    sensorOnRoute updating")
            sensorOnRoute_df = update_sensorOnRoute(routes, int(option.interval_size))
            log.info("    spp updating")
            spp = update_spp(sensorOnRoute_df, trips_len=no_trips, sensors_len=no_sensors)
            iteration += 1

        iteration -= 1
        log.info("sensor extra updating")
        sensor_extra = update_sensor_extra(best_all_routes, sensor_extra, option, interval, best_iteration)
        vfli = best_vehicle_from_last_interval

        log.info("{ interval : " +str(interval)+ "  best iteration  : " + str(best_iteration) + "  best waiting  : " + str(best_waiting) +
                 "  best average speed : " + str(best_average_speed) + " }")

        if best_iteration != iteration :
            #aggregated = pd.read_csv(option.output_location+str(interval)+"/" +str(best_iteration)+"_aggregated.csv", sep=";")
            edge2time = edge2time_list[best_iteration]
            #edge2id = real_data.set_index("edge")["id"].to_dict()
            log.info("    routes updating based on the best iteration")
            routes = route_update(routes, option, interval, edge2id,edge2time, net, best_iteration)
            log.info("    sensorOnRoute updating based on the best iteration")
            sensorOnRoute_df = update_sensorOnRoute(routes, int(option.interval_size))
            log.info("    spp updating based on the best iteration")
            spp = update_spp(sensorOnRoute_df, trips_len=no_trips, sensors_len=no_sensors)

        last_best_iteration = best_iteration


args = ["-n", "sample/osm.net.xml", "-m", "sample/sensor_edge_data_2021-10-16.csv", "-dod", "sample/init_dod.csv",
        "-is", "3600", "-r", "sample/tartu_routes.xml", "-l", "output3/", "--scale-number","0.7", # scale number should be depend on probability of sensor on trip in best condition of the network
        "--interval-begin", "0", "--interval-end", "24"]
option = get_options(args)

if __name__=="__main__":
    main(option)
