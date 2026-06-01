# -*- coding: utf-8 -*-
"""
Classification of cadastral parcel transformation processes and metric calculation.

The script compares cadastral parcel geometries for the 2022 and 2024 datasets,
classifies attribute/spatial transformation processes, calculates geometry
metrics, and writes municipality-level outputs.

Before running, set the input/output folders below or define the environment
variables CADASTRE_2022_DIR, CADASTRE_2024_DIR, and CADASTRAL_RESULTS_DIR.
"""

#------------------------------------------------------------------------------

from pandas import read_csv, concat
from geopandas import read_file, overlay, GeoDataFrame
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, shape
from shapely import wkt
from copy import deepcopy
from glob import glob
import networkx as nx

import sys
from os import chdir, path, makedirs, environ

script_directory = path.dirname(path.abspath(__file__))
if script_directory not in sys.path:
    sys.path.append(script_directory)
    
import numpy as np
import get_skeleton
import math
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads

import psutil
import gc

import fiona
from concurrent.futures import ProcessPoolExecutor
from chunk_processor import process_chunk
import logging

#------------------------------------------------------------------------------

def min_segment_in_polygon(polygon):
    # Calculate the minimum segment length of the polygon's exterior
    return min_segment_in_linestring(polygon.exterior)

#------------------------------------------------------------------------------

def min_segment_in_linestring(linestring):
    # Extract coordinates and calculate minimum length
    coords = list(linestring.coords)
    min_length = float('inf')
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        segment_length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        if segment_length < min_length:
            min_length = segment_length
    return min_length

#------------------------------------------------------------------------------

def minimum_length_segment(geometry):
    # Handle different types of geometries
    if isinstance(geometry, Polygon):
        return min_segment_in_polygon(geometry)
    elif isinstance(geometry, MultiPolygon):
        # Apply the function to each polygon in the MultiPolygon and get the minimum of all
        return min(minimum_length_segment(poly) for poly in geometry.geoms)
    else:
        raise TypeError("Unsupported geometry type")
        
#------------------------------------------------------------------------------

def shrink_polygon(geometry):
    
    dist = minimum_length_segment(geometry)
        
    rbound = geometry.bounds
    
    if dist == 0:
        dist = 2
        
    optimal_iter = math.ceil(np.max([(rbound[2] - rbound[0]), (rbound[3] - rbound[1])]) / dist) + 1
    
    # NOW BUILD VECTORS FOR ACCUMULATING AREA, PERIMETER, AND PARTS
    distances_array = np.zeros(optimal_iter)
    iteration_array = np.zeros(optimal_iter)
    area_array = np.zeros(optimal_iter)
    perim_array = np.zeros(optimal_iter)
    parts_array = np.zeros(optimal_iter)
    
    # NOW ACCUMULATE AREA, PERIMETER AND PARTS FOR FULL SHAPE
    distances_array[0] = 0
    iteration_array[0] = 1
    area_array[0] = geometry.area
    perim_array[0] = geometry.length
    parts_array[0] = count_parts(geometry)
    
    # PERFORM SHRINKING AND ACCUMULATE AREA, PERIMETER, AND PARTS FOR SHRUNKEN SHAPE
    for a in range (1, optimal_iter):
        
        geom_buffer = geometry.buffer((-1)*(a)*dist)
        
        if geom_buffer.is_empty == True:
            break
        
        else:
            distances_array[a] = (a)*dist
            iteration_array[a] = a+1
            area_array[a] = geom_buffer.area
            perim_array[a] = geom_buffer.length
            parts_array[a] = count_parts(geom_buffer)
            
    # Merge arrays into a DataFrame
    df = dict({'Iteration': iteration_array, 'Buffer_dist': distances_array,
               'Buffer_Area': area_array, 'Buffer_Perim': perim_array,
               'Buffer_Parts': parts_array})
    
    perimeters = list(df['Buffer_Perim'])
    cm = calculate_adjusted_complexity_metric(perimeters)
        
    return cm

#------------------------------------------------------------------------------

def count_parts(geometry):
    
    # If the geometry is "simple", it's just 1 part
    if geometry.geom_type in ['Point', 'LineString', 'Polygon']:
        nParts = 1
        
    # For "multi" geometries, count the number of parts
    elif geometry.geom_type in ['MultiPoint', 'MultiLineString', 'MultiPolygon']:
        nParts = len(geometry.geoms)

    return nParts

#------------------------------------------------------------------------------

def str_to_list(df, column, suffix = '', data_type = None):
    
    df[column + suffix] = df[column].apply(lambda x: x.split(";") if isinstance(x, str) == True else [])
    
    if data_type == "int":
        df[column + suffix] = df[column].apply(lambda x: list(map(int, x)) if x else [])
        
    elif data_type == "float":
        df[column + suffix] = df[column].apply(lambda x: list(map(float, x)) if x else [])
        
    return df

#------------------------------------------------------------------------------

def std_deviation_change_perimeter(perimeters):
    
    # Initialize an empty list to store relative changes
    relative_changes = []
    
    # Calculate relative changes, avoiding division by zero
    for i in range(1, len(perimeters)):
        if perimeters[i-1] != 0:
            change = abs((perimeters[i] - perimeters[i-1]) / perimeters[i-1])
            relative_changes.append(change)
    
    # Convert list to numpy array for statistical calculation
    changes_array = np.array(relative_changes)
    
    # Use numpy to calculate standard deviation safely, handling empty array case
    if changes_array.size > 0:
        return np.nanstd(changes_array)  # np.nanstd ignores NaN values automatically
    else:
        return 0
    
#------------------------------------------------------------------------------

def cumulative_change_perimeter(perimeters):
    cumulative_change = 0
    step_counts = 0  # Count the number of valid shrinking steps

    for i in range(1, len(perimeters)):
        if perimeters[i-1] > 0:  # Ensure no division by zero
            change = abs((perimeters[i] - perimeters[i-1]) / perimeters[i-1])
            cumulative_change += change
            step_counts += 1

    # Normalize cumulative change by the number of steps actually taken
    normalized_cumulative_change = cumulative_change / step_counts if step_counts > 0 else 0
    return normalized_cumulative_change

#------------------------------------------------------------------------------

def max_relative_change_perimeter(perimeters):
    max_change = 0  # Initialize max_change

    for i in range(1, len(perimeters)):
        if perimeters[i-1] != 0:
            change = abs((perimeters[i] - perimeters[i-1]) / perimeters[i-1])
            if change > max_change:
                max_change = change

    return max_change

#------------------------------------------------------------------------------

def calculate_adjusted_complexity_metric(perimeters):
    if len(perimeters) < 2:
        return 0  # Not enough data to compute changes

    # Calculate percentage changes between consecutive perimeters
    percentage_changes = [(perimeters[i] - perimeters[i - 1]) / perimeters[i - 1] if perimeters[i - 1] != 0 else 0
                          for i in range(1, len(perimeters))]

    # Normalize changes: Logarithmic transformation
    normalized_changes = [np.log1p(abs(change)) for change in percentage_changes if change != 0]

    # Calculate the coefficient of variation (Standard Deviation / Mean)
    if normalized_changes:
        mean_change = np.mean(normalized_changes)
        std_dev_change = np.std(normalized_changes)
        complexity_metric = std_dev_change / mean_change if mean_change != 0 else 0
    else:
        complexity_metric = 0

    return complexity_metric

#------------------------------------------------------------------------------

def calculate_rectangularity(polygon):
    area_polygon = polygon.area
    min_rect = polygon.minimum_rotated_rectangle
    area_min_rect = min_rect.area
    if area_min_rect == 0:  # Avoid division by zero
        return 0
    rectangularity = area_polygon / area_min_rect
    return rectangularity

#------------------------------------------------------------------------------

def aspect_ratio(polygon):
    if polygon.is_empty:
        return 0
    min_rect = polygon.minimum_rotated_rectangle
    # Rotated rectangle's coordinates
    coords = list(min_rect.exterior.coords)
    edge1 = np.linalg.norm(np.array(coords[0]) - np.array(coords[1]))
    edge2 = np.linalg.norm(np.array(coords[1]) - np.array(coords[2]))
    # Ensure width is the smaller edge
    width = min(edge1, edge2)
    height = max(edge1, edge2)
    if height == 0:  # Avoid division by zero
        return 0
    return width / height

#------------------------------------------------------------------------------

def count_holes(geom):
    if geom.geom_type == 'Polygon':
        return len(geom.interiors)
    elif geom.geom_type == 'MultiPolygon':
        return sum(len(polygon.interiors) for polygon in geom.geoms)
    else:
        raise ValueError("The geometry must be a Polygon or MultiPolygon.")
                    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Input folders for the cadastral datasets. Environment variables take priority;
# otherwise, the script expects a repository-friendly ./data layout.
wd_2022 = environ.get(
    "CADASTRE_2022_DIR",
    path.join("data", "cadastre_2022"),
)
wd_2024 = environ.get(
    "CADASTRE_2024_DIR",
    path.join("data", "cadastre_2024"),
)

# Output folder for classification results and calculated metrics.
wd_resultados = environ.get(
    "CADASTRAL_RESULTS_DIR",
    path.join("results", "transformation_processes_metrics"),
)
makedirs(wd_resultados, exist_ok=True) 

#------------------------------------------------------------------------------

chdir(wd_2022)
l_dirs_cat2022 = glob('*/*/*c_custom_noZV.shp')

chdir(wd_2024)
l_dirs_cat2024 = glob('*/*/*c_custom_noZV.shp')  

df_muns = read_csv(path.join(wd_2024,'df_muns.csv'), encoding='latin-1')

#------------------------------------------------------------------------------

lista_analisis_muns = []

chdir(wd_resultados)
muns_done = glob('*.csv')
mun_names_done = []

for m in muns_done:
    mun_names_done.append(m[:-25].upper())

for index, row in df_muns.iterrows():
    mun_name_upper = row.nombre.upper()
    tipo = row.tipo

    if (mun_name_upper not in mun_names_done) and (tipo == "1_prov"):
        lista_analisis_muns.append(mun_name_upper)

#------------------------------------------------------------------------------

d_values = {'ruta_shp_2022': None,
            'ruta_shp_2024': None}
            
d_analisis = dict.fromkeys(lista_analisis_muns)

for key in d_analisis.keys():
    d_analisis[key] = deepcopy(d_values)

for key in d_analisis.keys():
    
    for d in l_dirs_cat2022:
        split = d.split("\\")
        if key.upper() in split[1].upper():
            d_analisis[key]['ruta_shp_2022'] = deepcopy(path.join(wd_2022, d))

    for d in l_dirs_cat2024:
        split = d.split("\\")
        if key.upper() in split[1].upper():
            d_analisis[key]['ruta_shp_2024'] = deepcopy(path.join(wd_2024, d))

#------------------------------------------------------------------------------

for mun in d_analisis.keys():
    
    #mun = "BARCELONA"
    
    # crear las rutas si no existen
    wd_result_mun = path.join(wd_resultados, mun)
    makedirs(wd_result_mun, exist_ok=True)

    print("Trabajando con el municipio: ", mun)
    
    route_2022 = d_analisis[mun]['ruta_shp_2022']
    route_2024 = d_analisis[mun]['ruta_shp_2024']
    
    if route_2022 and route_2024:
                        
        gdf_2022 = read_file(route_2022)
        gdf_crs = deepcopy(gdf_2022.crs)
        
        '''
        # Exploration of the previous duplicate fixing
        mask_with_underscore = gdf_2022['REFCAT'].str.contains('_') # mask with those that have an underscore    
        gdf_underscores = gdf_2022[mask_with_underscore]
        gdf_underscores_svspt = deepcopy(gdf_underscores)
        '''
        
        # get a column with original refcats without duplicates fixiing
        gdf_2022['REFCAT_UNDUP'] = gdf_2022['REFCAT'].str.split('_').str[0] # generate new column with the prefixes

        # Identify duplicate REFCAT values (including the first occurrence)
        dup_mask_2022 = gdf_2022.duplicated('REFCAT_UNDUP', keep=False)
        
        # Only apply the renaming to duplicate rows
        gdf_duplicates_2022 = gdf_2022[dup_mask_2022]
        
        # update the new refcat column with the counts of each dup occurence
        # and also add _2022 to it
        gdf_2022.loc[dup_mask_2022, 'REFCAT'] = gdf_duplicates_2022.groupby('REFCAT_UNDUP').cumcount().add(1).astype(str).radd(gdf_duplicates_2022['REFCAT_UNDUP'].add("_"))
        
        list_dup_refcats_2022 = list(gdf_2022[dup_mask_2022]["REFCAT"])
        #list_dup_refcats_2022_nonLabeled = [x[:-5] for x in list_dup_refcats_2022]
        
        gdf_2022 = gdf_2022[["REFCAT","AREA", "USO", "geometry"]]
        gdf_2022["ATTR_CL"] = None
        gdf_2022["SPA_CL"] = None
        gdf_2022["GROUP_CL"] = None
        gdf_2022["YEAR"] = '2022'
        gdf_2022["AREA"] = gdf_2022.geometry.area #fix possible area issues
        
        #----------------------------------------------------------------------
        
        gdf_2024 = read_file(route_2024)
        gdf_2024 = gdf_2024[gdf_2024["TIPO"] == "U"]
    
        # get a column with original refcats without duplicates fixiing
        gdf_2024['REFCAT_UNDUP'] = gdf_2024['REFCAT'].str.split('_').str[0] # generate new column with the prefixes
        
        # Identify duplicate REFCAT values (including the first occurrence)
        dup_mask_2024 = gdf_2024.duplicated('REFCAT_UNDUP', keep=False)
        
        # Only apply the renaming to duplicate rows
        gdf_duplicates_2024 = gdf_2024[dup_mask_2024]
        
        # update the new refcat column with the counts of each dup occurence
        # and also add _2022 to it
        gdf_2024.loc[dup_mask_2024, 'REFCAT'] = gdf_duplicates_2024.groupby('REFCAT_UNDUP').cumcount().add(1).astype(str).radd(gdf_duplicates_2024['REFCAT_UNDUP'].add("_"))
        
        list_dup_refcats_2024 = list(gdf_2024[dup_mask_2024]["REFCAT"])
        #list_dup_refcats_2024_nonLabeled = [x[:-5] for x in list_dup_refcats_2024]
        
        gdf_2024 = gdf_2024[["REFCAT","AREA", "USO", "geometry"]]
        gdf_2024["ATTR_CL"] = None
        gdf_2024["SPA_CL"] = None
        gdf_2024["GROUP_CL"] = None
        gdf_2024["YEAR"] = '2024'
        gdf_2024["AREA"] = gdf_2024.geometry.area #fix possible area issues
        
        #----------------------------------------------------------------------
        # dups management 
        
        # Convert lists to sets
        set_dups_2022 = set(list_dup_refcats_2022)
        set_dups_2024 = set(list_dup_refcats_2024)
        
        # common dups between both years
        ls_dups_intersection = list(set_dups_2022 & set_dups_2024)
        
        # dups that only are on 2022
        ls_dups_diff_2022 = list(set_dups_2022 - set_dups_2024)
        
        # dups that only are on 2024
        ls_dups_diff_2024 = list(set_dups_2024 - set_dups_2022)

        for common_dup in ls_dups_intersection:
            gdf_2022_row = gdf_2022[gdf_2022["REFCAT"] == common_dup]
            gdf_2024_row = gdf_2024[gdf_2024["REFCAT"] == common_dup]
            
            geom_2022 = gdf_2022_row.geometry.values[0]
            geom_2024 = gdf_2024_row.geometry.values[0]
            
            if not geom_2022.equals(geom_2024):
                ls_dups_diff_2022.append(common_dup)
                ls_dups_diff_2024.append(common_dup)
                
        mask_dups_labeling_2022 = gdf_2022['REFCAT'].isin(ls_dups_diff_2022)
        mask_dups_labeling_2024 = gdf_2024['REFCAT'].isin(ls_dups_diff_2024)

        #gdf_duplicates_2022 = gdf_2022[mask_dups_labeling_2022] # get gdf with unique dups
        #gdf_duplicates_2024 = gdf_2024[mask_dups_labeling_2024] # get gdf with unique dups
        
        gdf_2022.loc[mask_dups_labeling_2022, "REFCAT"] = gdf_2022["REFCAT"] + "_2022"
        gdf_2024.loc[mask_dups_labeling_2024, "REFCAT"] = gdf_2024["REFCAT"] + "_2024"

        #----------------------------------------------------------------------

        list_refcats_2022 = list(gdf_2022["REFCAT"])
        #set_refcats_2022 = set(gdf_2022["REFCAT"])
 
        # create a list with all 2024 cadastral references
        list_refcats_2024 = list(gdf_2024["REFCAT"])
        #set_refcats_2024 = set(gdf_2024["REFCAT"])
        
        # create a list for each type of cadastral reference changes that can 
        # happen:
            
            # lost refcats: when they exist in 2022 but not in 2024
            # new refcats: when they exist in 2024 but not in 2022
            # static refcats: when in both years, 2022 and 2024, the same 
            #                 cadstral reference exist
            # static with different area (>1%) and different use
            # static with different area (>1%) and same use
            # static with same area and different use
            # static with same area and same use
            
        list_refcats_lost = []
        list_refcats_new = []
        list_refcats_static = []
        
        list_refcats_static_CHANGED_USE_CHANGED_AREA = []
        list_refcats_static_CHANGED_USE_UNCHANGED_AREA = []

        list_refcats_static_UNCHANGED_USE_CHANGED_AREA = []
        list_refcats_static_UNCHANGED_USE_UNCHANGED_AREA = []
        
        list_refcats_similar_area = []
        
        # iterate over the refcats of 2022
        for rf22 in list_refcats_2022:
            
            # if it is not a duplicated one
            #if rf22[14:15] != "_":

            # if it doesnt exist in the list of 2024 cadastral references
            # append to corresponding list
            if rf22 not in list_refcats_2024:
                list_refcats_lost.append(rf22)
            # if it exist append to existing list   
            else:
                list_refcats_static.append(rf22)
        
        # iterate over the refcats of 2024
        for rf24 in list_refcats_2024:
            
            # if it doesnt exit in 2022, append to list with new ones
            if rf24 not in list_refcats_2022:
                list_refcats_new.append(rf24)        
        
        #----------------------------------------------------------------------
        # iterate over the static cadastral references
        for rfs in list_refcats_static:
            
            '''#### TEST
            refcat_test = '5413102XM7151E'
            
            # get the rows of each year with that reference
            gdf_2022_refcat = gdf_2022[gdf_2022["REFCAT"] == refcat_test]
            gdf_2024_refcat = gdf_2024[gdf_2024["REFCAT"] == refcat_test]
            
            #### TEST'''
            
            # get the rows of each year with that reference
            gdf_2022_refcat = gdf_2022[gdf_2022["REFCAT"] == rfs]
            gdf_2024_refcat = gdf_2024[gdf_2024["REFCAT"] == rfs]
            
            # get the uses
            uso_2022 = str(gdf_2022_refcat["USO"].values[0])
            uso_2024 = str(gdf_2024_refcat["USO"].values[0])
            
            # calculate precise area values with their geometries
            geom_2022 = gdf_2022_refcat.geometry.values[0]
            geom_2024 = gdf_2024_refcat.geometry.values[0]
            
            # count the number of holes of the geometry of each year
            holes_2022 = count_holes(geom_2022)
            holes_2024 = count_holes(geom_2024)

            area_2022 = geom_2022.area
            area_2024 = geom_2024.area
            
            # calculate the ratio 
            r_area_22_24 = area_2022/area_2024
            
            # threshold to consider a change in area of a parcel with same CR
            similarity_percentage = 0.01

            if (r_area_22_24 > (1 - similarity_percentage)) and\
               (r_area_22_24 < (1 + similarity_percentage)) and\
               (holes_2022 == holes_2024):
                list_refcats_similar_area.append(rfs)
                        
            # fill the lists with each type of change: if the ratio is above 1%
            # that parcel with static refcat is labeled as DIF_AREA, and if the 
            # use is different is labeled as DIF_USE
            if (uso_2022 != uso_2024) and (area_2022 != area_2024):
                list_refcats_static_CHANGED_USE_CHANGED_AREA.append(rfs)
                
            elif (uso_2022 != uso_2024) and (area_2022 == area_2024):
                list_refcats_static_CHANGED_USE_UNCHANGED_AREA.append(rfs)
            
            elif (uso_2022 == uso_2024) and (area_2022 != area_2024):
                list_refcats_static_UNCHANGED_USE_CHANGED_AREA.append(rfs)
                
            elif (uso_2022 == uso_2024) and (area_2022 == area_2024):
                list_refcats_static_UNCHANGED_USE_UNCHANGED_AREA.append(rfs)

        # lost refs (only present in 2022)
        gdf_2022.loc[gdf_2022["REFCAT"].isin(list_refcats_lost), "ATTR_CL"] = 'EXT'
        
        # new refs (only present in 2024)
        gdf_2024.loc[gdf_2024["REFCAT"].isin(list_refcats_new), "ATTR_CL"] = 'NEW'
        
        # parcels that did not suffer any change or it is below threshold (1%)
        gdf_2022.loc[gdf_2022["REFCAT"].isin(list_refcats_static_UNCHANGED_USE_UNCHANGED_AREA), "ATTR_CL"] = 'UU_UA'
        gdf_2024.loc[gdf_2024["REFCAT"].isin(list_refcats_static_UNCHANGED_USE_UNCHANGED_AREA), "ATTR_CL"] = 'UU_UA'
        
        # refs that has different use but the area change is below threshold (1%)
        gdf_2022.loc[gdf_2022["REFCAT"].isin(list_refcats_static_CHANGED_USE_UNCHANGED_AREA), "ATTR_CL"] = 'CU_UA'
        gdf_2024.loc[gdf_2024["REFCAT"].isin(list_refcats_static_CHANGED_USE_UNCHANGED_AREA), "ATTR_CL"] = 'CU_UA'

        # parcels that suffered both, use and geometry change
        gdf_2022.loc[gdf_2022["REFCAT"].isin((list_refcats_static_CHANGED_USE_CHANGED_AREA)) &\
                     ~(gdf_2022["REFCAT"].isin(list_refcats_similar_area)), "ATTR_CL"] = 'CU_CA'
        gdf_2024.loc[gdf_2024["REFCAT"].isin((list_refcats_static_CHANGED_USE_CHANGED_AREA)) &\
                     ~(gdf_2024["REFCAT"].isin(list_refcats_similar_area)), "ATTR_CL"] = 'CU_CA'
        
        # refs that has different area but same use
        gdf_2022.loc[gdf_2022["REFCAT"].isin((list_refcats_static_UNCHANGED_USE_CHANGED_AREA)) &\
                     ~(gdf_2022["REFCAT"].isin(list_refcats_similar_area)), "ATTR_CL"] = 'UU_CA'
        gdf_2024.loc[gdf_2024["REFCAT"].isin((list_refcats_static_UNCHANGED_USE_CHANGED_AREA)) &\
                     ~(gdf_2024["REFCAT"].isin(list_refcats_similar_area)), "ATTR_CL"] = 'UU_CA'
        
        # parcels that suffered both, use and geometry change
        gdf_2022.loc[gdf_2022["REFCAT"].isin((list_refcats_static_CHANGED_USE_CHANGED_AREA)) &\
                     (gdf_2022["REFCAT"].isin(list_refcats_similar_area)), "ATTR_CL"] = 'Negligible_area_change'
        gdf_2024.loc[gdf_2024["REFCAT"].isin((list_refcats_static_CHANGED_USE_CHANGED_AREA)) &\
                     (gdf_2024["REFCAT"].isin(list_refcats_similar_area)), "ATTR_CL"] = 'Negligible_area_change'
        
        # refs that has different area but same use
        gdf_2022.loc[gdf_2022["REFCAT"].isin((list_refcats_static_UNCHANGED_USE_CHANGED_AREA)) &\
                     (gdf_2022["REFCAT"].isin(list_refcats_similar_area)), "ATTR_CL"] = 'Negligible_area_change'
        gdf_2024.loc[gdf_2024["REFCAT"].isin((list_refcats_static_UNCHANGED_USE_CHANGED_AREA)) &\
                     (gdf_2024["REFCAT"].isin(list_refcats_similar_area)), "ATTR_CL"] = 'Negligible_area_change'
        
        # generate a GDF for each change type and for each year 
        # filter with the static refs that has different area but same use
        gdf_STREF_UU_CA_2024 = gdf_2024[(gdf_2024["ATTR_CL"] == "UU_CA")]
        gdf_STREF_UU_CA_2022 = gdf_2022[(gdf_2022["ATTR_CL"] == "UU_CA")]

        # filter with the static refs that has different area and different use
        gdf_STREF_CU_CA_2024 = gdf_2024[(gdf_2024["ATTR_CL"] == "CU_CA")]
        gdf_STREF_CU_CA_2022 = gdf_2022[(gdf_2022["ATTR_CL"] == "CU_CA")]

        # filter with new refs (only 2024)
        gdf_new_refcats = gdf_2024[gdf_2024["ATTR_CL"] == "NEW"]
        
        # filter with lost refs (only 2022)
        gdf_lost_refcats = gdf_2022[gdf_2022["ATTR_CL"] == "EXT"]        

        #gdf_STREF_UU_CA_2024.to_file(path.join(wd_result_mun, (mun +'_gdf_STREF_UU_CA_2024.shp')))
        #gdf_STREF_UU_CA_2022.to_file(path.join(wd_result_mun, (mun +'_gdf_STREF_UU_CA_2022.shp')))
                
        #gdf_STREF_CU_CA_2024.to_file(path.join(wd_result_mun, (mun +'_gdf_STREF_CU_CA_2024.shp')))
        #gdf_STREF_CU_CA_2022.to_file(path.join(wd_result_mun, (mun +'_gdf_STREF_CU_CA_2022.shp')))
        
        #gdf_new_refcats.to_file(path.join(wd_result_mun, (mun +'_gdf_new_refcats.shp')))
        #gdf_lost_refcats.to_file(path.join(wd_result_mun, (mun +'_gdf_lost_refcats.shp')))
    
        ########################## SPATIAL ANALYSIS ###########################
        
        # get all the parcels that suffered any GEOMETRY change between both years
        gdf_all_changes_24 = concat([gdf_STREF_UU_CA_2024, gdf_STREF_CU_CA_2024, gdf_new_refcats], ignore_index=True)
        gdf_all_changes_22 = concat([gdf_STREF_UU_CA_2022, gdf_STREF_CU_CA_2022, gdf_lost_refcats], ignore_index=True)
        
        # get the overlay between transformed parcels
        overlay_22_24 = overlay(gdf_all_changes_22, gdf_all_changes_24,
                                how="intersection", keep_geom_type=True)
        
        # Find REFCATs in 2022 that do not intersect with any parcel in 2024
        intersecting_refcats_22 = overlay_22_24['REFCAT_1'].unique()
        refcat_22_not_in_24 = list(gdf_all_changes_22[~gdf_all_changes_22['REFCAT'].isin(intersecting_refcats_22)]["REFCAT"])
        gdf_2022.loc[gdf_2022["REFCAT"].isin(refcat_22_not_in_24), "SPA_CL"] = 'Not_intersected_2022'
        gdf_2022.loc[gdf_2022["REFCAT"].isin(refcat_22_not_in_24), "GROUP_CL"] = 'Not_intersected_2022'

        # Find REFCATs in 2024 that do not intersect with any parcel in 2022
        intersecting_refcats_24 = overlay_22_24['REFCAT_2'].unique()
        refcat_24_not_in_22 = list(gdf_all_changes_24[~gdf_all_changes_24['REFCAT'].isin(intersecting_refcats_24)]["REFCAT"])
        gdf_2024.loc[gdf_2024["REFCAT"].isin(refcat_24_not_in_22), "SPA_CL"] = 'Not_intersected_2024'
        gdf_2024.loc[gdf_2024["REFCAT"].isin(refcat_24_not_in_22), "GROUP_CL"] = 'Not_intersected_2024'
        
        # select variables
        overlay_22_24 = overlay_22_24[["REFCAT_1", "REFCAT_2", "AREA_1",
                                       "AREA_2", "USO_1", "USO_2", "geometry"]]
        
        # differentiate the refcats of year 2022 and 2024 for later use
        overlay_22_24["refcat1ID"] = overlay_22_24["REFCAT_1"].apply(lambda x: x + "_2022")
        overlay_22_24["refcat2ID"] = overlay_22_24["REFCAT_2"].apply(lambda x: x + "_2024")
            
        # calculate the % of overlayed area vs parcel area of both years
        overlay_22_24["R_OL_22"] = overlay_22_24.geometry.area / overlay_22_24["AREA_1"]
        overlay_22_24["R_OL_24"] = overlay_22_24.geometry.area / overlay_22_24["AREA_2"]
        
        #----------------------------------------------------------------------

        # create variables for the filtering and the geoms of overlaying parcels
        #overlay_22_24["TYPE"] = "Negligible_intersection"
        overlay_22_24["GEOM_22"] = None
        overlay_22_24["GEOM_24"] = None
        
        # avoid use any overlay that is not above 10% of both year parcels
        # the condition filters the overlays where BOTH parcels have less than
        # 10% in common. When small parcel overlays with a very big one, it is
        # included
        
        minimum_percentage_overlay = 0.10
        
        overlay_22_24.loc[((overlay_22_24["R_OL_22"] <= minimum_percentage_overlay) & (overlay_22_24["R_OL_24"] <= minimum_percentage_overlay)), "TYPE"] = "Negligible_intersection"
        overlay_22_24_filtered = overlay_22_24[~((overlay_22_24["R_OL_22"] <= minimum_percentage_overlay) & (overlay_22_24["R_OL_24"] <= minimum_percentage_overlay))]
        #overlay_22_24_filtered = overlay_22_24_filtered[~((overlay_22_24_filtered["R_OL_22"] >= 0.90) & (overlay_22_24_filtered["R_OL_24"] >= 0.90) & (overlay_22_24_filtered["R_22_24"] >= 0.99) & (overlay_22_24_filtered["R_22_24"] <= 1.01))]
        #overlay_22_24_filtered = overlay_22_24_filtered[~((overlay_22_24_filtered["REFCAT_1"] == overlay_22_24_filtered["REFCAT_2"]) & (overlay_22_24_filtered["R_22_24"] >= 0.99) & (overlay_22_24_filtered["R_22_24"] <= 1.01))]
        #overlay_22_24_filtered = overlay_22_24_filtered[((overlay_22_24_filtered["R_22_24"] <= 0.95) | (overlay_22_24_filtered["R_22_24"] >= 1.05))]
        #overlay_22_24_filtered_v1 = overlay_22_24_filtered[~((overlay_22_24_filtered["R_OL_22"] >= 0.99) & (overlay_22_24_filtered["R_OL_24"] >= 0.99))]
        #overlay_22_24_filtered_v2 = overlay_22_24_filtered[((overlay_22_24_filtered["R_22_24"] <= 0.99) | (overlay_22_24_filtered["R_22_24"] >= 1.01))]

        # iterate over each element of the overlay gdf. Each element has a new
        # geometry with the overlayed shape and the references to identifier and
        # variables of the parcels of both years
        for index, row in overlay_22_24_filtered.iterrows():
            
            # Get identifiers and variables for the intersecting parcels
            refcat_2022 = row['REFCAT_1']
            refcat_2024 = row['REFCAT_2']
            
            area_2022 = row["AREA_1"]
            area_2024 = row["AREA_2"]
            
            uso_2022 = row["USO_1"]
            uso_2024 = row["USO_2"]
                        
            # Count intersections for this parcel from Set A
            count_2022 = overlay_22_24_filtered[overlay_22_24_filtered['REFCAT_1'] == refcat_2022].shape[0]
            
            # Count intersections for this parcel from Set B
            count_2024 = overlay_22_24_filtered[overlay_22_24_filtered['REFCAT_2'] == refcat_2024].shape[0]
            
            # get the original geometry of 2022 intersected parcel
            geom_2022 = gdf_all_changes_22[gdf_all_changes_22["REFCAT"] == refcat_2022].geometry.values[0]
            
            # get the original geometry of 2024 intersected parcel
            geom_2024 = gdf_all_changes_24[gdf_all_changes_24["REFCAT"] == refcat_2024].geometry.values[0]
            
            # add original geometries
            overlay_22_24.loc[index, "GEOM_22"] = geom_2022.wkt
            overlay_22_24.loc[index, "GEOM_24"] = geom_2024.wkt
            
            # variable for transformation type
            transformation_type = None
            
            # Parcel Resized
            if (count_2022 == 1) and (count_2024 == 1):
                
                r_area_22_24 = geom_2022.area/geom_2024.area
                                
                if (r_area_22_24 > (1 - similarity_percentage)) and\
                   (r_area_22_24 < (1 + similarity_percentage)):
                    transformation_type = "Negligible_area_change"
                    
                else:
                    transformation_type = "Reshape"
                    
            # Parcel Subdivided    
            elif (count_2024 == 1 and count_2022 > 1):
                transformation_type = "Subdivision"

            # Parcel Aggregated   
            elif (count_2022 == 1 and count_2024 > 1):
                transformation_type ="Aggregation"

            # Not clear what type of transformation/spatial relation
            else:
                transformation_type = "Reconfiguration"

            overlay_22_24.loc[index, "TYPE"] = transformation_type
        
        # first addition of the transformation type using exclusively the 
        # individual spatial relations
        set_individual_class = set(overlay_22_24["TYPE"])
        for c in set_individual_class:
            ol_c = overlay_22_24[overlay_22_24["TYPE"] == c]
            refcats_2022 = list(ol_c["REFCAT_1"])
            refcats_2024 = list(ol_c["REFCAT_2"])
            gdf_2022.loc[gdf_2022["REFCAT"].isin(refcats_2022), "SPA_CL"] = c
            gdf_2024.loc[gdf_2024["REFCAT"].isin(refcats_2024), "SPA_CL"] = c
        
        overlay_22_24_filtered = overlay_22_24[~(overlay_22_24["TYPE"].isin(["Negligible_intersection",
                                                                             "Negligible_area_change"]))]
        
        #----------------------------------------------------------------------
        
        # create graph object and fill it with each OL relation and its type
        G = nx.Graph()
        for index, row in overlay_22_24_filtered.iterrows():
            G.add_edge(row['refcat1ID'], row['refcat2ID'], label=row['TYPE'])
        
        # get connected components
        connected_components = list(nx.connected_components(G))
        
        # dictionary to save the labels of each connection
        label_mapping = {}
        ID_mapping = {}
        original_n_parcels_mapping = {}
        destined_n_parcels_mapping = {}
        
        # create a variable to identify each transformation group inside
        # the components
        transformation_ID = 0
        
        # analyze and assing labels
        for component in connected_components:
            label_counts = {}
            years_parcel = [x[-4:] for x in component]
            n_2022 = years_parcel.count("2022")
            n_2024 = years_parcel.count("2024")
            transformation_ID += 1
            
            for node in component:
                for _, _, data in G.edges(node, data=True):
                    label = data['label']
                    
                    if label in label_counts:
                        label_counts[label] += 1
                        
                    else:
                        label_counts[label] = 1
        
            # Check the number of unique labels
            unique_labels = len(label_counts)
            
            if unique_labels > 1:
                
                if n_2024 < n_2022:
                    group_label = "Reconfiguration_towards_aggregation"
                    
                elif n_2024 > n_2022:
                    group_label = "Reconfiguration_towards_subdivision"
                    
                else:
                    group_label = "Reconfiguration"
            else:
                group_label = max(label_counts, key=label_counts.get)
                
            # Assign this prevalent label to all nodes in the component
            for node in component:
                label_mapping[node] = group_label
                ID_mapping[node] = transformation_ID
                original_n_parcels_mapping[node] = n_2022
                destined_n_parcels_mapping[node] = n_2024
                
        #----------------------------------------------------------------------
        def update_label(row):
            
            # Update label for REFCAT_1
            if row['refcat1ID'] in label_mapping:
                return label_mapping[row['refcat1ID']]
            
            # If REFCAT_1 is not in the mapping, it means it didn't change, 
            # so keep the original label
            return row['TYPE']
        #----------------------------------------------------------------------
        def update_ID(row):
            
            # Update label for REFCAT_1
            if row['refcat1ID'] in ID_mapping:
                return ID_mapping[row['refcat1ID']]
        #----------------------------------------------------------------------
        def update_original_n_parcels(row):

            # Update label for REFCAT_1
            if row['refcat1ID'] in original_n_parcels_mapping:
                return original_n_parcels_mapping[row['refcat1ID']]
        #----------------------------------------------------------------------
        def update_destined_n_parcels(row):

            # Update label for REFCAT_1
            if row['refcat1ID'] in destined_n_parcels_mapping:
                return destined_n_parcels_mapping[row['refcat1ID']]
        #----------------------------------------------------------------------
        
        # Apply the function to update labels
        overlay_22_24_filtered['GROUP_TYPE'] = overlay_22_24_filtered.apply(update_label, axis=1)
        overlay_22_24_filtered['TR_ID'] = overlay_22_24_filtered.apply(update_ID, axis=1)
        overlay_22_24_filtered['O_N_PARCEL'] = overlay_22_24_filtered.apply(update_original_n_parcels, axis=1)
        overlay_22_24_filtered['D_N_PARCEL'] = overlay_22_24_filtered.apply(update_destined_n_parcels, axis=1)
        
        #----------------------------------------------------------------------
        # NODES AND EDGES GENERATION FOR NETWORK VISUALIZATION
        
        # Generate centroids for each parcel
        nodes_2022 = overlay_22_24_filtered[["refcat1ID", "refcat2ID", 'GEOM_22', 'TR_ID']].copy()
        nodes_2024 = overlay_22_24_filtered[["refcat1ID", "refcat2ID", 'GEOM_24', 'TR_ID']].copy()        

        nodes_2022 = nodes_2022.rename(columns={'GEOM_22': 'geometry'})
        nodes_2024 = nodes_2024.rename(columns={'GEOM_24': 'geometry'})
        
        nodes_2022['geometry'] = nodes_2022["geometry"].apply(wkt.loads)
        nodes_2024['geometry'] = nodes_2024["geometry"].apply(wkt.loads)
        
        nodes_2022 = GeoDataFrame(nodes_2022, geometry='geometry', crs=gdf_crs)
        nodes_2024 = GeoDataFrame(nodes_2024, geometry='geometry', crs=gdf_crs)

        #nodes_2022['geometry'] = nodes_2022.geometry.representative_point()
        nodes_2022['geometry'] = nodes_2022.geometry.apply(lambda geom: geom.representative_point())

        #nodes_2024['geometry'] = nodes_2024.geometry.representative_point()
        nodes_2024['geometry'] = nodes_2024.geometry.apply(lambda geom: geom.representative_point())

        # Create GeoDataFrame for edges
        edges_list = []
        for u, v in G.edges():
            
            u_refcat = u[-4:]
            v_refcat = v[-4:]
            
            if u_refcat == "2024":
                source_ID = v
                source_geom = nodes_2024.loc[nodes_2024['refcat2ID'] == u, 'geometry'].values[0]
                target_geom = nodes_2022.loc[nodes_2022['refcat1ID'] == v, 'geometry'].values[0]
                source_geom
                
            elif u_refcat == "2022":
                source_ID = u
                source_geom = nodes_2022.loc[nodes_2022['refcat1ID'] == u, 'geometry'].values[0]
                target_geom = nodes_2024.loc[nodes_2024['refcat2ID'] == v, 'geometry'].values[0]
                
            edges_list.append({'geometry': LineString([source_geom, target_geom]), 'from': u, 'to': v, 'cluster_id': nodes_2022.loc[nodes_2022['refcat1ID'] == source_ID, 'TR_ID'].values[0]})
        
        edges = GeoDataFrame(edges_list, crs=gdf_crs)
        
        # Save nodes and edges to shapefiles
        nodes_2022.to_file(path.join(wd_result_mun, (mun +"_nodes_2022.shp")))
        nodes_2024.to_file(path.join(wd_result_mun, (mun +"_nodes_2024.shp")))
        edges.to_file(path.join(wd_result_mun, (mun +"_edges.shp")))
        
        #----------------------------------------------------------------------
        
        set_individual_class = set(overlay_22_24_filtered["TYPE"])
        for c in set_individual_class:
            ol_c = overlay_22_24_filtered[overlay_22_24_filtered["TYPE"] == c]
            refcats_2022 = list(ol_c["REFCAT_1"])
            refcats_2024 = list(ol_c["REFCAT_2"])
            gdf_2022.loc[gdf_2022["REFCAT"].isin(refcats_2022), "SPA_CL"] = c
            gdf_2024.loc[gdf_2024["REFCAT"].isin(refcats_2024), "SPA_CL"] = c
        
        set_group_class = set(overlay_22_24_filtered["GROUP_TYPE"])
        for c in set_group_class:
            ol_c = overlay_22_24_filtered[overlay_22_24_filtered["GROUP_TYPE"] == c]
            refcats_2022 = list(ol_c["REFCAT_1"])
            refcats_2024 = list(ol_c["REFCAT_2"])
            gdf_2022.loc[gdf_2022["REFCAT"].isin(refcats_2022), "GROUP_CL"] = c
            gdf_2024.loc[gdf_2024["REFCAT"].isin(refcats_2024), "GROUP_CL"] = c

        set_tr_id = set(overlay_22_24_filtered["TR_ID"])
        for c in set_tr_id:
            ol_c = overlay_22_24_filtered[overlay_22_24_filtered["TR_ID"] == c]
            refcats_2022 = list(ol_c["REFCAT_1"])
            refcats_2024 = list(ol_c["REFCAT_2"])
            gdf_2022.loc[gdf_2022["REFCAT"].isin(refcats_2022), "TR_ID"] = c
            gdf_2024.loc[gdf_2024["REFCAT"].isin(refcats_2024), "TR_ID"] = c
            set_group_class = set(overlay_22_24_filtered["TR_ID"])
            
        set_o_n_parcels = set(overlay_22_24_filtered["O_N_PARCEL"])
        for c in set_group_class:
            ol_c = overlay_22_24_filtered[overlay_22_24_filtered["O_N_PARCEL"] == c]
            refcats_2022 = list(ol_c["REFCAT_1"])
            refcats_2024 = list(ol_c["REFCAT_2"])
            gdf_2022.loc[gdf_2022["REFCAT"].isin(refcats_2022), "O_N_PARCEL"] = c
            gdf_2024.loc[gdf_2024["REFCAT"].isin(refcats_2024), "O_N_PARCEL"] = c

        set_d_n_parcels = set(overlay_22_24_filtered["D_N_PARCEL"])
        for c in set_group_class:
            ol_c = overlay_22_24_filtered[overlay_22_24_filtered["D_N_PARCEL"] == c]
            refcats_2022 = list(ol_c["REFCAT_1"])
            refcats_2024 = list(ol_c["REFCAT_2"])
            gdf_2022.loc[gdf_2022["REFCAT"].isin(refcats_2022), "D_N_PARCEL"] = c
            gdf_2024.loc[gdf_2024["REFCAT"].isin(refcats_2024), "D_N_PARCEL"] = c
               
        gdf_2024_any_change = gdf_2024[~(gdf_2024["ATTR_CL"] == "UU_UA")]
        
        gdf_fusion = concat([gdf_2022, gdf_2024_any_change], ignore_index=True)
        
        gdf_fusion["PR_SUB"] = gdf_fusion["D_N_PARCEL"]/gdf_fusion["O_N_PARCEL"]
        gdf_fusion["PR_AGG"] = gdf_fusion["O_N_PARCEL"]/gdf_fusion["D_N_PARCEL"]

        gdf_fusion['CVXH']  = gdf_fusion.apply(lambda row: row['geometry'].convex_hull, axis=1)
        gdf_fusion['CVXH_AREA']  = gdf_fusion["CVXH"].apply(lambda x: x.area)
        gdf_fusion['CVXH_PERIM']  = gdf_fusion["CVXH"].apply(lambda x: x.length)

        gdf_fusion["AREA"] = gdf_fusion["geometry"].apply(lambda x: x.area)
        gdf_fusion["PERIM"] = gdf_fusion["geometry"].apply(lambda x: x.length)
        gdf_fusion["PAR"] = gdf_fusion['PERIM'] / gdf_fusion['AREA']
        gdf_fusion["SHAPE"] = gdf_fusion['PERIM'] / np.sqrt(gdf_fusion['AREA'])
        gdf_fusion["FD"] = (2*(np.log(gdf_fusion["PERIM"]))) / (np.log(gdf_fusion["AREA"]))
        gdf_fusion["SCI"] = 1 - (gdf_fusion["AREA"] / gdf_fusion["CVXH_AREA"])
        gdf_fusion["SII"] = 1 - (gdf_fusion["CVXH_PERIM"] / gdf_fusion["PERIM"])
        gdf_fusion["COMP"] = (4*(np.pi)*gdf_fusion["AREA"]) / (gdf_fusion['PERIM']**2)
        gdf_fusion['RECT'] = gdf_fusion['geometry'].apply(calculate_rectangularity)
        gdf_fusion['AR'] = gdf_fusion['geometry'].apply(aspect_ratio)

        gdf_fusion["SHRINK"] = gdf_fusion["geometry"].apply(lambda x: shrink_polygon(x))

        list_group_vars = ["O_G_AREA", "O_G_PERIM", "O_G_SHAPE", "O_G_SK_NB",
                           "O_G_SK_LR"]
        
        cols_to_drop_no_geoms = ["ORIGINAL_GEOM", "AGGREGATED_GEOM"]
        
        cols_to_drop_list_vars = ["O_G_AREA_GROUP", "O_G_PERIM_GROUP",
                                  "O_G_SHAPE_GROUP",  "O_G_SK_NB_GROUP",
                                  "O_G_SK_LR_GROUP"]
        
        cols_to_drop = ["TIPO", "CCAA", "PROV", "ORIGINAL_GEOMS_L", "CVXH",
                        "ORIGINAL_USES_L", "O_G_TR", "S_G_TR", "SK", "O_G_SK"]
        
        cols_to_drop.extend(cols_to_drop_no_geoms)
        cols_to_drop.extend(cols_to_drop_list_vars)
        
        for field in gdf_fusion.columns:
            if field in cols_to_drop:
                gdf_fusion = gdf_fusion.drop(field, axis = 1)
        
        result_name = mun + "_gdf_fusion_metrics.shp"
        
        gdf_fusion.to_file(path.join(wd_result_mun, result_name))
