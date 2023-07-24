import copy
import numpy as np
import os
import sys
import random
import logging
from typing import Tuple, List
import time
import pandas as pd
from turtle import fd
import gym
import logging
from pathlib import Path
from datetime import datetime
import time
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import flood_fill
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import seaborn as sns
import matplotlib as mpl

EMPTY = 0.0
DEAD = 1.0
INFECTED = 0.3
ALIVE = 3.0

class Threat_basic:
    def __init__(self, w: int, h: int, ow: int, oh: int,
                 grid: np.ndarray, coords_x: List, coords_y: List, infect_list):
        # self.cx, self.cy = cx, cy
        self.w, self.h = w, h
        self.ow, self.oh = ow, oh

        self.grid = grid
        self.infect_day_mat = np.zeros(self.grid.shape)
        self.coords_x = coords_x
        self.coords_y = coords_y
        self.destruction_limit = [7,8,9]
        self.infect_list = infect_list

        logging.info("---------- Threat initiated ----------")
        logging.info(f"Name: {self.__class__.__name__}")
        logging.info(f"Destruction days: {self.destruction_limit}")
        logging.info("---------- Threat log end ----------")

    def compute_dense(self, cx: int, cy: int, mat: np.ndarray, val: float = None):
        self.cx, self.cy = cx, cy
        self.west_edge, self.east_edge = int(cx - self.w/2), int(cx + self.w/2)
        self.north_edge, self.south_edge = int(cy + self.h/2), int(cy - self.h/2)
        if val == None:
            ty, tx = np.where(mat[self.south_edge:self.north_edge+1, self.west_edge:self.east_edge+1] != EMPTY)
        else:
            ty, tx = np.where(mat[self.south_edge:self.north_edge+1, self.west_edge:self.east_edge+1] == val)
        # print(tx, ty)
        ty = ty + self.south_edge
        tx = tx + self.west_edge
        # print(tx, ty)
        if val != None:
            for y, x in zip(ty, tx):
                mat[y,x] = val

        return mat, (tx,ty)

    def compute_hollow(self, cx: int, cy: int, mat: np.ndarray, val: float = None):
        self.cx, self.cy = cx, cy
        # to prevent double counting from the compute_dense(), move the nsew off by 1
        #
        #    *  *  *  *  *
        #       /\
        #       |
        #       _
        #    *  *  *  *  *
        #     <-|     |->
        #    *  *  *  *  *
        #     <-|     |->
        #    *  *  *  *  *
        #       _
        #       |
        #       \/
        #    *  *  *  *  *
        #
        #
        self.west_edge, self.east_edge = int(cx - self.w/2) - 1, int(cx + self.w/2) + 1
        self.north_edge, self.south_edge = int(cy + self.h/2) + 1, int(cy - self.h/2) - 1
        self.o_west_edge, self.o_east_edge = int(cx - self.ow/2), int(cx + self.ow/2)
        self.o_north_edge, self.o_south_edge = int(cy + self.oh/2), int(cy - self.oh/2)

        if val == None:
            ty, tx = np.where(mat[self.north_edge:self.o_north_edge+1, self.west_edge:self.east_edge+1] != EMPTY)
        else:
            ty, tx = np.where(mat[self.north_edge:self.o_north_edge+1, self.west_edge:self.east_edge+1] == val)

        ty1 = ty + self.north_edge
        tx1 = tx + self.west_edge

        if val != None:
            for y, x in zip(ty1, tx1):
                mat[y,x] = val

        if val == None:
            ty, tx = np.where(mat[self.o_south_edge:self.south_edge+1, self.west_edge:self.east_edge+1] != EMPTY)
        else:
            ty, tx = np.where(mat[self.o_south_edge:self.south_edge+1, self.west_edge:self.east_edge+1] == val)

        ty2 = ty + self.o_south_edge
        tx2 = tx + self.west_edge

        if val != None:
            for y, x in zip(ty2, tx2):
                mat[y,x] = val

        if val == None:
            ty, tx = np.where(mat[self.o_south_edge:self.o_north_edge+1, self.o_west_edge:self.west_edge+1] != EMPTY)
        else:
            ty, tx = np.where(mat[self.o_south_edge:self.o_north_edge+1, self.o_west_edge:self.west_edge+1] == val)

        ty3 = ty + self.o_south_edge
        tx3 = tx + self.o_west_edge

        if val != None:
            for y, x in zip(ty3, tx3):
                mat[y,x] = val

        if val == None:
            ty, tx = np.where(mat[self.o_south_edge:self.o_north_edge+1, self.east_edge:self.o_east_edge+1] != EMPTY)
        else:
            ty, tx = np.where(mat[self.o_south_edge:self.o_north_edge+1, self.east_edge:self.o_east_edge+1] == val)

        ty4 = ty + self.o_south_edge
        tx4 = tx + self.east_edge
        if val != None:
            for y, x in zip(ty4, tx4):
                mat[y,x] = val

        tx = np.hstack((tx1, tx2, tx3, tx4))
        ty = np.hstack((ty1, ty2, ty3, ty4))

        return mat, (tx,ty)

    def draw(self, ax, c='k', lw=1, **kwargs):
        try:
            self.west_edge
        except AttributeError:
            return
        x1, y1 = self.west_edge, self.north_edge
        x2, y2 = self.east_edge, self.south_edge

        x3, y3 = self.o_west_edge, self.o_north_edge
        x4, y4 = self.o_east_edge, self.o_south_edge

    def apply_infection_reproductive_stage(self):
        # indx, indy=np.where((self.grid==0.3))
        # for i, j in zip(indx, indy):
        #   self.grid[indx, indy] = INFECTED

        for infect in self.infect_list:
            y, x = infect
            self.grid[y, x] = INFECTED

    def spread_probabilities(self):
        high_infect = np.random.choice([0,1], size=25, replace=True, p=[0.55, 0.45])
        medium_infect = np.random.choice([0,1], size=39, replace=True, p=[0.6, 0.4])
        low_infect = np.random.choice([0,1], size=24, replace=True, p=[0.95, 0.05])
        self.infection = {'high': high_infect, 'medium': medium_infect, 'low': low_infect}
        # print(self.infection)
    def infection_tracker(self):
        temp_list = list(self.infect_list)
        dense_list = [0]
        loop_list = [0]
        apply_list = [0]
        start_loop = time.time()
        idx = 0
        # print(temp_list)
        for k in temp_list:
            x, y = k

    def check_neighbour(self):
        temp_list = list(self.infect_list)

        dense_list = [0]
        dense_loop = [0]
        hollow_list = [0]
        hollow_loop = [0]

        start_loop = time.time()
        for plant_infected in temp_list:
            y, x = plant_infected
            # print(x, y)
            # print(plant_infected)
            if self.grid[y, x] == DEAD:
                continue
            # print(self.grid[x,y])

            start = time.time()
            _, neighbouring_idx = self.compute_dense(x, y, self.grid)
            end = time.time()
            dense_list.append(end - start)
            idx = 0
            # print(neighbouring_idx[0])
            # print(neighbouring_idx[1])
            start = time.time()
            for (i, j) in zip(neighbouring_idx[0], neighbouring_idx[1]):
                # print(i, j)
                if (i, j) == (x, y):
                    continue
                if (j, i) not in self.infect_list:
                    if self.infection['high'][idx] == 1:
                        self.grid[j, i] = INFECTED
                        self.infect_list.append((j,i))

                idx += 1

            end = time.time()
            dense_loop.append(end - start)

            start = time.time()
            _, neighbouring_idx = self.compute_hollow(x, y, self.grid)
            end = time.time()
            hollow_list.append(end - start)
            idx = 0
            start = time.time()
            for i, j in zip(neighbouring_idx[0], neighbouring_idx[1]):
                # print(i, j)
                if (i, j) == (x, y):
                    continue
                if (j, i) not in self.infect_list:
                    if self.infection['medium'][idx] == 1:
                        self.grid[j, i] = INFECTED
                        self.infect_list.append((j,i))
                idx += 1
            # print(self.grid)
            end = time.time()
            hollow_loop.append(end - start)

        end = time.time()
        logging.debug(f"check_neighbour whole loop | Time taken: {end - start_loop}")
        logging.debug(f"check_neighbour Dense | Total {sum(dense_list)}, Average {np.round(np.average(dense_list), 6)}, Max: {np.round(max(dense_list), 6)}, Min: {np.round(min(dense_list), 6)} ")
        logging.debug(f"check_neighbour Loop | Total {sum(dense_loop)}, Average {np.round(np.average(dense_loop), 6)}, Max: {np.round(max(dense_loop), 6)}, Min: {np.round(min(dense_loop), 6)} ")
        logging.debug(f"check_neighbour Hollow | Total {sum(hollow_list)}, Average {np.round(np.average(hollow_list), 6)}, Max: {np.round(max(hollow_list), 6)}, Min: {np.round(min(hollow_list), 6)} ")
        logging.debug(f"check_neighbour Loop | Total {sum(hollow_loop)}, Average {np.round(np.average(hollow_loop), 6)}, Max: {np.round(max(hollow_loop), 6)}, Min: {np.round(min(hollow_loop), 6)} ")

    def compute_infection(self, grid):
        self.grid = grid
        self.apply_infection_reproductive_stage()
        self.spread_probabilities()
        self.check_neighbour()
        return self.grid


