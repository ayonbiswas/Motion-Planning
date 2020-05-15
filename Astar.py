import numpy as np
import time
import matplotlib.pyplot as plt#; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import Planner
from pyrr import *
from pyrr import geometric_tests as gt
import numpy as np
from pqdict import pqdict
from numpy import linalg as LA
import math
import argparse

#collsion detection 
#modified from pyrr
def ray_intersect_aabb(ray, aabb):
    """Calculates the intersection point of a ray and an AABB
    :param numpy.array ray1: The ray to check.
    :param numpy.array aabb: The Axis-Aligned Bounding Box to check against.
    :rtype: numpy.array
    :return: Returns a vector if an intersection occurs.
        Returns None if no intersection occurs.
    """

    direction = ray[1]
    dir_fraction = np.empty(3, dtype = ray.dtype)
    dir_fraction[direction == 0.0] = 1e16
    dir_fraction[direction != 0.0] = np.divide(1.0, direction[direction != 0.0])

    t1 = (aabb[0,0] - ray[0,0]) * dir_fraction[ 0 ]
    t2 = (aabb[1,0] - ray[0,0]) * dir_fraction[ 0 ]
    t3 = (aabb[0,1] - ray[0,1]) * dir_fraction[ 1 ]
    t4 = (aabb[1,1] - ray[0,1]) * dir_fraction[ 1 ]
    t5 = (aabb[0,2] - ray[0,2]) * dir_fraction[ 2 ]
    t6 = (aabb[1,2] - ray[0,2]) * dir_fraction[ 2 ]

    tmin = max(min(t1, t2), min(t3, t4), min(t5, t6))
    tmax = min(max(t1, t2), max(t3, t4), max(t5, t6))
    # if tmax < 0, ray (line) is intersecting AABB
    # but the whole AABB is behind the ray start
    if tmax < 0:
        return None

    # if tmin > tmax, ray doesn't intersect AABB
    if tmin > tmax:
        return None

    # t is the distance from the ray point
    # to intersection

    t = min(x for x in [tmin, tmax] if x >= 0)
    point = ray[0] + (ray[1] * t)
    return tuple(point)

#check if a point lies between a line segment        
def point_intersect_line_segment(point, line):
    l1 = line[0]
    l2 = line[1]
    return (point[0]>l1[0] and point[0]<=l2[0]) or (point[1]>l1[1] and point[1]<=l2[1]) or (point[2]>l1[2] and point[2]<=l2[2]) \
    or (point[0]<l1[0] and point[0]>=l2[0]) or (point[1]<l1[1] and point[1]>=l2[1]) or (point[2]<l1[2] and point[2]>=l2[2])

class Planner:
  __slots__ = ['boundary', 'blocks','d']
  
  def __init__(self, boundary, blocks, d):
    self.boundary = boundary
    self.blocks = blocks
    self.d = d

  #check collision with AABB
  def check_collision(self, start_node,end_node):
    line = np.array([ start_node, end_node ])
    ray_ = ray.create_from_line(line)

    if( end_node[0] < self.boundary[0,0] or end_node[0] > self.boundary[0,3] or \
        end_node[1] < self.boundary[0,1] or end_node[1] > self.boundary[0,4] or \
        end_node[2] < self.boundary[0,2] or end_node[2] > self.boundary[0,5] ):
        return True

    for k in range(self.blocks.shape[0]):
        if( end_node[0] >= self.blocks[k,0] and end_node[0] <= self.blocks[k,3] and\
          end_node[1] >=self.blocks[k,1] and end_node[1] <= self.blocks[k,4] and\
          end_node[2] >= self.blocks[k,2] and end_node[2] <= self.blocks[k,5] ):
            return True
            
    for i in range(self.blocks.shape[0]):
        min_ = self.blocks[i,:3]
        max_ = self.blocks[i,3:6]
        aabbox = aabb.create_from_bounds(min_, max_)
        intersection = ray_intersect_aabb(ray_, aabbox)

        if intersection:
            point_check = point_intersect_line_segment(intersection, line)
            if(point_check):
                return True
    return False
  
  #get children within bounds in all the directions and their cost
  def get_children(self,node,g_i,directions):    

    children = []
    cost = {}
    for j in range(directions.shape[1]):
        node = np.array(node)
        next_node = np.add(node, self.d*directions[:,j])

        if(not self.check_collision(node,next_node)):
            children.append(tuple(next_node))
            cost[tuple(next_node)] = self.d*LA.norm(directions[:,j])

    return children, cost    
    
  #perform A*
  def plan(self,start,goal):

    numofdirs = 26
    alpha = 1.0
    [dX,dY,dZ] = np.meshgrid([-1.0,0.0,1.0],[-1.0,0.0,1.0],[-1.0,0.0,1.0])
    dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
    dR = np.delete(dR,13,axis=1)
        
    self.boundary[:,:3] = self.boundary[:,:3] +0.0005
    self.boundary[:,3:] = self.boundary[:,3:] -0.0005
    start = tuple(start)
    goal = tuple(goal)
    open_set = pqdict({start:0})
    closed_set = set()

    g = {}
    g[start] = 0
    g[goal] = np.inf
    parent = {}

    while(goal not in closed_set):
        i = open_set.pop()        
        closed_set.add(i)
        children, cost = self.get_children(i,g[i],dR)
        for j in children:
            if j not in closed_set:

                if(j not in g or  g[j] > g[i]+cost[j]):
                    
                    g[j] = g[i] + cost[j]
                    heuristic = LA.norm(np.subtract(goal,j))
                    
                    if( heuristic <= self.d*(np.sqrt(3)/2) and g[j] + heuristic < g[goal]):
                        g[goal] = g[j] + heuristic
                        parent[goal] = j 
                        open_set[goal] = g[goal]
                    parent[j] = i
                    open_set[j] = g[j] + heuristic

    nd = goal
    path = [nd]
    while True:
        nd = parent[nd]
        path.append(nd)
        if(nd  == start):
            break
    path = np.array(path)

    return path, g[goal], len(closed_set) + len(open_set)


def tic():
  return time.time()


def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))
  return time.time() - tstart

def load_map(fname):
  '''
  Loads the bounady and blocks from map file fname.
  
  boundary = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
  
  blocks = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'],
            ...,
            ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
  '''
  mapdata = np.loadtxt(fname,dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'),\
                                    'formats': ('S8','f', 'f', 'f', 'f', 'f', 'f', 'f','f','f')})
  blockIdx = mapdata['type'] == b'block'
  boundary = mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  blocks = mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  return boundary, blocks


def draw_map(boundary, blocks, start, goal):
  '''
  Visualization of a planning problem with environment boundary, obstacle blocks, and start and goal points
  '''
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  hb = draw_block_list(ax,blocks)
  hs = ax.plot(start[0:1],start[1:2],start[2:],'ro',markersize=7,markeredgecolor='k')
  hg = ax.plot(goal[0:1],goal[1:2],goal[2:],'go',markersize=7,markeredgecolor='k')  
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim(boundary[0,0],boundary[0,3])
  ax.set_ylim(boundary[0,1],boundary[0,4])
  ax.set_zlim(boundary[0,2],boundary[0,5])  
  return fig, ax, hb, hs, hg

def draw_block_list(ax,blocks):
  '''
  Subroutine used by draw_map() to display the environment blocks
  '''
  v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype='float')
  f = np.array([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[0,1,2,3],[4,5,6,7]])
  clr = blocks[:,6:]/255
  n = blocks.shape[0]
  d = blocks[:,3:6] - blocks[:,:3] 
  vl = np.zeros((8*n,3))
  fl = np.zeros((6*n,4),dtype='int64')
  fcl = np.zeros((6*n,3))
  for k in range(n):
    vl[k*8:(k+1)*8,:] = v * d[k] + blocks[k,:3]
    fl[k*6:(k+1)*6,:] = f + k*8
    fcl[k*6:(k+1)*6,:] = clr[k,:]
  
  if type(ax) is Poly3DCollection:
    ax.set_verts(vl[fl])
  else:
    pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
    pc.set_facecolor(fcl)
    h = ax.add_collection3d(pc)
    return h


def runtest(mapfile, start, goal, d, fname, verbose = True):
  '''
  This function:
   * load the provided mapfile
   * creates a motion planner
   * plans a path from start to goal
   * checks whether the path is collision free and reaches the goal
   * computes the path length as a sum of the Euclidean norm of the path segments
  '''
  # Load a map and instantiate a motion planner
  boundary, blocks = load_map(mapfile)
#   print(blocks)
  MP = Planner(boundary, blocks, d) # TODO: replace this with your own planner implementation
  
  # Display the environment
#   if verbose:
    
  
      # Call the motion planner
  t0 = tic()
  path, cost , nodes_visted= MP.plan(start, goal)
  t2 = toc(t0,"Planning")
  
  # Plot the path
  if verbose:
    fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)  
    ax.plot(path[:,0],path[:,1],path[:,2],'r-')
    plt.show()

  # TODO: You should verify whether the path actually intersects any of the obstacles in continuous space
  # TODO: You can implement your own algorithm or use an existing library for segment and 
  #       axis-aligned bounding box (AABB) intersection 
  collision = True
  goal_reached = sum((path[-1]-goal)**2) <= 0.1
  pathlength = np.sum(np.sqrt(np.sum(np.diff(path,axis=0)**2,axis=1)))
  
  env = mapfile.split(".")[1].split("/")[-1]
  fname = fname + "{}_{}_{}_{}.txt".format(env, d,np.round(pathlength,2),np.round(t2,2))
  if fname:
    np.save(fname,path)
  
  print('Path length: %d'%pathlength)
  print("Total cost of the path : {}".format(cost))
  print("Total visited nodes : {}".format(nodes_visted))
  print('\n')
    
  

def test_single_cube(d,fname,verbose = False):
  print('Running single cube test...\n') 
  start = np.array([2.3, 2.3, 1.3])
  goal = np.array([7.0, 7.0, 5.5])
  runtest('./maps/single_cube.txt', start, goal,d, fname,verbose)

  
  
def test_maze(d,fname,verbose = False):
  print('Running maze test...\n') 
  start = np.array([0.0, 0.0, 1.0])
  goal = np.array([12.0, 12.0, 5.0])
  runtest('./maps/maze.txt', start, goal,d, fname,verbose)
  

    
def test_window(d,fname,verbose = False):
  print('Running window test...\n') 
  start = np.array([0.2, -4.9, 0.2])
  goal = np.array([6.0, 18.0, 3.0])
  runtest('./maps/window.txt', start, goal, d,fname,verbose)
  

  
def test_tower(d,fname,verbose = False):
  print('Running tower test...\n') 
  start = np.array([2.5, 4.0, 0.5])
  goal = np.array([4.0, 2.5, 19.5])
  runtest('./maps/tower.txt', start, goal, d,fname,verbose)


     
def test_flappy_bird(d,fname,verbose = False):
  print('Running flappy bird test...\n') 
  start = np.array([0.5, 2.5, 5.5])
  goal = np.array([19.0, 2.5, 5.5])
  runtest('./maps/flappy_bird.txt', start, goal,d, fname,verbose)


  
def test_room(d,fname,verbose = False):
  print('Running room test...\n') 
  start = np.array([1.0, 5.0, 1.5])
  goal = np.array([9.0, 7.0, 1.5])
  runtest('./maps/room.txt', start, goal,d, fname,verbose)
  


def test_monza(d,fname,verbose = False):
  print('Running monza test...\n')
  start = np.array([0.5, 1.0, 4.9])
  goal = np.array([3.8, 1.0, 0.1])
  runtest('./maps/monza.txt', start, goal,d, fname,verbose)
  



if __name__ == "__main__":
#  # Create an argument parser
	parser = argparse.ArgumentParser(description='Optimal motion planning demo program.')

	# Add a filename argument
	parser.add_argument('-d', '--grid_size', type=float, default=0.5, \
		help= 'set grid size' )
	parser.add_argument('-e', '--env',default='flappy_bird',choices=['cube','maze','flappy_bird','monza','window','tower','room'],\
		help='choose environment')
	parser.add_argument('-f', '--file', default="./astar/path.npy", \
		help='(Optional) Specify an output path for the found solution path.')
	# Parse the arguments
	args = parser.parse_args()

	if(args.env == 'cube'):
		test_single_cube( args.grid_size,args.file, True)
	elif(args.env == 'maze'):
		test_maze( args.grid_size,args.file, True)
	elif(args.env == 'flappy_bird'):
		test_flappy_bird( args.grid_size,args.file, True)
	elif(args.env == 'monza'):
		test_monza( args.grid_size,args.file, True)
	elif(args.env == 'window'):
		test_window( args.grid_size,args.file, True)
	elif(args.env == 'tower'):
		test_tower( args.grid_size,args.file, True)
	elif(args.env == 'room'):
		test_room( args.grid_size,args.file, True)
