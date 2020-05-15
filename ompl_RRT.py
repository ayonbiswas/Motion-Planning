import sys
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
try:
	from ompl import util as ou
	from ompl import base as ob
	from ompl import geometric as og
except ImportError:
 # if the ompl module is not in the PYTHONPATH assume it is installed in a
 # subdirectory of the parent directory called "py-bindings."
	from os.path import abspath, dirname, join
	sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'py-bindings'))
	from ompl import util as ou
	from ompl import base as ob
	from ompl import geometric as og
from math import sqrt
import argparse

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))
  return (time.time() - tstart)
  

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

class ValidityChecker(ob.StateValidityChecker):
	def __init__(self,si, boundary, blocks):
		super(ValidityChecker, self).__init__(si)
		self.boundary = boundary
		self.blocks = blocks
 # Returns whether the given state's position overlaps the
 # circular obstacle
	def isValid(self, state):
		return self.clearance(state)

 # Returns the distance from the given state's position to the
 # boundary of the circular obstacle.
	def clearance(self, state):	
	 # Extract the robot's (x,y,z) position from its state
		x = state[0]
		y = state[1]
		z = state[2]

		if( x < self.boundary[0,0] or x > self.boundary[0,3] or \
			y < self.boundary[0,1] or y > self.boundary[0,4] or \
			z < self.boundary[0,2] or z > self.boundary[0,5] ):
				return False

		for k in range(self.blocks.shape[0]):
			if( x >= self.blocks[k,0] and x <= self.blocks[k,3] and\
			  y >= self.blocks[k,1] and y <= self.blocks[k,4] and\
			  z >= self.blocks[k,2] and z <= self.blocks[k,5] ):
				return False

		return True

class MyMotionValidator(ob.MotionValidator):
	def __init__(self, si, boundary, blocks):
		super(MyMotionValidator, self).__init__(si)
		self.boundary = boundary
		self.blocks = blocks
		self.isValidstate = ValidityChecker(si, boundary, blocks)

	def checkMotion(self, s1, s2):
		
		start_node = np.array([s1[0],s1[1],s1[2]])
		end_node = np.array([s2[0],s2[1],s2[2]])
		line = np.array([ start_node, end_node ])
		ray_ = ray.create_from_line(line)

		if(not self.isValidstate.isValid(s2)):
			return False
		
		for i in range(self.blocks.shape[0]):
			min_ = self.blocks[i,:3]
			max_ = self.blocks[i,3:6]
			aabbox = aabb.create_from_bounds(min_, max_)
			intersection = ray_intersect_aabb(ray_, aabbox)

			if intersection:
				point_check = point_intersect_line_segment(intersection, line)
				if(point_check):
					return False
		return True


def getPathLengthObjective(si):
 return ob.PathLengthOptimizationObjective(si)


def getThresholdPathLengthObj(si):
 obj = ob.PathLengthOptimizationObjective(si)
 obj.setCostThreshold(ob.Cost(1.51))
 return obj


class ClearanceObjective(ob.StateCostIntegralObjective):
 def __init__(self, si):
	 super(ClearanceObjective, self).__init__(si, True)
	 self.si_ = si

 def stateCost(self, s):
	 return ob.Cost(1 / (self.si_.getStateValidityChecker().clearance(s) +
		 sys.float_info.min))


def getClearanceObjective(si):
 return ClearanceObjective(si)


def getBalancedObjective1(si):
 lengthObj = ob.PathLengthOptimizationObjective(si)
 clearObj = ClearanceObjective(si)

 opt = ob.MultiOptimizationObjective(si)
 opt.addObjective(lengthObj, 5.0)
 opt.addObjective(clearObj, 1.0)

 return opt


def getPathLengthObjWithCostToGo(si):
	obj = ob.PathLengthOptimizationObjective(si)
	obj.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))
	return obj


# Keep these in alphabetical order and all lower case
def allocatePlanner(si,d, plannerType):
	if plannerType.lower() == "bfmtstar":
		return og.BFMT(si)
	elif plannerType.lower() == "bitstar":
		return og.BITstar(si)
	elif plannerType.lower() == "fmtstar":
		return og.FMT(si)
	elif plannerType.lower() == "informedrrtstar":
		return og.InformedRRTstar(si)
	elif plannerType.lower() == "prmstar":
		return og.PRMstar(si)
	elif plannerType.lower() == "rrtstar":
		planner = og.RRTstar(si)
		planner.setRange(float(d))
		return planner
	elif plannerType.lower() == "sorrtstar":
		return og.SORRTstar(si)
	else:
		ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")


# Keep these in alphabetical order and all lower case
def allocateObjective(si, objectiveType):
	if objectiveType.lower() == "pathclearance":
		return getClearanceObjective(si)
	elif objectiveType.lower() == "pathlength":
		return getPathLengthObjective(si)
	elif objectiveType.lower() == "thresholdpathlength":
		return getThresholdPathLengthObj(si)
	elif objectiveType.lower() == "weightedlengthandclearancecombo":
		return getBalancedObjective1(si)
	else:
		ou.OMPL_ERROR("Optimization-objective is not implemented in allocation function.")


def plan(mapfile,start_node, goal_node,runTime, plannerType, objectiveType, d, fname):

	boundary, blocks = load_map(mapfile)
	boundary[:,:3] = boundary[:,:3] +0.0005
	boundary[:,3:] = boundary[:,3:] -0.0005
	alpha = 1.05
	blocks = blocks*alpha
# Construct the robot state space in which we're planning. We're
	space = ob.RealVectorStateSpace(3)
	sbounds = ob.RealVectorBounds(3)
	sbounds.low[0] =float(boundary[0,0])
	sbounds.high[0] =float(boundary[0,3])

	sbounds.low[1] = float(boundary[0,1])
	sbounds.high[1] = float(boundary[0,4])

	sbounds.low[2] = float(boundary[0,2])
	sbounds.high[2] = float(boundary[0,5])

	space.setBounds(sbounds)
	# Construct a space information instance for this state space
	si = ob.SpaceInformation(space)

	# Set the object used to check which states in the space are valid
	validityChecker = ValidityChecker(si, boundary, blocks)
	si.setStateValidityChecker(validityChecker)

	mv = MyMotionValidator(si, boundary, blocks)
	si.setMotionValidator(mv)

	si.setup()

	# Set our robot's starting state
	start = ob.State(space)
	start[0] = start_node[0]
	start[1] = start_node[1]
	start[2] = start_node[2]
	# Set our robot's goal state 
	goal = ob.State(space)
	goal[0] = goal_node[0]
	goal[1] = goal_node[1]
	goal[2] = goal_node[2]

	# Create a problem instance
	pdef = ob.ProblemDefinition(si)

	# Set the start and goal states
	pdef.setStartAndGoalStates(start, goal)

	# Create the optimization objective specified by our command-line argument.
	# This helper function is simply a switch statement.
	pdef.setOptimizationObjective(allocateObjective(si, objectiveType))

	# Construct the optimal planner specified by our command line argument.
	# This helper function is simply a switch statement.
	optimizingPlanner = allocatePlanner(si,d, plannerType)

	# Set the problem instance for our planner to solve
	optimizingPlanner.setProblemDefinition(pdef)
	optimizingPlanner.setup()

	solved = "Approximate solution"
	# attempt to solve the planning problem in the given runtime
	t1 = tic()
	while(str(solved) == 'Approximate solution'):

		solved = optimizingPlanner.solve(runTime)
		# print(pdef.getSolutionPath().printAsMatrix())
	t2 = toc(t1)
	p = pdef.getSolutionPath()
	ps = og.PathSimplifier(si)
	ps.simplifyMax(p)

	if solved:
	 # Output the length of the path found
		print('{0} found solution of path length {1:.4f} with an optimization ' \
		 'objective value of {2:.4f}'.format( \
		 optimizingPlanner.getName(), \
		 pdef.getSolutionPath().length(), \
		 pdef.getSolutionPath().cost(pdef.getOptimizationObjective()).value()))

		# If a filename was specified, output the path as a matrix to
		# that file for visualization
	env = mapfile.split(".")[1].split("/")[-1]
	fname = fname + "{}_{}_{}_{}.txt".format(env, d,np.round(pdef.getSolutionPath().length(),2),np.round(t2,2))

	if fname:
		with open(fname, 'w') as outFile:
		 outFile.write(pdef.getSolutionPath().printAsMatrix())

	path = np.genfromtxt(fname)
	print("The found path : ")
	print(path)
	pathlength = np.sum(np.sqrt(np.sum(np.diff(path,axis=0)**2,axis=1)))
	print("pathlength : {}".format(pathlength))
	fig, ax, hb, hs, hg = draw_map(boundary, blocks, start_node, goal_node)  
	ax.plot(path[:,0],path[:,1],path[:,2],'r-')
	plt.show()

def test_single_cube( runTime, plannerType, objectiveType, d, fname):
  print('Running single cube test...\n') 
  start = np.array([2.3, 2.3, 1.3])
  goal = np.array([7.0, 7.0, 5.5])
  plan('./maps/single_cube.txt',start, goal, runTime, plannerType, objectiveType, d, fname)
  print('\n')
  
  
def test_maze( runTime, plannerType, objectiveType, d, fname):
  print('Running maze test...\n') 
  start = np.array([0.0, 0.0, 1.0])
  goal = np.array([12.0, 12.0, 5.0])
  plan('./maps/maze.txt',start, goal, runTime, plannerType, objectiveType, d, fname)
  print('\n')

    
def test_window( runTime, plannerType, objectiveType, d, fname):
  print('Running window test...\n') 
  start = np.array([0.2, -4.9, 0.2])
  goal = np.array([6.0, 18.0, 3.0])
  plan('./maps/window.txt',start, goal, runTime, plannerType, objectiveType, d, fname)
  print('\n')

  
def test_tower( runTime, plannerType, objectiveType, d, fname):
  print('Running tower test...\n') 
  start = np.array([2.5, 4.0, 0.5])
  goal = np.array([4.0, 2.5, 19.5])
  plan('./maps/tower.txt',start, goal, runTime, plannerType, objectiveType, d, fname)
  print('\n')

     
def test_flappy_bird( runTime, plannerType, objectiveType, d, fname):
  print('Running flappy bird test...\n') 
  start = np.array([0.5, 2.5, 5.5])
  goal = np.array([19.0, 2.5, 5.5])
  plan('./maps/flappy_bird.txt',start, goal, runTime, plannerType, objectiveType, d, fname)
  print('\n')

  
def test_room( runTime, plannerType, objectiveType, d, fname):
  print('Running room test...\n') 
  start = np.array([1.0, 5.0, 1.5])
  goal = np.array([9.0, 7.0, 1.5])
  plan('./maps/room.txt',start, goal, runTime, plannerType, objectiveType, d, fname)
  print('\n')


def test_monza( runTime, plannerType, objectiveType, d, fname):
  print('Running monza test...\n')
  start = np.array([0.5, 1.0, 4.9])
  goal = np.array([3.8, 1.0, 0.1])
  plan('./maps/monza.txt',start, goal, runTime, plannerType, objectiveType, d, fname)
  print('\n')


if __name__ == "__main__":
#  # Create an argument parser
	parser = argparse.ArgumentParser(description='Optimal motion planning demo program.')

	# Add a filename argument
	parser.add_argument('-t', '--runtime', type=float, default=1.0, help=\
		'(Optional) Specify the runtime in seconds. Defaults to 1 and must be greater than 0.')
	parser.add_argument('-p', '--planner', default='RRTstar', \
		choices=['BFMTstar', 'BITstar', 'FMTstar', 'InformedRRTstar', 'PRMstar', 'RRTstar', \
		'SORRTstar'], \
		help='(Optional) Specify the optimal planner to use, defaults to RRTstar if not given.')
	parser.add_argument('-o', '--objective', default='PathLength', \
		choices=['PathClearance', 'PathLength', 'ThresholdPathLength', \
		'WeightedLengthAndClearanceCombo'], \
		help='(Optional) Specify the optimization objective, defaults to PathLength if not given.')
	parser.add_argument('-f', '--dir', default=".", \
		help='(Optional) Specify an output path for the found solution path.')
	parser.add_argument('-i', '--info', type=int, default=0, choices=[0, 1, 2], \
		help='(Optional) Set the OMPL log level. 0 for WARN, 1 for INFO, 2 for DEBUG.' \
		' Defaults to WARN.')
	parser.add_argument('-e', '--env',default='flappy_bird',choices=['cube','maze','flappy_bird','monza','window','tower','room'],\
		help='choose environment')
	parser.add_argument('-d', '--range',default=1.0,\
		help='choose sampling range')
	# Parse the arguments
	args = parser.parse_args()

	# Check that time is positive
	if args.runtime <= 0:
		raise argparse.ArgumentTypeError(
			"argument -t/--runtime: invalid choice: %r (choose a positive number greater than 0)" \
			% (args.runtime,))

	# Set the log level
	if args.info == 0:
		ou.setLogLevel(ou.LOG_WARN)
	elif args.info == 1:
		ou.setLogLevel(ou.LOG_INFO)
	elif args.info == 2:
		ou.setLogLevel(ou.LOG_DEBUG)
	else:
		ou.OMPL_ERROR("Invalid log-level integer.")


	if(args.env == 'cube'):
		test_single_cube( args.runtime, args.planner, args.objective,args.range, args.dir)
	elif(args.env == 'maze'):
		test_maze( args.runtime, args.planner, args.objective,args.range, args.dir)
	elif(args.env == 'flappy_bird'):
		test_flappy_bird( args.runtime, args.planner, args.objective,args.range, args.dir)
	elif(args.env == 'monza'):
		test_monza( args.runtime, args.planner, args.objective,args.range, args.dir)
	elif(args.env == 'window'):
		test_window( args.runtime, args.planner, args.objective,args.range, args.dir)
	elif(args.env == 'tower'):
		test_tower( args.runtime, args.planner, args.objective,args.range, args.dir)
	elif(args.env == 'room'):
		test_room( args.runtime, args.planner, args.objective,args.range, args.dir)