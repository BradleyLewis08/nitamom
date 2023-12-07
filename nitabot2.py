"""youbot_controller controller."""

from controller import Robot, Motor, Camera, Accelerometer, GPS, Gyro, LightSensor, Receiver, RangeFinder, Lidar
from controller import Supervisor

from youbot_zombie import *
   
#------------------CHANGE CODE BELOW HERE ONLY--------------------------
#define functions here for making decisions and using sensor inputs
import cv2    
import numpy as np

import time
import math

BLUE = [33,158,238]
AQUA = [29, 219, 192]
GREEN = [36, 200, 35]
PURPLE = [144, 59, 213]
BLUE_SHADOW = [11, 39, 94]
AQUA_SHADOW = [10, 68, 69]
GREEN_SHADOW = [10, 53, 15]
PURPLE_SHADOW = [45, 21, 102]

RED = [211, 61, 43]
YELLOW = [211, 199, 29]
PINK = [198, 127, 171]
ORANGE = [194, 123, 83]
RED_SHADOW = [71, 19, 18]
YELLOW_SHADOW = [70, 69, 13]
PINK_SHADOW = [63, 39, 69]
ORANGE_SHADOW = [63, 39, 32]

YELLOW3 = [145, 135, 21]

CAMERA_SIZE = 8192

BERRY_LIST = [RED, YELLOW, PINK, ORANGE, RED_SHADOW, YELLOW_SHADOW, PINK_SHADOW, ORANGE_SHADOW]
BERRY_VARIATION = 5

ZOMBIE_LIST = [BLUE, AQUA, GREEN, PURPLE, BLUE_SHADOW, AQUA_SHADOW, GREEN_SHADOW, PURPLE_SHADOW]
ZOMBIE_VARIATION = 20

BORDER = [75, 75, 75]
WALL_SHADOW = [71, 77, 97]
WALL = [211, 212, 216]
WALL_LIST = [BORDER, WALL, WALL_SHADOW]
WALL_THRESHOLD = 10

FRONT_THRESHOLD = 800
LEFT_THRESHOLD = 800
RIGHT_THRESHOLD = 800

BERRY_FRONT_THRESHOLD = 50
BERRY_LEFT_THRESHOLD = 50
BERRY_RIGHT_THRESHOLD = 50


HEALTH_THRESHOLD = 30
# ENERGY_THRESHOLD = 30

MAX_SPEED = 14.8

class ZombieWorldState:

    def __init__(self, fr, fl, br, bl, arm1, arm2, arm3, arm4):

        self.fr = fr
        self.fl = fl
        self.br = br
        self.bl = bl

        self.arm1 = arm1
        self.arm2 = arm2
        self.arm3 = arm3
        self.arm4 = arm4

        self.isTrapped = False
        self.isGrinding = False
        self.zombieDetected = False
        self.frontZombie = False
        self.leftZombie = False
        self.rightZombie = False
        self.berryDetected = False
        self.frontBerry = False
        self.leftBerry = False
        self.rightBerry = False

        self.frontWallPixels = -1
        self.nearWall = False
        self.leftWall = False
        self.rightWall = False

    def base_forwards(self):
        self.fr.setVelocity(MAX_SPEED)
        self.fl.setVelocity(MAX_SPEED)
        self.br.setVelocity(MAX_SPEED)
        self.bl.setVelocity(MAX_SPEED)

    def base_reset(self):
        self.fr.setVelocity(0)
        self.fl.setVelocity(0)
        self.br.setVelocity(0)
        self.bl.setVelocity(0)

    def base_turn_left(self):
        self.fr.setVelocity(MAX_SPEED)
        self.fl.setVelocity(-1)
        self.br.setVelocity(MAX_SPEED)
        self.bl.setVelocity(-1)
        # self.fr.setVelocity(MAX_SPEED)
        # self.fl.setVelocity(-MAX_SPEED*.5)
        # self.br.setVelocity(MAX_SPEED)
        # self.bl.setVelocity(-MAX_SPEED*.5)

    def base_turn_right(self):
        self.fr.setVelocity(1)
        self.fl.setVelocity(MAX_SPEED)
        self.br.setVelocity(1)
        self.bl.setVelocity(MAX_SPEED)
        # self.fr.setVelocity(-MAX_SPEED*.5)
        # self.fl.setVelocity(MAX_SPEED)
        # self.br.setVelocity(-MAX_SPEED*.5)
        # self.bl.setVelocity(MAX_SPEED)

    def base_backwards(self):
        self.fr.setVelocity(-MAX_SPEED)
        self.fl.setVelocity(-MAX_SPEED)
        self.br.setVelocity(-MAX_SPEED)
        self.bl.setVelocity(-MAX_SPEED)

    def custom_turn(self, fr, fl, br, bl):
        self.fr.setVelocity(fr)
        self.fl.setVelocity(fl)
        self.br.setVelocity(br)
        self.bl.setVelocity(bl)

    def random_turn(self):
        return random.choice([self.base_turn_left, self.base_turn_right])

    def detect_obstacles(self, prev_location, curr_location, front_image, left_image, right_image, i):
        stuck = False
        if i % 20 == 0:
            dist_traveled = math.sqrt((prev_location[0] - curr_location[0])**2 + (prev_location[2] - curr_location[2])**2)
            stuck = dist_traveled < 0.01
            print(dist_traveled)
    #     grinding = math.sqrt((prev_location[0] - curr_location[0])**2 + (prev_location[2] - curr_location[2])**2) < 0.5
    #     # print("total distance: ", math.sqrt((prev_location[0] - curr_location[0])**2 + (prev_location[2] - curr_location[2])**2))
       
        walls = [np.asarray(wall) for wall in WALL_LIST]
        wall_bounds = [(wall - WALL_THRESHOLD, wall + WALL_THRESHOLD) for wall in walls]

        front_masks = [cv2.inRange(front_image, lower_bound, upper_bound) for lower_bound, upper_bound in wall_bounds]
        front_wall_pixels = [np.count_nonzero(mask) for mask in front_masks]

        left_masks = [cv2.inRange(left_image, lower_bound, upper_bound) for lower_bound, upper_bound in wall_bounds]
        right_masks = [cv2.inRange(right_image, lower_bound, upper_bound) for lower_bound, upper_bound in wall_bounds]

        left_wall_pixels = [np.count_nonzero(mask) for mask in left_masks]
        right_wall_pixels = [np.count_nonzero(mask) for mask in right_masks]

        self.nearWall = sum(front_wall_pixels) > .8*CAMERA_SIZE
        self.leftWall = sum(left_wall_pixels) > .8*CAMERA_SIZE
        self.rightWall = sum(right_wall_pixels) > .8*CAMERA_SIZE
        print("front wall pixels: ", sum(front_wall_pixels))
        print("left wall pixels: ", sum(left_wall_pixels))
        



        # if sum(left_wall_pixels) > sum(right_wall_pixels):
        #     self.leftWall = True
        # else:
        #     self.leftWall = False
        # print(sum(left_wall_pixels), sum(right_wall_pixels))


        # if sum(front_wall_pixels) > FRONT_WALL_THRESHOLD:
        #     # print("too much front wall")
        #     self.frontWall = True
        # else:
        #     self.frontWall = False
        # if sum(left_wall_pixels) > .5*CAMERA_SIZE:
        #     self.leftWall = True
        # else:
        #     self.leftWall = False
        # if sum(right_wall_pixels) > RIGHT_WALL_THRESHOLD:
        #     self.rightWall = True
        # else: 
        #     self.rightWall = False
    #     print(wall_bounds, "wall_bounds")
    #     #wall_bounds = [(0, 0, 0], [255, 255, 255]]

    #     front_masks = [cv2.inRange(front_image, lower_bound, upper_bound) for lower_bound, upper_bound in wall_bounds]
    #     self.frontWallPixels = [np.count_nonzero(mask) for mask in front_masks]
    #     print("YAY")
    #     print("wall: ", self.frontWallPixels)

    #     

        self.isTrapped = stuck
    #     self.isGrinding = grinding

    def detect_zombies(self, front_image, left_image, right_image):
        zombies = [np.asarray(zombie) for zombie in ZOMBIE_LIST]
        zombie_bounds = [(zombie - ZOMBIE_VARIATION, zombie + ZOMBIE_VARIATION) for zombie in zombies]

        front_masks = [cv2.inRange(front_image, lower_bound, upper_bound) for lower_bound, upper_bound in zombie_bounds]
        front_zombie_pixels = [np.count_nonzero(mask) for mask in front_masks]

        left_masks = [cv2.inRange(left_image, lower_bound, upper_bound) for lower_bound, upper_bound in zombie_bounds]
        right_masks = [cv2.inRange(right_image, lower_bound, upper_bound) for lower_bound, upper_bound in zombie_bounds]

        left_zombie_pixels = [np.count_nonzero(mask) for mask in left_masks]
        right_zombie_pixels = [np.count_nonzero(mask) for mask in right_masks]

        # print("zombie pix", sum(front_zombie_pixels), sum(left_zombie_pixels), sum(right_zombie_pixels))
        # print("here")
        if sum(front_zombie_pixels) > FRONT_THRESHOLD:
            # print(sum(front_zombie_pixels))
            self.frontZombie = True
        if sum(left_zombie_pixels) > LEFT_THRESHOLD: # by a certain margin
            # print(sum(left_zombie_pixels))
            self.leftZombie = True
        if sum(right_zombie_pixels) > RIGHT_THRESHOLD: # by a certain margin
            # print(sum(right_zombie_pixels))
            self.rightZombie = True
       
        if sum(left_zombie_pixels) > sum(right_zombie_pixels)*2:
            self.rightZombie = False
        elif sum(right_zombie_pixels) > sum(left_zombie_pixels)*2:
            self.leftZombie = False

        if self.frontZombie or self.leftZombie or self.rightZombie:
            self.zombieDetected = True

    def avoid_zombies(self):
       
        if self.frontZombie:
            if self.leftZombie and self.rightZombie:
                self.base_backwards()
            elif self.leftZombie:
                self.base_turn_right()
            elif self.rightZombie:
                self.base_turn_left()
            else:
                random.choice([self.base_turn_left(), self.base_turn_right()])
        elif self.leftZombie and self.rightZombie:
            self.base_forwards()
        elif self.leftZombie:
            self.base_turn_right()
        elif self.rightZombie:
            self.base_turn_left()        
        else:
            # should never hit in theory
            self.base_forwards()
       
        self.reset_zombie_detection()

    def reset_zombie_detection(self):
        self.zombieDetected = False
        self.frontZombie = False
        self.leftZombie = False
        self.rightZombie = False

    def detect_berry(self, front_image, left_image, right_image):
        berries = [np.asarray(berry) for berry in BERRY_LIST]
        berry_bounds = [(berry - BERRY_VARIATION, berry + BERRY_VARIATION) for berry in berries]

        front_masks = [cv2.inRange(front_image, lower_bound, upper_bound) for lower_bound, upper_bound in berry_bounds]
        front_berry_pixels = [np.count_nonzero(mask) for mask in front_masks]

        left_masks = [cv2.inRange(left_image, lower_bound, upper_bound) for lower_bound, upper_bound in berry_bounds]
        right_masks = [cv2.inRange(right_image, lower_bound, upper_bound) for lower_bound, upper_bound in berry_bounds]

        left_berry_pixels = [np.count_nonzero(mask) for mask in left_masks]
        right_berry_pixels = [np.count_nonzero(mask) for mask in right_masks]

        if sum(front_berry_pixels) > BERRY_FRONT_THRESHOLD:
            self.frontBerry = True
        if sum(left_berry_pixels) > BERRY_LEFT_THRESHOLD:
            self.leftBerry = True
        if sum(right_berry_pixels) > BERRY_RIGHT_THRESHOLD:
            self.rightBerry = True

        if sum(left_berry_pixels) > sum(right_berry_pixels)*2:
            self.rightBerry = False
        elif sum(right_berry_pixels) > sum(left_berry_pixels)*2:
            self.leftBerry = False
        else:
            if self.leftBerry and self.rightBerry:
                choice = random.choice([self.leftBerry, self.rightBerry])
                choice = False

        # print(front_berry_pixels, left_berry_pixels, right_berry_pixels)

        if self.frontBerry or self.leftBerry or self.rightBerry:
            self.berryDetected = True
   
    def go_to_berry(self):
        if self.frontBerry:
            # Orient until the center pixel of our camera is the same color as the berry
            # print("orienting towards berry: front")
            self.base_forwards()
        elif self.leftBerry:
            # self.base_turn_left()
            self.custom_turn(10, -1, 10, -1)
            # print("orienting towards berry: left")
        elif self.rightBerry:
            # self.base_turn_right()
            self.custom_turn(-1, 10, -1, 10)
            # print("orienting towards berry: right")

        self.reset_berry_detection()

    def reset_berry_detection(self):      
        self.frontBerry = False
        self.leftBerry = False
        self.rightBerry = False
        self.berryDetected= False





def process_camera_data(front_camera, left_camera, right_camera):

    front_image = np.frombuffer(front_camera.getImage(), np.uint8).reshape((front_camera.getHeight(), front_camera.getWidth(), 4))
    left_image = np.frombuffer(left_camera.getImage(), np.uint8).reshape((left_camera.getHeight(), left_camera.getWidth(), 4))
    right_image = np.frombuffer(right_camera.getImage(), np.uint8).reshape((right_camera.getHeight(), right_camera.getWidth(), 4))

    front_rgb_image = cv2.cvtColor(front_image, cv2.COLOR_BGRA2RGB)
    left_rgb_image = cv2.cvtColor(left_image, cv2.COLOR_BGRA2RGB)
    right_rgb_image = cv2.cvtColor(right_image, cv2.COLOR_BGRA2RGB)

    return front_rgb_image, left_rgb_image, right_rgb_image
   
#------------------CHANGE CODE ABOVE HERE ONLY--------------------------

def main():
    robot = Supervisor()

    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())
   
    #health, energy, armour in that order
    robot_info = [100,100,0]
    passive_wait(0.1, robot, timestep)
    pc = 0
    timer = 0
   
    robot_node = robot.getFromDef("Youbot")
    trans_field = robot_node.getField("translation")
   
    get_all_berry_pos(robot)
   
    robot_not_dead = 1
    prev_values = None
    curr_values = None

    #------------------CHANGE CODE BELOW HERE ONLY--------------------------
   
    #COMMENT OUT ALL SENSORS THAT ARE NOT USED. READ SPEC SHEET FOR MORE DETAILS
    # accelerometer = robot.getDevice("accelerometer")
    # accelerometer.enable(timestep)
   
    gps = robot.getDevice("gps")
    gps.enable(timestep)
   
    # compass = robot.getDevice("compass")
    # compass.enable(timestep)
   
    camera1 = robot.getDevice("ForwardLowResBigFov")
    camera1.enable(timestep)
   
    # camera2 = robot.getDevice("ForwardHighResSmallFov")
    # camera2.enable(timestep)
   
    # camera3 = robot.getDevice("ForwardHighRes")
    # camera3.enable(timestep)
   
    # camera4 = robot.getDevice("ForwardHighResSmall")
    # camera4.enable(timestep)
   
    # camera5 = robot.getDevice("BackLowRes")
    # camera5.enable(timestep)
   
    camera6 = robot.getDevice("RightLowRes")
    camera6.enable(timestep)
   
    camera7 = robot.getDevice("LeftLowRes")
    camera7.enable(timestep)
   
    # camera8 = robot.getDevice("BackHighRes")
    # camera8.enable(timestep)
   
    # gyro = robot.getDevice("gyro")
    # gyro.enable(timestep)
   
    # lightSensor = robot.getDevice("light sensor")
    # lightSensor.enable(timestep)
   
    # receiver = robot.getDevice("receiver")
    # receiver.enable(timestep)
   
    # rangeFinder = robot.getDevice("range-finder")
    # rangeFinder.enable(timestep)
   
    # lidar = robot.getDevice("lidar")
    # lidar.enable(timestep)
   
    fr = robot.getDevice("wheel1")
    fl = robot.getDevice("wheel2")
    br = robot.getDevice("wheel3")
    bl = robot.getDevice("wheel4")

    arm1 = robot.getDevice("arm1")
    arm2 = robot.getDevice("arm2")
    arm3 = robot.getDevice("arm3")
    arm4 = robot.getDevice("arm4")
   
    fr.setPosition(float('inf'))
    fl.setPosition(float('inf'))
    br.setPosition(float('inf'))
    bl.setPosition(float('inf'))

    arm1.setPosition(-1)
    arm2.setPosition(-1)
    arm3.setPosition(-1.2)
    arm4.setPosition(.7)

    i=0

    emergency_turn_counter = 0
    random_counter = 0
    prev_location = [-1, -1, -1]
    prev = robot_info[0], robot_info[1]
    randomChoice = None

    zombieState = ZombieWorldState(fr, fl, br, bl, arm1, arm2, arm3, arm4)

    #------------------CHANGE CODE ABOVE HERE ONLY--------------------------
   
   
    while(robot_not_dead == 1):

        print(zombieState.nearWall)

        desperate = robot_info[0] < HEALTH_THRESHOLD
       
        if(robot_info[0] < 0):

           
            robot_not_dead = 0
            print("ROBOT IS OUT OF HEALTH")
            #if(zombieTest):
            #    print("TEST PASSED")
            #else:
            #    print("TEST FAILED")
            #robot.simulationQuit(20)
            #exit()
           
        if(timer%2==0):
            trans = trans_field.getSFVec3f()
            robot_info = check_berry_collision(robot_info, trans[0], trans[2], robot)
            robot_info = check_zombie_collision(robot_info, trans[0], trans[2], robot)
           
        if(timer%16==0):
            robot_info = update_robot(robot_info)
            timer = 0
       
        if(robot.step(timestep)==-1):
            exit()
           
           
        timer += 1
       
     #------------------CHANGE CODE BELOW HERE ONLY--------------------------   #called every timestep
        curr = robot_info[0], robot_info[1]
        # print(prev, curr)

        cameraData = process_camera_data(camera1, camera7, camera6)

        # print("emergency: ", emergency_turn_counter)

        #GO BACKWARDS IF ISTRAPPED == TRUE
        # if emergency_turn_counter > 0:
        #     print("now moving with emergency counter")
        #     zombieState.base_turn_left()

        #     print("left wall detected: ", zombieState.leftWall)
        #     print("right wall detected: ", zombieState.rightWall)
        #     emergency_turn_counter -= 1
        #     # print("bruh")
        #     if emergency_turn_counter == 0:
        #     #    choice = random.choice([[10, 5, 10, 5], [5, 10, 5, 10]])
        #     #    zombieState.custom_turn(choice[0], choice[1], choice[2], choice[3])
        #         zombieState.isTrapped = False
        #         zombieState.isGrinding = False
        #     #    grinding_turn_counter = 0
        #     continue

        if emergency_turn_counter > 0:
            choice = [-14, -5, -14, -5]
            #choice = random.choice([[-10, -10, -10, -10], [-10, -10, -10, -10], [-10, -5, -10, -5], [-5, -10, -5, -10]])
            zombieState.custom_turn(choice[0], choice[1], choice[2], choice[3])
            emergency_turn_counter -= 1
            # print("bruh")
            if emergency_turn_counter == 0:
            #    choice = random.choice([[10, 5, 10, 5], [5, 10, 5, 10]])
            #    zombieState.custom_turn(choice[0], choice[1], choice[2], choice[3])
                zombieState.isTrapped = False
                zombieState.isGrinding = False
            #    grinding_turn_counter = 0
            continue

        if random_counter > 0:
            print("random behavior loop")
            zombieState.custom_turn(randomChoice[0], randomChoice[1], randomChoice[2], randomChoice[3])
            random_counter -= 1
            continue

       

        zombieState.detect_zombies(cameraData[0], cameraData[1], cameraData[2])
        zombieState.detect_berry(cameraData[0], cameraData[1], cameraData[2])

        curr_location = gps.getValues()

        # 
        zombieState.detect_obstacles(prev_location, curr_location, cameraData[0], cameraData[1], cameraData[2], i)
        if i % 10 == 0:
            prev_location = curr_location

        if zombieState.isTrapped:
            emergency_turn_counter = 15
        elif zombieState.zombieDetected and not desperate:
            print("calling avoid zombies")
            zombieState.avoid_zombies()
        # elif zombieState.berryDetected:
        #     print("calling go to berry")
        #     zombieState.go_to_berry()
        elif zombieState.nearWall:
            print("zombiestate.nearwall is true")
            if zombieState.rightWall and not zombieState.leftWall:
                zombieState.base_turn_left() #might need to add counter here
            elif zombieState.leftWall and not zombieState.rightWall:
                zombieState.base_turn_right()
            else:
                zombieState.base_turn_right()
            # elif zombieState.leftWall and zombieState.rightWall:
            #     if sum(zombieState.left)
        else:
            print("calling random behavior")
            # if not i%25:
            if desperate:
                print("desperate")
                choice = random.choice([[MAX_SPEED]*4, [MAX_SPEED]*4, [10, 4, 10, 4], [4, 10, 4, 10]])
                # print(choice)
                zombieState.custom_turn(choice[0], choice[1], choice[2], choice[3])
            else:
                random_counter = 5
                randomChoice = random.choice([[10, 7, 10, 7], [7, 10, 7, 10], [10, 10, 10, 10]])


                    
                    
        prev = curr
                   
        #if i <100
            #base_forwards() -> can implement in Python with Webots C code (/Zombie world/libraries/youbot_control) as an example or make your own
       
        #if == 100
            # base_reset()
            # base_turn_left()  
            #it takes about 150 timesteps for the robot to complete the turn
                 
        #if i==300
            # i = 0
       
        i+=1
        print(i)
       
        #make decisions using inputs if you choose to do so
         
        #------------------CHANGE CODE ABOVE HERE ONLY--------------------------
       
       
    return 0  


main()
