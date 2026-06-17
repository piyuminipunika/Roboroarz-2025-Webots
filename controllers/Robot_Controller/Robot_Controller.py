from controller import Robot
import cv2
import numpy as np
import math
from collections import deque

# --- CONSTANTS ---
TIME_STEP = 32
MAX_SPEED = 6.28
TILE_SIZE = 0.25
WHEEL_RADIUS = 0.0205
MAZE_W = 12
MAZE_H = 12

# --- SENSOR THRESHOLDS FOR E-PUCK ---
FRONT_WALL_THRESHOLD = 100.0    
LEFT_WALL_THRESHOLD = 100.0      
RIGHT_WALL_THRESHOLD = 100.0     
ALIGN_TOO_CLOSE = 135.0         
ALIGN_CENTER = 120.0            
COLLISION_THRESHOLD = 180.0     

# --- DIRECTIONS (+Y=Forward, +X=Left, -Y=Back, -X=Right) ---
DIR_FWD = 0 
DIR_LFT = 1 
DIR_BCK = 2 
DIR_RGT = 3 

# --- STATES ---
STATE_SCAN = 0
STATE_ALIGN = 1
STATE_PRE_TURN = 2
STATE_TURN = 3
STATE_MOVE = 4
STATE_STOP = 5

class RoboroarzController:
    def __init__(self):
        self.robot = Robot()
        
        # Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # Encoders
        self.ps_left_enc = self.robot.getDevice('left wheel sensor')
        self.ps_right_enc = self.robot.getDevice('right wheel sensor')
        self.ps_left_enc.enable(TIME_STEP)
        self.ps_right_enc.enable(TIME_STEP)
        
        # Camera & IMU
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(TIME_STEP)
        self.imu = self.robot.getDevice('inertial unit')
        if not self.imu: self.imu = self.robot.getDevice('imu')
        if self.imu: self.imu.enable(TIME_STEP)

        # Distance Sensors
        self.ps = [self.robot.getDevice(f'ps{i}') for i in range(8)]
        for s in self.ps: s.enable(TIME_STEP)

        # Map Memory
        self.walls = [[0]*MAZE_H for _ in range(MAZE_W)]
        self.costs = [[999]*MAZE_H for _ in range(MAZE_W)]
        
        # Odometry
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.heading = 0.0
        self.start_heading = 0.0 
        self.prev_left = 0.0
        self.prev_right = 0.0
        self.pre_turn_start_x = 0.0
        self.pre_turn_start_y = 0.0
        self.move_start_x = 0.0
        self.move_start_y = 0.0
        
        # Goal State & Queuing
        self.current_cell_x = 0
        self.current_cell_y = 0
        self.final_goal_x = 0
        self.final_goal_y = 1
        self.target_cell_x = 0
        self.target_cell_y = 1
        
        self.state = STATE_SCAN
        self.target_heading_rad = 0.0 
        self.tag_sequence_started = False
        
        
        # New Goal Management Variables
        self.goal_queue = []
        self.completed_goals = set()
        
        self.step_counter = 0

    # --- MATH & HELPERS ---
    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

    def update_odometry(self):
        curr_left = self.ps_left_enc.getValue()
        curr_right = self.ps_right_enc.getValue()
        if math.isnan(curr_left) or math.isnan(curr_right): return
        
        d_left = curr_left - self.prev_left
        d_right = curr_right - self.prev_right
        self.prev_left = curr_left
        self.prev_right = curr_right
        
        linear_dist = ((d_left * WHEEL_RADIUS) + (d_right * WHEEL_RADIUS)) / 2.0
        if self.imu: self.heading = self.imu.getRollPitchYaw()[2]
        self.heading = self.normalize_angle(self.heading)
        
        rel_angle = self.normalize_angle(self.heading - self.start_heading)
        self.pos_x += linear_dist * math.sin(rel_angle) 
        self.pos_y += linear_dist * math.cos(rel_angle) 

    def decode_tag(self):
        img = self.camera.getImage()
        if not img:
            return []

        h = self.camera.getHeight()
        w = self.camera.getWidth()
        frame = np.frombuffer(img, np.uint8).reshape((h, w, 4))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        CAM_W, CAM_H = 640, 480
        if gray.shape != (CAM_H, CAM_W):
            gray = cv2.resize(gray, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)

        found_tags = []
        if ids is not None:
            for i in range(len(ids)):
                area = abs(cv2.contourArea(corners[i][0]))
                if 120 <= area <= 20000:
                    val = int(ids[i][0])
                    tag_coord = ((val >> 4) & 0x0F, val & 0x0F)
                    found_tags.append(tag_coord)

        return found_tags

    # --- MAPPING LOGIC ---
    def get_heading_enum(self):
        angle = self.normalize_angle(self.heading - self.start_heading)
        if -math.pi/4 < angle <= math.pi/4: return DIR_FWD
        if math.pi/4 < angle <= 3*math.pi/4: return DIR_LFT
        if angle > 3*math.pi/4 or angle <= -3*math.pi/4: return DIR_BCK
        return DIR_RGT

    def set_wall(self, x, y, direction):
        if not (0 <= x < MAZE_W and 0 <= y < MAZE_H): return False
        added = False
        mask = 1 << direction
        if not (self.walls[x][y] & mask):
            self.walls[x][y] |= mask
            added = True
            
        nx, ny = x, y
        if direction == DIR_FWD: ny += 1
        elif direction == DIR_LFT: nx += 1
        elif direction == DIR_BCK: ny -= 1
        elif direction == DIR_RGT: nx -= 1

        if 0 <= nx < MAZE_W and 0 <= ny < MAZE_H:
            n_dir = (direction + 2) % 4
            self.walls[nx][ny] |= (1 << n_dir)
        return added

    def update_walls_epuck(self):
        c_dir = self.get_heading_enum()
        front_val = (self.ps[0].getValue() + self.ps[7].getValue()) / 2.0
        left_val = self.ps[5].getValue()
        right_val = self.ps[2].getValue()

        changed = False
        if front_val > FRONT_WALL_THRESHOLD:
            if self.set_wall(self.current_cell_x, self.current_cell_y, c_dir): changed = True
        if left_val > LEFT_WALL_THRESHOLD:
            if self.set_wall(self.current_cell_x, self.current_cell_y, (c_dir + 1) % 4): changed = True
        if right_val > RIGHT_WALL_THRESHOLD:
            if self.set_wall(self.current_cell_x, self.current_cell_y, (c_dir + 3) % 4): changed = True
            
        return changed

    def run_flood_fill(self, tx, ty):
        self.costs = [[999]*MAZE_H for _ in range(MAZE_W)]
        if not (0 <= tx < MAZE_W and 0 <= ty < MAZE_H): return
        
        self.costs[tx][ty] = 0
        q = deque([(tx, ty)])
        
        dx = [0, 1, 0, -1] # FWD, LFT, BCK, RGT
        dy = [1, 0, -1, 0]

        while q:
            cx, cy = q.popleft()
            c_cost = self.costs[cx][cy]
            for d in range(4):
                if not (self.walls[cx][cy] & (1 << d)):
                    nx, ny = cx + dx[d], cy + dy[d]
                    if 0 <= nx < MAZE_W and 0 <= ny < MAZE_H:
                        if self.costs[nx][ny] == 999:
                            self.costs[nx][ny] = c_cost + 1
                            q.append((nx, ny))

    def print_maze_detailed(self):
        c_dir = self.get_heading_enum()
        robot_char = " R "
        if c_dir == DIR_FWD: robot_char = " ^ "
        elif c_dir == DIR_LFT: robot_char = " < "
        elif c_dir == DIR_BCK: robot_char = " v "
        elif c_dir == DIR_RGT: robot_char = " > "

        print("\n================= CURRENT MAP STATE =================")
        print(f"Robot Pos: ({self.current_cell_x}, {self.current_cell_y}) | Goal: ({self.final_goal_x}, {self.final_goal_y})")

        for y in range(MAZE_H - 1, -1, -1):
            for x in range(MAZE_W - 1, -1, -1):
                print("+", end="")
                if self.walls[x][y] & (1 << DIR_FWD): print("---", end="")
                else: print("   ", end="")
            print("+")

            for x in range(MAZE_W - 1, -1, -1):
                if self.walls[x][y] & (1 << DIR_LFT): print("|", end="")
                else: print(" ", end="")
                
                # Render the robot, Goal (G), or Queued Target (E)
                if x == self.current_cell_x and y == self.current_cell_y: print(robot_char, end="")
                elif x == self.final_goal_x and y == self.final_goal_y: print(" G ", end="")
                elif (x, y) in self.goal_queue: print(" E ", end="")
                else: print("   ", end="")
            
            if self.walls[0][y] & (1 << DIR_RGT): print("|")
            else: print(" ")
            
        for x in range(MAZE_W - 1, -1, -1):
            print("+", end="")
            if self.walls[x][0] & (1 << DIR_BCK): print("---", end="")
            else: print("   ", end="")
        print("+\n=====================================================")
    
    # --- MAIN LOOP ---
    def run(self):
        for _ in range(15): self.robot.step(TIME_STEP)
        self.start_heading = self.imu.getRollPitchYaw()[2] if self.imu else 0.0
        self.prev_left = self.ps_left_enc.getValue()
        self.prev_right = self.ps_right_enc.getValue()
        
        print("--- MICRO MOUSE MAPPING STARTED ---")
        self.print_maze_detailed() 
        
        while self.robot.step(TIME_STEP) != -1:
            self.update_odometry()
            
            # --- ALWAYS READ MULTIPLE AR TAGS ---
            detected_tags = self.decode_tag()
            if detected_tags:
                curr_goal = (self.final_goal_x, self.final_goal_y)
                map_needs_print = False
                
                # Loop through all detected tags and queue any new ones
                for new_tag in detected_tags:
                    if new_tag != curr_goal and new_tag not in self.goal_queue and new_tag not in self.completed_goals:
                        print(f">> [AR TAG DETECTED] New Target Queued (E): {new_tag}")
                        self.goal_queue.append(new_tag)
                        map_needs_print = True
                        
                if map_needs_print:
                    self.print_maze_detailed()
            
            self.step_counter += 1
            if self.step_counter % 20 == 0: 
                f_val = (self.ps[0].getValue() + self.ps[7].getValue()) / 2.0
                l_val = self.ps[5].getValue()
                r_val = self.ps[2].getValue()
                print(f"[Sensors] Front: {f_val:.1f} | Left: {l_val:.1f} | Right: {r_val:.1f} | Queue: {self.goal_queue}")

            if self.state == STATE_SCAN:
                self.left_motor.setVelocity(0.0)
                self.right_motor.setVelocity(0.0)

                # 1. Update Map FIRST
                if self.update_walls_epuck():
                    print(">> [MAP] New Wall Detected. Updating Map...")
                    self.print_maze_detailed()

                # 2. ALIGNMENT CHECK
                front_val = (self.ps[0].getValue() + self.ps[7].getValue()) / 2.0
                if front_val > ALIGN_TOO_CLOSE:
                    self.state = STATE_ALIGN
                    continue

                # 3. Check Victory/Goal Reached Condition
                if self.current_cell_x == self.final_goal_x and self.current_cell_y == self.final_goal_y:
                    if self.tag_sequence_started:
                        img = self.camera.getImage()
                        w, h = self.camera.getWidth(), self.camera.getHeight()
                        r, g, b = self.camera.imageGetRed(img, w, w//2, h//2), self.camera.imageGetGreen(img, w, w//2, h//2), self.camera.imageGetBlue(img, w, w//2, h//2)
                        
                        color = None
                        if r < 45 and g < 45 and b < 45: color = "Black"
                        elif g > 180 and r < 120: color = "Green"
                        elif r > 180 and g < 120: color = "Red"
                        elif b > 180 and r < 120: color = "Blue"
                        
                        if color:
                            self.completed_goals.add((self.final_goal_x, self.final_goal_y))
                            self.robot.setCustomData(color)
                            
                            # Check Queue for Next Target
                            if len(self.goal_queue) > 0:
                                next_goal = self.goal_queue.pop(0)
                                print(f">>> [GOAL REACHED] Wall is {color}. Setting course for queued goal: {next_goal} <<<")
                                self.final_goal_x, self.final_goal_y = next_goal[0], next_goal[1]
                                self.print_maze_detailed()
                            else:
                                print(f">>> [VICTORY] All Targets Reached. Final Wall is {color}. <<<")
                                self.state = STATE_STOP
                                continue
                    else:
                        # Reached the initial exploratory nudge (0,1). Pick up queue if it exists.
                        if len(self.goal_queue) > 0:
                            next_goal = self.goal_queue.pop(0)
                            print(f">> [INITIAL SCAN DONE] Moving to first queued goal: {next_goal}")
                            self.final_goal_x, self.final_goal_y = next_goal[0], next_goal[1]
                            self.tag_sequence_started = True
                            self.print_maze_detailed()

                # 4. Flood Fill Pathfinding
                self.run_flood_fill(self.final_goal_x, self.final_goal_y)
                
                best_val = 999
                best_dir = -1
                dx = [0, 1, 0, -1] # FWD, LFT, BCK, RGT
                dy = [1, 0, -1, 0]
                
                for d in range(4):
                    if not (self.walls[self.current_cell_x][self.current_cell_y] & (1 << d)):
                        nx, ny = self.current_cell_x + dx[d], self.current_cell_y + dy[d]
                        if 0 <= nx < MAZE_W and 0 <= ny < MAZE_H:
                            if self.costs[nx][ny] < best_val:
                                best_val = self.costs[nx][ny]
                                best_dir = d
                                self.target_cell_x, self.target_cell_y = nx, ny

                # 5. Prepare to Turn or Handle Deadlocks
                if best_dir != -1:
                    if best_dir == DIR_FWD: self.target_heading_rad = self.start_heading
                    elif best_dir == DIR_LFT: self.target_heading_rad = self.start_heading + (math.pi / 2)
                    elif best_dir == DIR_BCK: self.target_heading_rad = self.start_heading + math.pi
                    elif best_dir == DIR_RGT: self.target_heading_rad = self.start_heading - (math.pi / 2)
                    
                    self.pre_turn_start_x = self.pos_x
                    self.pre_turn_start_y = self.pos_y
                    self.state = STATE_PRE_TURN
                else:
                    print(">> [ERROR] No Path Found! Clearing hallucinated walls and trying again...")
                    self.walls = [[0]*MAZE_H for _ in range(MAZE_W)]
                    self.state = STATE_SCAN

            elif self.state == STATE_ALIGN:
                front_val = (self.ps[0].getValue() + self.ps[7].getValue()) / 2.0
                if front_val > ALIGN_CENTER:
                    self.left_motor.setVelocity(-2.0)
                    self.right_motor.setVelocity(-2.0)
                else:
                    self.left_motor.setVelocity(0.0)
                    self.right_motor.setVelocity(0.0)
                    self.pos_x = self.current_cell_x * TILE_SIZE
                    self.pos_y = self.current_cell_y * TILE_SIZE
                    self.state = STATE_SCAN

            elif self.state == STATE_PRE_TURN:
                dist_moved = math.sqrt((self.pos_x - self.pre_turn_start_x)**2 + (self.pos_y - self.pre_turn_start_y)**2)
                if dist_moved < 0.02:
                    self.left_motor.setVelocity(MAX_SPEED * 0.5)
                    self.right_motor.setVelocity(MAX_SPEED * 0.5)
                else:
                    self.left_motor.setVelocity(0.0)
                    self.right_motor.setVelocity(0.0)
                    self.pos_x = self.current_cell_x * TILE_SIZE
                    self.pos_y = self.current_cell_y * TILE_SIZE
                    self.state = STATE_TURN

            elif self.state == STATE_TURN:
                diff = self.normalize_angle(self.target_heading_rad - self.heading)
                if abs(diff) < 0.05:
                    self.pos_x = self.current_cell_x * TILE_SIZE
                    self.pos_y = self.current_cell_y * TILE_SIZE
                    self.move_start_x = self.pos_x
                    self.move_start_y = self.pos_y
                    self.state = STATE_MOVE
                    self.print_maze_detailed() 
                else:
                    speed = max(min(diff * 4.0, MAX_SPEED), -MAX_SPEED)
                    self.left_motor.setVelocity(-speed)
                    self.right_motor.setVelocity(speed)

            elif self.state == STATE_MOVE:
                front_val = (self.ps[0].getValue() + self.ps[7].getValue()) / 2.0
                if front_val > COLLISION_THRESHOLD:
                    print(">> [COLLISION] Path blocked mid-move! Re-scanning...")
                    
                    self.left_motor.setVelocity(0)
                    self.right_motor.setVelocity(0)
                    
                    dx = self.pos_x - self.move_start_x
                    dy = self.pos_y - self.move_start_y
                    if abs(dx) > TILE_SIZE * 0.7 or abs(dy) > TILE_SIZE * 0.7:
                        self.current_cell_x = self.target_cell_x
                        self.current_cell_y = self.target_cell_y
                    
                    c_dir = self.get_heading_enum()
                    self.set_wall(self.current_cell_x, self.current_cell_y, c_dir)
                    
                    self.state = STATE_ALIGN
                    continue

                tx, ty = self.target_cell_x * TILE_SIZE, self.target_cell_y * TILE_SIZE
                dist = math.sqrt((tx - self.pos_x)**2 + (ty - self.pos_y)**2)
                err = self.normalize_angle(self.target_heading_rad - self.heading)
                
                if dist < 0.02:
                    self.left_motor.setVelocity(0); self.right_motor.setVelocity(0)
                    self.current_cell_x, self.current_cell_y = self.target_cell_x, self.target_cell_y
                    self.pos_x, self.pos_y = self.current_cell_x * TILE_SIZE, self.current_cell_y * TILE_SIZE
                    
                    self.state = STATE_SCAN
                    self.print_maze_detailed() 
                else:
                    corr = err * 2.0
                    l_speed = max(min(MAX_SPEED * 0.8 - corr, MAX_SPEED), -MAX_SPEED)
                    r_speed = max(min(MAX_SPEED * 0.8 + corr, MAX_SPEED), -MAX_SPEED)
                    self.left_motor.setVelocity(l_speed)
                    self.right_motor.setVelocity(r_speed)

            elif self.state == STATE_STOP:
                self.left_motor.setVelocity(0)
                self.right_motor.setVelocity(0)

if __name__ == "__main__":
    controller = RoboroarzController()
    controller.run()