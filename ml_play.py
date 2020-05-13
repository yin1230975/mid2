import pickle
import numpy as np
from mlgame.communication import ml as comm
from os import path

def ml_loop(side: str):
    """
    The main loop for the machine learning process
    The `side` parameter can be used for switch the code for either of both sides,
    so you can write the code for both sides in the same script. Such as:
    ```python
    if side == "1P":
        ml_loop_for_1P()
    else:
        ml_loop_for_2P()
    ```
    @param side The side which this script is executed for. Either "1P" or "2P".
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here
    ball_served = False
    filename1p = path.join(path.dirname(__file__),"save","clf_SVR1_pingpong1p.pickle")
    filename2p = path.join(path.dirname(__file__),"save","clf_SVR_pingpong2p.pickle")
    with open(filename1p, 'rb') as file:
        clf1p = pickle.load(file)
    with open(filename2p, 'rb') as file:
        clf2p = pickle.load(file)


    # 2. Inform the game process that ml process is ready before start the loop.
    
    
    def get_direction(VectorX,VectorY):
        if(VectorX>=0 and VectorY>=0):
            return 0
        elif(VectorX>0 and VectorY<0):
            return 1
        elif(VectorX<0 and VectorY>0):
            return 2
        elif(VectorX<0 and VectorY<0):
            return 3
        else:
            return 4
    def move_to(player, pred) : #move platform to predicted position to catch ball 
        if player == '1P':
            if scene_info["platform_1P"][0]  > (pred) and scene_info["platform_1P"][0] < (pred): return 0 # NONE
            elif scene_info["platform_1P"][0] <= (pred) : return 1 # goes right
            else : return 2 # goes left
        else :
            if scene_info["platform_2P"][0]  > (pred) and scene_info["platform_2P"][0] < (pred): return 0 # NONE
            elif scene_info["platform_2P"][0] <= (pred) : return 1 # goes right
            else : return 2 # goes left
    #2.
    comm.ml_ready()
    # 3. Start an endless loop.
    while True:
        scene_info = comm.recv_from_game()

        feature1p = []
        feature1p.append(scene_info['ball'][0])
        feature1p.append(scene_info['ball'][1])
        #feature1p.append(scene_info['platform_1P'][0])
        #feature.append(scene_info['platform_1P'][1])
        #feature1p.append(scene_info['platform_2P'][0])
        #feature.append(scene_info['platform_2P'][1])
        feature1p.append(scene_info['ball_speed'][0])
        feature1p.append(scene_info['ball_speed'][1])
        #feature1p.append(get_direction(scene_info['ball_speed'][0],scene_info['ball_speed'][1]))

        feature1p = np.array(feature1p)
        feature1p = feature1p.reshape((-1,4))

        feature2p = []
        feature2p.append(scene_info['ball'][0])
        feature2p.append(scene_info['ball'][1])
        #feature2p.append(scene_info['platform_1P'][0])
        #feature.append(scene_info['platform_1P'][1])
        #feature2p.append(scene_info['platform_2P'][0])
        #feature.append(scene_info['platform_2P'][1])
        feature2p.append(scene_info['ball_speed'][0])
        feature2p.append(scene_info['ball_speed'][1])
        #feature2p.append(get_direction(scene_info['ball_speed'][0],scene_info['ball_speed'][1]))

        feature2p = np.array(feature2p)
        feature2p = feature2p.reshape((-1,4))


        # 3.2. If either of two sides wins the game, do the updating or
        #      resetting stuff and inform the game process when the ml process
        #      is ready.
        if scene_info["status"] != "GAME_ALIVE":
            # Do some updating or resetting stuff
            ball_served = False

            # 3.2.1 Inform the game process that
            #       the ml process is ready for the next round
            comm.ml_ready()
            continue

        if not ball_served:
            comm.send_to_game({"frame": scene_info["frame"], "command": "SERVE_TO_LEFT"})
            ball_served = True
        else:
            if side == "1P":
                command = move_to('1P',clf1p.predict(feature1p))
            else:
                command = move_to('2P',clf2p.predict(feature2p))

            if command == 0:
                comm.send_to_game({"frame": scene_info["frame"], "command": "NONE"})
            elif command == 1:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
            else :
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})
