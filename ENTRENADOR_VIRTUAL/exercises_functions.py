
def chintucks(nose_angle, stage=None, rep_counter=None, serie_counter=None, max_reps=None, max_series=None):

    if nose_angle > 55:  
        stage = "UP"

    if nose_angle < 45 and stage == "UP":  
        stage = "DOWN"
        rep_counter += 1

        if rep_counter == max_reps:  
            if serie_counter < max_series:
                serie_counter += 1
                rep_counter = 0

    return stage, rep_counter, serie_counter

def headshakes(nose_position, left_shoulder_position, right_shoulder_position, stage=None, rep_counter=None, serie_counter=None, max_reps=None, max_series=None):

    nose_rightShoulder_distance = abs(nose_position - right_shoulder_position)
    nose_leftShoulder_distance = abs(nose_position - left_shoulder_position)

    if nose_rightShoulder_distance > nose_leftShoulder_distance:  
        stage = "RIGHT"

    if nose_rightShoulder_distance < nose_leftShoulder_distance and stage == "RIGHT":  
        stage = "LEFT"
        rep_counter += 1

        if rep_counter == max_reps:  
            if serie_counter < max_series:
                serie_counter += 1
                rep_counter = 0

    return stage, rep_counter, serie_counter

def jumpingjacks(shoulder_angle, left_hip_position, right_hip_position, left_ankle_position, right_angkle_position, stage=None, rep_counter=None, serie_counter=None, max_reps=None, max_series=None):
    
    # calc distance between ankles
    ankle_distance = left_ankle_position - right_angkle_position
    hip_distance =  left_hip_position - right_hip_position
    min_ankle_distance = 2 * hip_distance

    if shoulder_angle < 140 and ankle_distance >= min_ankle_distance:  
        stage = "CLOSE"

    if shoulder_angle > 200 and ankle_distance < min_ankle_distance and stage == "CLOSE" :  
        stage = "OPEN"
        rep_counter += 1

        if rep_counter == max_reps:  
            if serie_counter < max_series:
                serie_counter += 1
                rep_counter = 0

    return stage, rep_counter, serie_counter

def legswings(knee_angle, stage=None, rep_counter=None, serie_counter=None, max_reps=None, max_series=None):
    
    if knee_angle > 190:  
        stage = "START"
    
    if knee_angle < 185 and stage == "START":  
        stage = "MID REP"
        rep_counter += 1

        if rep_counter == max_reps:  
            if serie_counter < max_series:
                serie_counter += 1
                rep_counter = 0
                
    return stage, rep_counter, serie_counter

def kneesraises(left_knee_position, left_hip_position, stage=None, rep_counter=None, serie_counter=None, max_reps=None, max_series=None):
    
    if left_knee_position >= left_hip_position:  
        stage = "DOWN"
    
    if left_knee_position < left_hip_position and stage == "DOWN":  
        stage = "UP"
        rep_counter += 1

        if rep_counter == max_reps:  
            if serie_counter < max_series:
                serie_counter += 1
                rep_counter = 0

    return stage, rep_counter, serie_counter

def verticaljump(angle, stage=None, rep_counter=None, serie_counter=None, max_reps=None, max_series=None):
    
    if angle >= 25:  
        stage = "DOWN"
    
    if angle < 24 and stage == "DOWN":  
        stage = "UP"
        rep_counter += 1

        if rep_counter == max_reps:  
            if serie_counter < max_series:
                serie_counter += 1
                rep_counter = 0

    return stage, rep_counter, serie_counter