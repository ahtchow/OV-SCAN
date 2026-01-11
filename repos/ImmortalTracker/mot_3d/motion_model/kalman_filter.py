import numpy as np
from ..data_protos import BBox
from filterpy.kalman import KalmanFilter

def normalize_angle(angle):
    # Normalize angle to [0, π)
    angle = angle % (2 * np.pi)  # Map angle to [0, 2π)
    if angle >= np.pi:
        angle -= np.pi  # Map angle to [0, π)
    return angle

def minimal_angular_difference(theta1, theta2):
    # Compute the minimal difference considering wrap-around at π
    diff = np.abs(theta1 - theta2) % np.pi
    minimal_diff = min(diff, np.pi - diff)
    return minimal_diff

class KalmanFilterMotionModel:
    def __init__(self, bbox: BBox, inst_type, time_stamp, covariance='default'):
        # the time stamp of last observation
        self.prev_time_stamp = time_stamp
        self.latest_time_stamp = time_stamp
        # define constant velocity model
        self.score = bbox.s
        self.inst_type = inst_type

        self.kf = KalmanFilter(dim_x=10, dim_z=7) 
        self.kf.x[:7] = BBox.bbox2array(bbox)[:7].reshape((7, 1))
        self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                              [0,1,0,0,0,0,0,0,1,0],
                              [0,0,1,0,0,0,0,0,0,1],
                              [0,0,0,1,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])     

        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                              [0,1,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0]])
        
        self.kf.B = np.zeros((10, 1))                     # dummy control transition matrix

        # # with angular velocity
        # self.kf = KalmanFilter(dim_x=11, dim_z=7)       
        # self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
        #                       [0,1,0,0,0,0,0,0,1,0,0],
        #                       [0,0,1,0,0,0,0,0,0,1,0],
        #                       [0,0,0,1,0,0,0,0,0,0,1],  
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0],
        #                       [0,0,0,0,0,0,0,1,0,0,0],
        #                       [0,0,0,0,0,0,0,0,1,0,0],
        #                       [0,0,0,0,0,0,0,0,0,1,0],
        #                       [0,0,0,0,0,0,0,0,0,0,1]])     

        # self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
        #                       [0,1,0,0,0,0,0,0,0,0,0],
        #                       [0,0,1,0,0,0,0,0,0,0,0],
        #                       [0,0,0,1,0,0,0,0,0,0,0],
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0]])

        self.covariance_type = covariance
        if self.covariance_type == 'default':
            # self.kf.R[0:,0:] *= 10.   # measurement uncertainty

            # Modified by Adrian: increase the uncertainty of the shape
            self.kf.R[4,4] *= 10.
            self.kf.R[5,5] *= 10.
            self.kf.R[6,6] *= 10.

            self.kf.P[7:, 7:] *= 1000. 	# state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
            self.kf.P *= 10.
    
            # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
            # self.kf.Q[7:, 7:] *= 0.01

        self.history = [bbox]
    
    def predict(self, time_stamp=None):
        self.kf.predict()
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        return

    def update(self, det_bbox: BBox, aux_info=None): 
        """ 
        Updates the state vector with observed bbox.
        """
        bbox = BBox.bbox2array(det_bbox)[:7]

        # Full pipeline of KF: first predict, then update
        self.predict()

        ######################### Orientation Correction

        # Normalize angles to [0, π)
        self.kf.x[3] = normalize_angle(self.kf.x[3])
        new_theta = normalize_angle(bbox[3])
        bbox[3] = new_theta

        predicted_theta = self.kf.x[3]

        # If the angular difference is greater than 90 degrees, flip the angle
        if minimal_angular_difference(new_theta, predicted_theta) > (np.pi / 2):
            # Flip the angle and normalize again
            self.kf.x[3] = normalize_angle(self.kf.x[3] + np.pi)

        #########################

        self.kf.update(bbox)
        self.prev_time_stamp = self.latest_time_stamp

        # Normalize the updated angle to [0, π)
        self.kf.x[3] = normalize_angle(self.kf.x[3])

        if det_bbox.s is None:
            self.score = self.score * 0.01
        else:
            self.score = det_bbox.s

        cur_bbox = self.kf.x[:9].reshape(-1).tolist()
        cur_bbox = BBox.array2bbox(cur_bbox + [self.score])
        self.history[-1] = cur_bbox
        return

    def get_prediction(self, time_stamp=None):       
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        time_lag = time_stamp - self.prev_time_stamp
        self.latest_time_stamp = time_stamp
        self.kf.F = np.array([[1,0,0,0,0,0,0,time_lag,0,0],      # state transition matrix
                              [0,1,0,0,0,0,0,0,time_lag,0],
                              [0,0,1,0,0,0,0,0,0,time_lag],
                              [0,0,0,1,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])
        pred_x = self.kf.get_prediction()[0]
        if pred_x[3] >= np.pi: pred_x[3] -= np.pi * 2
        if pred_x[3] < -np.pi: pred_x[3] += np.pi * 2
        pred_bbox = BBox.array2bbox(pred_x[:9].reshape(-1))

        self.history.append(pred_bbox)
        return pred_bbox

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.history[-1]
    
    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalonobis distance
        """
        return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R
    
    def sync_time_stamp(self, time_stamp):
        self.time_stamp = time_stamp
        return