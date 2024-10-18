import numpy as np
from scipy.linalg import block_diag
import h5py
import scipy.io as sio


def lognormal_with_mean_cov(mean, sigma):
    # if mean=1, it return the same value of lognormal_with_mean_one(sigma)
    # mean, sigma are value of lognormal distribution
    temp = np.log(sigma ** 2 / mean ** 2 + 1)
    std_dev = np.sqrt(temp)
    mean = np.log(mean) - temp / 2
    return mean, std_dev


def lognormal_with_mean_one(percen):
    percen_v = percen ** 2
    std_dev = np.sqrt(np.log(percen_v + 1))
    mean = - std_dev ** 2 / 2
    return mean, std_dev


class model:
    def __init__(self, dataset="CMC1"):
        # basic parameters
        self.N_sensors = 4  # number of sensors
        self.x_dim = 9  # dimension of state vector
        self.z_dim = 4  # Assume all sensors have the same dimension of observation vector
        self.xv_dim = 5  # dimension of process noise
        self.zv_dim = 4  # dimension of observation noise
        self.XMAX = [2.03, 6.3]  # 2.03 5.77 6.3
        self.YMAX = [0.00, 3.41]  # [0.05 3.41];
        self.ZMAX = [0, 3]  # 5.77

        self.mode_type = ["Upright", "Fallen"]  # modes

        # param for ellipsoid plotting
        self.ellipsoid_n = 10

        # camera positions, image size and room dimensions
        self.sensor_pos = np.zeros((4, 3))
        self.sensor_pos[0] = [0.21, 3.11, 2.24]
        self.sensor_pos[1] = [7.17, 3.34, 2.16]
        self.sensor_pos[2] = [7.55, 0.47, 2.16]
        self.sensor_pos[3] = [0.21, 1.26, 2.20]
        self.imagesize = [1600, 900]
        self.room_dim = [7.67, 3.41, 2.7]

        # load camera parameters
        self.cam_mat = np.zeros((self.N_sensors, 3, 4))
        # self.cam_mat[0] = h5py.File("./cmc/cam1_cam_mat.mat", mode='r').get("cam1_cam_mat")[()].T
        # self.cam_mat[1] = h5py.File("./cmc/cam2_cam_mat.mat", mode='r').get("cam2_cam_mat")[()].T
        # self.cam_mat[2] = h5py.File("./cmc/cam3_cam_mat.mat", mode='r').get("cam3_cam_mat")[()].T
        # self.cam_mat[3] = h5py.File("./cmc/cam4_cam_mat.mat", mode='r').get("cam4_cam_mat")[()].T

        # dynamical model parameters (CV model)
        T = 1  # sampling period
        A0 = np.array([[1, T],
                       [0, 1]])  # transition matrix
        self.F = block_diag(*[np.kron(np.eye(3, dtype='f8'), A0), np.eye(3, dtype='f8')])
        n_mu0, n_std_dev0 = lognormal_with_mean_one(0.06)  # input is std dev of multiplicative lognormal noise.
        n_mu1, n_std_dev1 = lognormal_with_mean_one(0.02)

        if dataset in ["CMC1", "CMC2", "CMC3", "WILDTRACK"]:
            sigma_v, sigma_radius, sigma_heig = 0.035, n_std_dev0, n_std_dev1
            B0 = sigma_v * np.array([[(T ** 2) / 2], [T]])
            B1 = np.diag([sigma_radius, sigma_radius, sigma_heig])
            B = block_diag(*[np.kron(np.eye(3, dtype='f8'), B0), B1])
            self.Q = np.dot(B, B.T)[np.newaxis, :, :]  # process noise covariance
            self.r_birth = np.array([0.004])  # prob of birth
            self.mode = np.log(np.array([1.0]))
            self.w_birth = np.log(np.array([1.0]))  # weight of Gaussians - must be column_vector
            self.scale = 1
            n_mu_hold, n_std_dev_hold = lognormal_with_mean_one(0.1)
            self.m_birth = np.array([2.3, 0.0, 1.2, 0, 0.825, 0, (np.log(0.3)) + n_mu_hold,
                                     (np.log(0.3)) + n_mu_hold, (np.log(0.84)) + n_mu_hold])[:, np.newaxis]
            B_birth = np.diagflat([[0.25, 0.1, 0.25, 0.1, 0.15, 0.1, n_std_dev_hold, n_std_dev_hold, n_std_dev_hold]])
            self.P_birth = np.dot(B_birth, B_birth.T)[:, :, np.newaxis]  # cov of Gaussians
            # Markov transition matrix for mode 0 is standing 1 is fall, # probability of survival
            self.mode_trans_matrix = np.log(np.array([[0.99, 0.01], [0.99, 0.01]]))
            self.n_mu = np.array([[n_mu0], [n_mu1]])

            # Adaptive birth parameters
            self.tau_ru = 0.9
            self.num_det = 2
            self.rB_max = 0.001  # cap birth probability
            self.rB_min = 1e-5

        if dataset in ["CMC4", "CMC5"]:
            Q = []
            # transition for standing to standing
            self.n_mu = np.zeros((2, 3))
            self.n_mu[:, 0] = [n_mu0, n_mu1]
            sigma_vz, sigma_vxy, sigma_radius, sigma_heig = 0.035, 0.035, n_std_dev0, n_std_dev1
            B0 = np.array([[T ** 2 / 2], [T]])
            B1 = np.diag([sigma_radius, sigma_radius, sigma_heig])
            B = block_diag(*[np.kron(np.diag([sigma_vxy, sigma_vxy, sigma_vz]), B0), B1])
            Q.append(np.dot(B, B.T))  # process noise covariance

            # transition for falling to falling
            n_mu0, sigma_radius = lognormal_with_mean_one(0.4)
            n_mu1, sigma_heig = lognormal_with_mean_one(0.2)
            self.n_mu[:, 1] = [n_mu0, n_mu1]
            B1 = np.diag([sigma_radius, sigma_radius, sigma_heig])
            B = block_diag(*[np.kron(np.diag([sigma_vxy, sigma_vxy, sigma_vz]), B0), B1])
            Q.append(np.dot(B, B.T))  # process noise covariance

            # transition from standing to fallen (vice versa)
            sigma_vz = 0.07
            sigma_vxy = 0.07
            n_mu0, n_std_dev = lognormal_with_mean_one(0.1)
            self.n_mu[:, 2] = [n_mu0, 0]
            sigma_radius = n_std_dev
            sigma_heig = n_std_dev
            B1 = np.diag([sigma_radius, sigma_radius, sigma_heig])
            B = block_diag(*[np.kron(np.diag([sigma_vxy, sigma_vxy, sigma_vz]), B0), B1])
            Q.append(np.dot(B, B.T))  # process noise covariance
            self.Q = np.array(Q)

            self.r_birth = np.array([0.001])  # prob of birth
            self.mode = np.log(np.array([0.6, 0.4]))
            self.w_birth = np.log(np.array([1.0, 1.0]))  # weight of Gaussians - must be column_vector
            n_mu_hold, n_std_dev_hold = lognormal_with_mean_one(0.2)
            m_birth1 = np.array([2.3, 0, 1.2, 0, 0.825, 0, (np.log(0.3)) + n_mu_hold, (np.log(0.3)) + n_mu_hold,
                                 (np.log(0.84)) + n_mu_hold])
            m_birth2 = np.array([2.3, 0, 1.2, 0, 0.825 / 2, 0, (np.log(0.84)) + n_mu_hold, (np.log(0.84)) + n_mu_hold,
                                 (np.log(0.3)) + n_mu_hold])
            self.m_birth = np.array([m_birth1, m_birth2]).T
            B_birth = np.diagflat([[0.25, 0.1, 0.25, 0.1, 0.15, 0.1, n_std_dev_hold, n_std_dev_hold, n_std_dev_hold]])
            self.P_birth = np.array([np.dot(B_birth, B_birth.T), np.dot(B_birth, B_birth.T)]).T
            # Markov transition matrix for mode 0 is standing 1 is fall, # probability of survival
            self.mode_trans_matrix = np.log(np.array([[0.6, 0.4], [0.4, 0.6]]))

            # Adaptive birth parameters
            self.tau_ru = 0.9
            self.num_det = 2
            self.rB_max = 0.001  # cap birth probability
            self.rB_min = 1e-5

        if dataset == "CMC5":  # different calib params due to different set of recording
            self.cam_mat[2] = sio.loadmat('./cmc/cam3_cam_mat__.mat')["cam3_cam_mat__"]
            self.cam_mat[3] = sio.loadmat('./cmc/cam4_cam_mat__.mat')["cam4_cam_mat__"]

        if dataset in ["WILDTRACK"]:  # change some parameters
            # from wildtrack.wildtrack import compute_wildtrack_cam_mat
            # self.sensor_pos, self.cam_mat = compute_wildtrack_cam_mat("./wildtrack")
            self.XMAX = [-3.0, 9.0]
            self.YMAX = [-8.975, 26.9999479167]
            self.ZMAX = [-1, 5]
            self.N_sensors = 6

            sigma_v, sigma_radius, sigma_heig = 0.15, n_std_dev0, n_std_dev1
            T = 1
            B0 = sigma_v * np.array([[(T ** 2) / 2], [T]])
            B1 = np.diag([sigma_radius, sigma_radius, sigma_heig])
            B = block_diag(*[np.kron(np.eye(3, dtype='f8'), B0), B1])
            self.Q = np.dot(B, B.T)[np.newaxis, :, :]  # process noise covariance
            # n_mu_hold, n_std_dev_hold = lognormal_with_mean_one(0.5)
            # B_birth = np.diagflat([[5.5, 0.55, 5.5, 0.55, 3.15, 0.1, n_std_dev_hold, n_std_dev_hold, n_std_dev_hold]])

            self.P_births = {"pedestrian": [0.1, 0.1, 0.1, np.log(1.005), np.log(1.005), np.log(1.005)],
                             "car": [2, 2, 2, np.log(1.5), np.log(1.5), np.log(1.5)],
                             "truck": [2, 2, 2, np.log(1.5), np.log(1.5), np.log(1.5)],
                             "bus": [2, 2, 2, np.log(1.5), np.log(1.5), np.log(1.5)],
                             "trailer": [2, 2, 2, np.log(1.5), np.log(1.5), np.log(1.5)],
                             "motorcycle": [0.5, 0.5, 0.5, np.log(1.005), np.log(1.5), np.log(1.005)],
                             "bicycle": [0.5, 0.5, 0.5, np.log(1.005), np.log(1.5), np.log(1.005)]}
            _, n_std_dev_hold = lognormal_with_mean_one(0.1)
            B_birth = np.diagflat([[0.5, 0.55, 0.5, 0.55, 0.15, 0.1, n_std_dev_hold, n_std_dev_hold, n_std_dev_hold]])
            self.P_births["pedestrian"] = np.dot(B_birth, B_birth.T)[:, :, np.newaxis]
            _, n_std_dev_hold = lognormal_with_mean_one(1.5)
            B_birth = np.diagflat([[5.5, 5.55, 5.5, 5.55, 5.15, 5.5, n_std_dev_hold, n_std_dev_hold, n_std_dev_hold]])
            self.P_births["car"] = np.dot(B_birth, B_birth.T)[:, :, np.newaxis]
            self.P_births["truck"] = np.dot(B_birth, B_birth.T)[:, :, np.newaxis]
            self.P_births["bus"] = np.dot(B_birth, B_birth.T)[:, :, np.newaxis]
            self.P_births["trailer"] = np.dot(B_birth, B_birth.T)[:, :, np.newaxis]
            _, n_std_dev_hold = lognormal_with_mean_one(0.3)
            B_birth = np.diagflat([[1.5, 0, 1.5, 0, 0.15, 0, n_std_dev_hold, n_std_dev_hold, n_std_dev_hold]])
            self.P_births["motorcycle"] = np.dot(B_birth, B_birth.T)[:, :, np.newaxis]
            self.P_births["bicycle"] = np.dot(B_birth, B_birth.T)[:, :, np.newaxis]

            # self.m_birth_class = np.tile([2.3, 0.0, 1.2, 0, 0.825, 0], (7, 1))
            wlhs = {"pedestrian": np.log([0.3, 0.3, 0.84]) + n_mu_hold,
                    "car": np.log([1.48 / 2, 3.4 / 2, 1.5 / 2]) + n_mu_hold,
                    "truck": np.log([2.4 / 2, 13.6 / 2, 2.7 / 2]) + n_mu_hold,
                    "bus": np.log([2.55 / 2, 12 / 2, 3.81 / 2]) + n_mu_hold,
                    "trailer": np.log([2.48 / 2, 13.6 / 2, 2.7 / 2]) + n_mu_hold,
                    "motorcycle": np.log([0.715 / 2, 2.120 / 2, 1.080 / 2]) + n_mu_hold,
                    "bicycle": np.log([0.55 / 2, 1.75 / 2, 1.05 / 2]) + n_mu_hold}
            # locn = 5 ** 2
            self.wlhs_noise = {"pedestrian": [0.1, 0.1, 0.1, np.log(1.005), np.log(1.005), np.log(1.005)],
                               "car": [2, 2, 2, np.log(1.5), np.log(1.5), np.log(1.5)],
                               "truck": [2, 2, 2, np.log(1.5), np.log(1.5), np.log(1.5)],
                               "bus": [2, 2, 2, np.log(1.5), np.log(1.5), np.log(1.5)],
                               "trailer": [2, 2, 2, np.log(1.5), np.log(1.5), np.log(1.5)],
                               "motorcycle": [0.5, 0.5, 0.5, np.log(1.005), np.log(1.5), np.log(1.005)],
                               "bicycle": [0.5, 0.5, 0.5, np.log(1.005), np.log(1.5), np.log(1.005)]}
            for key in self.wlhs_noise:
                self.wlhs_noise[key] = np.round(self.wlhs_noise[key], 3)
            # wlhs = np.log(wlhs) + n_mu_hold
            # self.m_birth_class = np.column_stack((self.m_birth_class, wlhs))
            self.m_birth_class = {}
            for k, v in wlhs.items():
                self.m_birth_class[k] = np.append([2.3, 0.0, 1.2, 0, 0.825, 0], wlhs[k])[:, np.newaxis]

            n_mu_hold, n_std_dev_hold = lognormal_with_mean_one(0.1)
            B_birth = np.diagflat([[0.5, 0.55, 0.5, 0.55, 0.15, 0.1, n_std_dev_hold, n_std_dev_hold, n_std_dev_hold]])
            # self.P_birth = np.dot(B_birth, B_birth.T)[:, :, np.newaxis]  # cov of Gaussians
            # self.P_births = {
            #     "bicycle": [0.28021903, 0.03254728, 0.28021903, 0.03254728, 0.25416326, 0.0283209, 0.00282392,
            #                 0.02989084, 0.00133662],
            #     "bus": [0.79859858, 0.04607647, 0.79859858, 0.04607647, 0.72539383, 0.04202037, 0.03222038, 0.03222038,
            #             0.01150735],
            #     "car": [0.80430522, 0.04822918, 0.80430522, 0.04822918, 0.68016285, 0.03915121, 0.03023908, 0.03023908,
            #             0.01130043],
            #     "motorcycle": [0.28443879, 0.03421095, 0.28443879, 0.03421095, 0.24790552, 0.02783006, 0.00284326,
            #                    0.03020073, 0.00139094],
            #     "pedestrian": [0.08001821, 0.02644624, 0.08001821, 0.02644624, 0.07830736, 0.01780629, 0.00283702,
            #                    0.00283702, 0.00136064],
            #     "trailer": [0.80242215, 0.04776387, 0.80242215, 0.04776387, 0.68951711, 0.04008719, 0.0310963,
            #                 0.0310963, 0.01141934],
            #     "truck": [0.80266065, 0.04689716, 0.80266065, 0.04689716, 0.70670056, 0.0407705, 0.03144011, 0.03144011,
            #               0.01142295]}
            # for k, v in self.P_births.items():
            #     self.P_births[k] = np.diag(self.P_births[k])[:, np.newaxis]

            # Adaptive birth parameters
            self.tau_ru = 0.9
            self.num_det = 0
            self.rB_max = 0.2  # cap birth probability
            self.rB_min = 1e-25

        # survival/death parameters
        self.P_S = 0.999999999
        self.Q_S = 1 - self.P_S

        # measurement parameters
        self.meas_n_mu = np.zeros((2, 2))
        # mode 0 (standing)
        self.meas_n_mu[0, 0], meas_n_std_dev0 = lognormal_with_mean_one(0.05)
        self.meas_n_mu[1, 0], meas_n_std_dev1 = lognormal_with_mean_one(0.1)
        D0 = np.diag([20, 20, meas_n_std_dev0, meas_n_std_dev1])
        # mode 1 (fallen)
        self.meas_n_mu[0, 1], meas_n_std_dev0 = lognormal_with_mean_one(0.05)
        self.meas_n_mu[1, 1], meas_n_std_dev1 = lognormal_with_mean_one(0.1)
        D1 = np.diag([20, 20, meas_n_std_dev0, meas_n_std_dev1])
        self.R = np.array([np.dot(D0, D0.T), np.dot(D1, D1.T)])  # observation noise covariance

        # detection probabilities
        self.P_D = 0.97  # probability of detection in measurements
        self.Q_D = 1 - self.P_D  # probability of missed detection in measurements

        lambda_c = 5
        self.lambda_c = np.tile(lambda_c, (7, 1))  # poisson average rate of uniform clutter (per scan)
        self.lambda_c = np.log(self.lambda_c)
        self.range_c = np.array([[1, 1920], [1, 1024], [1, 1920], [1, 1024]], dtype='f8')  # uniform clutter region
        self.pdf_c = 1 / np.prod(self.range_c[:, 1] - self.range_c[:, 0])  # uniform clutter density
        range_temp = self.range_c[:, 1] - self.range_c[:, 0]
        range_temp[2: 4] = np.log(range_temp[2: 4])
        self.pdf_c = np.tile(1 / np.prod(range_temp), (7, 1))
        self.pdf_c = np.log(self.pdf_c)


if __name__ == '__main__':
    import pprint

    model_params = model()
    pprint.pprint(vars(model_params))
