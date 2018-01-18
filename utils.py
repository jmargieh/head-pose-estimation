import numpy as np


def get_image_points(shape):
    return np.array([
                            (shape[17][0], shape[17][1]),
                            (shape[21][0], shape[21][1]),
                            (shape[22][0], shape[22][1]),
                            (shape[26][0], shape[26][1]),
                            (shape[36][0], shape[36][1]),
                            (shape[39][0], shape[39][1]),
                            (shape[42][0], shape[42][1]),
                            (shape[45][0], shape[45][1]),
                            (shape[31][0], shape[31][1]),
                            (shape[30][0], shape[30][1]),
                            (shape[35][0], shape[35][1]),
                            (shape[48][0], shape[48][1]),
                            (shape[54][0], shape[54][1]),
                            (shape[57][0], shape[57][1]),
                            (shape[8][0], shape[8][1]),
                        ], dtype="double")


def get_model_points():
    return np.array([
        (6.825897, 6.760612, 4.402142),
        (1.330353, 7.122144, 6.903745),
        (-1.330353, 7.122144, 6.903745),
        (-6.825897, 6.760612, 4.402142),
        (5.311432, 5.485328, 3.987654),
        (1.789930, 5.393625, 4.413414),
        (-1.789930, 5.393625, 4.413414),
        (-5.311432, 5.485328, 3.987654),
        (2.005628, 1.409845, 6.165652),
        (0.0, 1.409845, 6.165652),
        (-2.005628, 1.409845, 6.165652),
        (2.774015, -2.080775, 5.048531),
        (-2.774015, -2.080775, 5.048531),
        (0.000000, -3.116408, 6.097667),
        (0.000000, -7.415691, 4.070434)
    ])


def get_reproject_matrix():
    proj = 10.0
    return np.array([(proj, proj, proj),
              (proj, proj, -proj),
              (proj, -proj, -proj),
              (proj, -proj, proj),
              (-proj, proj, proj),
              (-proj, proj, -proj),
              (-proj, -proj, -proj),
              (-proj, -proj, proj)])

    # return np.array([(0.0, 1.409845, 1000.0)])

