import numpy as np
from psychopy.visual import ElementArrayStim, RadialStim, Circle

def _sample_dot_positions(n=10, circle_radius=20, dot_radius=1, max_n_tries=10000):

    coords = np.zeros((0, 2))
    tries = 0

    while((coords.shape[0] < n) & (tries < max_n_tries)):
        radius = np.random.rand() * np.pi * 2
        ecc = np.random.rand() * (circle_radius - dot_radius)
        coord = np.array([[np.cos(radius), np.sin(radius)]]) * ecc

        distances = np.sqrt(((coords - coord)**2).sum(1))

        # Make the radius slightly larger
        if (distances > (dot_radius * 2) * 1.1).all():
            coords = np.vstack((coords, coord))

        tries += 1

    if tries == max_n_tries:
        raise Exception

    return coords


class RadialStimArray(object):

    def __init__(self, win, xys, sizes):
        # self.stimulus = RadialStim(win, tex='sinXsin', size=sizes, pos=[0, 0],
        #                    radialCycles=0., angularCycles=0, interpolate=False,
        #                    radialPhase=0., angularPhase=0)

        self.stimulus = Circle(win, radius=sizes, edges=128, fillColor=[1, 1, 1], lineColor=[1, 1, 1])
        self.xys = xys

    def draw(self):
        for pos in self.xys:
            self.stimulus.pos = pos
            self.stimulus.draw()

def _create_stimulus_array(win, n_dots, circle_radius, dot_radius):
    xys = _sample_dot_positions(n_dots, circle_radius, dot_radius)
    return RadialStimArray(win, xys, dot_radius)