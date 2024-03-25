import numpy as np
from psychopy.visual import ElementArrayStim, RadialStim, Circle
import os.path as op
import logging
from exptools2.core import Trial
import yaml
from instruction import InstructionTrial

def _sample_dot_positions(n=10, circle_radius=20, dot_radius=1, min_ecc=0.2, max_n_tries=10000):

    coords = np.zeros((0, 2))
    tries = 0

    while((coords.shape[0] < n) & (tries < max_n_tries)):
        radius = np.random.rand() * np.pi * 2
        min_ecc_frac = min_ecc / circle_radius
        ecc = np.sqrt((np.random.rand() + min_ecc_frac) / (1.+min_ecc_frac))  * (circle_radius - dot_radius)
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

        self.stimulus = Circle(win, radius=sizes, edges=128, fillColor=[1, 1, 1])
        self.xys = xys

    def draw(self):
        for pos in self.xys:
            self.stimulus.pos = pos
            self.stimulus.draw()

def _create_stimulus_array(win, n_dots, circle_radius, dot_radius):
    xys = _sample_dot_positions(n_dots, circle_radius, dot_radius)
    return RadialStimArray(win, xys, dot_radius)


def get_output_dir_str(subject, session, task, run):
    output_dir = op.join(op.dirname(__file__), 'logs', f'sub-{subject}')
    logging.warn(f'Writing results to  {output_dir}')

    if session:
        output_dir = op.join(output_dir, f'ses-{session}')
        output_str = f'sub-{subject}_ses-{session}_task-{task}'
    else:
        output_str = f'sub-{subject}_task-{task}'

    if run:
        output_str += f'_run-{run}'

    return output_dir, output_str

class DummyWaiterTrial(InstructionTrial):
    """ Simple trial with text (trial x) and fixation. """

    def __init__(self, session, trial_nr, phase_durations=None, n_triggers=1,
                 txt="Waiting for scanner triggers.", **kwargs):

        phase_durations = [np.inf] * n_triggers
        phase_names = [f'trigger_{n+1}' for n in range(n_triggers)]

        super().__init__(session, trial_nr, txt=txt, phase_durations=phase_durations,
                         bottom_txt='', 
                         phase_names=phase_names,
                         **kwargs)

        self.last_trigger = 0.0

    def get_events(self):
        events = Trial.get_events(self)

        if events:
            for key, t in events:
                if key == self.session.mri_trigger:
                    if t - self.last_trigger > .5:
                        self.stop_phase()
                        self.last_trigger = t
class OutroTrial(InstructionTrial): 
    """ Simple trial with only fixation cross.  """

    def __init__(self, session, trial_nr=0, phase_durations=None, **kwargs):

        txt = '''Please lie still for a few moments.'''

        if phase_durations is None:
            phase_durations = [5*60]

        super().__init__(session=session, trial_nr=trial_nr, phase_durations=phase_durations, txt=txt,
                         bottom_txt='', 
                         phase_names=['outro'],
                         **kwargs)

    def draw(self):
        self.session.fixation_lines.draw()
        super().draw()

    def get_events(self):
        events = Trial.get_events(self)

        if events:
            for key, t in events:
                if key == 'space':
                    self.stop_phase()


def get_settings(settings):
    settings_fn = op.join(op.dirname(__file__), 'settings', f'{settings}.yml')
    print(settings_fn)

    with open(settings_fn, 'r') as f:
        settings = yaml.safe_load(f)

    use_eyetracker = 'eyetracker' in settings.keys()

    return settings_fn, use_eyetracker