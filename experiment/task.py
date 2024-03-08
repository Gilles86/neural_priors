import os.path as op
from psychopy.visual import Slider
from psychopy import event
from exptools2.core import PylinkEyetrackerSession, Trial
from utils import _create_stimulus_array
from stimuli import FixationLines, ResponseSlider
import numpy as np
import logging
from psychopy.visual import Line, Rect
from session import EstimationSession

class TaskTrial(Trial):
    def __init__(self, session, trial_nr, phase_durations=None,
                jitter=1,
                n=15, **kwargs):

        if phase_durations is None:
            phase_durations = [.3, .3, .6, jitter, 4., .5, 2.]

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        self.parameters['n'] = n
        self.stimulus_array = _create_stimulus_array(self.session.win, n, self.session.settings['cloud'].get('aperture_radius'), self.session.settings['cloud'].get( 'dot_radius'),)

    def get_events(self):
        self.session.win.mouseVisible = False
        super().get_events()

        response_slider = self.session.response_slider

        if self.phase == 3:
            self.last_mouse_pos = self.session.mouse.getPos()[0]
            response_slider.show_marker = False

        elif self.phase == 4:

            current_mouse_pos = self.session.mouse.getPos()[0]
            if np.abs(self.last_mouse_pos - current_mouse_pos) > response_slider.delta_rating_deg:
                response_slider.show_marker = True
                response_slider.setMarkerPosition(response_slider.marker_position + (current_mouse_pos - self.last_mouse_pos) / response_slider.delta_rating_deg)
                self.last_mouse_pos  = current_mouse_pos
            
            if self.session.mouse.getPressed()[0]:
                self.stop_phase()

    def draw(self):

        self.session.fixation_lines.draw()

        if self.phase == 0:
            self.session.fixation_lines.setColor((-1, 1, -1))
        elif self.phase == 1:
            self.session.fixation_lines.setColor((1, -1, -1))
        elif self.phase == 2:
            self.stimulus_array.draw()
        elif self.phase == 4:
            self.session.response_slider.draw()
        elif self.phase == 5:
            self.session.response_slider.draw()

class TaskSession(EstimationSession):


    def create_trials(self):
        """Create trials."""
        n_trials = self.settings['estimation_task'].get('n_trials_session')
        n_range = self.settings['estimation_task'].get('n_range')
        print(n_trials)
        ns = np.random.randint(n_range[0], n_range[1] + 1, n_trials)
        print(ns)
        self.trials = [TaskTrial(self, i+1, n=n) for i, n in enumerate(ns)]

def main(settings='default'):
    settings_fn = op.join(op.dirname(__file__), 'settings', f'{settings}.yml')
    session = TaskSession(output_str='test', subject='test', output_dir='test', settings_file=settings_fn, 
                          run=1, eyetracker_on=False)

    session.create_trials()
    session.run()

if __name__ == "__main__":
    main()
