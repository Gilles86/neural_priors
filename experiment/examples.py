from session import EstimationSession
from exptools2.core import Trial
import numpy as np
from utils import _create_stimulus_array
from psychopy.visual import TextStim
import os.path as op
import argparse
from instruction import InstructionTrial

class ExampleTrial(Trial):

    def __init__(self, session, trial_nr, n=15, **kwargs):
        
        phase_durations = [1., 60.]

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        self.parameters['n'] = n

        aperture_radius = self.session.settings['cloud'].get('aperture_radius')
        dot_radius = self.session.settings['cloud'].get('dot_radius')

        self.stimulus_array = _create_stimulus_array(self.session.win, n, aperture_radius, dot_radius)

        text_pos = (0, -aperture_radius * 1.15)
        self.n_text_stimulus = TextStim(self.session.win, text=f'There are {n} dots', pos=text_pos, color=(0, 1, 0))

    def draw(self):
        self.session.fixation_lines.draw()
        self.stimulus_array.draw()
        self.n_text_stimulus.draw()

    def get_events(self):
        events = super().get_events()

        if self.phase == 1:
            if events or self.session.mouse.getPressed()[0]:
                self.stop_phase()

class ExampleSession(EstimationSession):

    def create_trials(self):
        """Create trials."""

        n_examples = self.settings['example_session'].get('n_examples')
        n_range = self.settings['example_session'].get('n_range')
        ns = np.random.randint(n_range[0], n_range[1] + 1, n_examples)

        self.trials += [ExampleTrial(self, i+1, n=n) for i, n in enumerate(ns)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str, help='subject name')
    parser.add_argument('run', type=int, help='Run')
    parser.add_argument('--settings', type=str, help='Settings label', deafult='default')

    args = parser.parse_args()
    settings_fn = op.join(op.dirname(__file__), 'settings', f'{args.settings}.yml')

    session = ExampleSession(output_str='test', subject='test', eyetracker_on=False, output_dir='data', settings_file=settings_fn, run=1)
    session.create_trials()
    session.run()