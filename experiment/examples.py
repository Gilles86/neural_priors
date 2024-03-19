from session import EstimationSession
from exptools2.core import Trial
import numpy as np
from utils import _create_stimulus_array, get_output_dir_str
from psychopy.visual import TextStim
import os.path as op
import argparse
from instruction import InstructionTrial
import yaml

class ExampleTrial(Trial):

    def __init__(self, session, trial_nr, n=15, **kwargs):
        
        phase_durations = [1., 60.]

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        self.parameters['n'] = n

        aperture_radius = self.session.settings['cloud'].get('aperture_radius')
        dot_radius = self.session.settings['cloud'].get('dot_radius')

        self.stimulus_array = _create_stimulus_array(self.session.win, n, aperture_radius, dot_radius)

        text_pos = (0, -aperture_radius * 1.25)
        self.n_text_stimulus = TextStim(self.session.win, text=n, pos=text_pos, color=(-1, 1, -1))

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
        instruction_trial2 = InstructionTrial(self, 0, self.instructions['intro_block'].format(range_low=self.settings['range'][0],
                                                                                               range_high=self.settings['range'][1]))
        instruction_trial3 = InstructionTrial(self, 0, self.instructions['intro_part1'])

        self.trials = [instruction_trial2, instruction_trial3]

        n_examples = self.settings['examples'].get('n_examples')
        ns = np.random.randint(self.settings['range'][0], self.settings['range'][1] + 1, n_examples)

        ns[0] = self.settings['range'][0]
        ns[1] = self.settings['range'][1]

        self.trials += [ExampleTrial(self, i+1, n=n) for i, n in enumerate(ns)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str, help='Subject nr')
    parser.add_argument('session', type=str, help='Session')
    parser.add_argument('run', type=int, help='Run')
    parser.add_argument('range', choices=['narrow', 'wide'], help='Range (either narrow or wide)')
    parser.add_argument('--settings', type=str, help='Settings label', default='default')

    args = parser.parse_args()

    settings_fn = op.join(op.dirname(__file__), 'settings', f'{args.settings}.yml')
    output_dir, output_str = get_output_dir_str(args.subject, args.session, 'examples', args.run)

    session = ExampleSession(output_str=output_str, output_dir=output_dir, subject=args.subject,
                             eyetracker_on=False,
                             range=args.range,
                             settings_file=settings_fn,
                             run=args.run)
    session.create_trials()
    session.run()