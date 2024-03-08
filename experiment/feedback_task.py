from session import EstimationSession
from exptools2.core import Trial
import numpy as np
from utils import _create_stimulus_array
from psychopy.visual import TextStim
import os.path as op
import argparse

class FeedbackTrial(Trial):

    def __init__(self, session, trial_nr, n=15, **kwargs):
        
        phase_durations = [.5, 120, 1., 2.]

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        self.parameters['n'] = n

        aperture_radius = self.session.settings['cloud'].get('aperture_radius')
        dot_radius = self.session.settings['cloud'].get('dot_radius')

        self.stimulus_array = _create_stimulus_array(self.session.win, n, aperture_radius, dot_radius)

        text_pos = (0, -aperture_radius * 1.25)

        self.n_text_stimulus = TextStim(self.session.win, text=f'N = {n}', pos=text_pos, color=(1, 1, 1))


    def draw(self):

        response_slider = self.session.response_slider

        if self.phase == 0:
            response_slider.show_marker = False
            self.session.fixation_lines.draw()
            self.stimulus_array.draw()
            response_slider.draw()

        elif self.phase == 1:
            response_slider.show_marker = True
            self.session.fixation_lines.draw()
            self.stimulus_array.draw()
            response_slider.draw()

            self.n_text_stimulus.text = f'N = {response_slider.marker_position}'
            self.n_text_stimulus.draw()
        
        elif self.phase == 2:
            self.session.fixation_lines.draw()
            self.stimulus_array.draw()
            response_slider.draw()
            self.n_text_stimulus.text = f'You answered N = {response_slider.marker_position}'
            self.n_text_stimulus.draw()

        elif self.phase == 3:
            self.session.fixation_lines.draw()
            self.stimulus_array.draw()
            response_slider.draw()
            self.n_text_stimulus.text = f'There were {self.parameters["n"]} dots!'
            self.n_text_stimulus.draw()


    def get_events(self):
        events = super().get_events()

        response_slider = self.session.response_slider

        if self.phase == 0:
            self.last_mouse_pos = self.session.mouse.getPos()[0]

        elif self.phase == 1:
            current_mouse_pos = self.session.mouse.getPos()[0]
            if np.abs(self.last_mouse_pos - current_mouse_pos) > response_slider.delta_rating_deg:
                response_slider.show_marker = True
                direction = 1 if current_mouse_pos > self.last_mouse_pos else -1
                response_slider.setMarkerPosition(response_slider.marker_position + direction)
                self.last_mouse_pos  = current_mouse_pos

            if self.session.mouse.getPressed()[0]:  # Check if the left mouse button is pressed
                self.parameters['response'] = response_slider.marker_position
                self.stop_phase()


class FeedbackSession(EstimationSession):

    def __init__(self, output_str, subject=None, output_dir=None, settings_file=None, run=None, eyetracker_on=False):
        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file, eyetracker_on=eyetracker_on, subject=subject, run=run)
        aperture_radius = self.settings['cloud'].get('aperture_radius')
        self.response_slider.pos = (0, -aperture_radius * 1.5)

    def create_trials(self):
        """Create trials."""

        n_examples = self.settings['example_session'].get('n_examples')
        n_range = self.settings['example_session'].get('n_range')
        ns = np.random.randint(n_range[0], n_range[1] + 1, n_examples)

        self.trials = [FeedbackTrial(self, i+1, n=n) for i, n in enumerate(ns)]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str, help='subject name', default='test')
    parser.add_argument('run', type=int, help='Run', default=0)
    parser.add_argument('--settings', type=str, default='default', help='Which settings to use (default=default)')

    args = parser.parse_args()

    settings = args.settings
    settings_fn = op.join(op.dirname(__file__), 'settings', f'{settings}.yml')

    session = FeedbackSession(output_str=f'sub-{args.subject}_run-{args.run}',
                              subject=args.subject,
                              eyetracker_on=False, output_dir='data', settings_file=settings_fn, run=args.run)
    session.create_trials()
    session.run()