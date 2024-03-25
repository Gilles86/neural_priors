from session import EstimationSession
from exptools2.core import Trial
import numpy as np
from utils import _create_stimulus_array, get_output_dir_str, OutroTrial
from psychopy.visual import TextStim
import os.path as op
import argparse
from instruction import InstructionTrial
import yaml

class FeedbackTrial(Trial):

    def __init__(self, session, trial_nr, n=15, **kwargs):

        phase_durations = [session.settings['durations']['first_fixation'],  # 0
                            session.settings['durations']['second_fixation'],# 1
                            session.settings['durations']['array_duration'], # 2
                            120,                                            # 3             
                            session.settings['durations']['feedback']]    # 4

        phase_names = ['fixation1', 'fixation2', 'stimulus', 'response', 'feedback']
        super().__init__(session, trial_nr, phase_durations, phase_names=phase_names, **kwargs)

        self.parameters['n'] = n

        aperture_radius = self.session.settings['cloud'].get('aperture_radius')
        dot_radius = self.session.settings['cloud'].get('dot_radius')

        self.stimulus_array = _create_stimulus_array(self.session.win, n, aperture_radius, dot_radius)

        text_pos = (0, self.session.response_slider.height * 1.5)

        self.n_text_stimulus = TextStim(self.session.win, text=n, pos=text_pos, color=(-1, 1, -1),
                                        height=self.session.settings['slider'].get('text_height'))

        self.start_marker_position = np.random.randint(self.session.settings['range'][0], self.session.settings['range'][1] + 1)


    def draw(self):

        response_slider = self.session.response_slider

        if self.phase == 0:
            self.session.fixation_lines.setColor((-1, .5, -1), fixation_cross_only=True)
            self.session.fixation_lines.draw()

        elif self.phase == 1:
            self.session.fixation_lines.setColor((1, -1, -1), fixation_cross_only=True)
            self.session.fixation_lines.draw()
            
        # Show stimulus
        elif self.phase == 2:
            self.session.fixation_lines.draw()
            self.stimulus_array.draw()
            response_slider.show_marker = False
            response_slider.setMarkerPosition(self.start_marker_position)

        # Show slider
        elif self.phase == 3:
            response_slider.marker.inner_color = self.session.settings['slider'].get('color')
            self.session.fixation_lines.draw()
            response_slider.draw()

        elif self.phase == 4:
            self.session.fixation_lines.draw()
            response_slider.marker.inner_color = self.session.settings['slider'].get('feedbackColor')
            response_slider.draw()
            self.n_text_stimulus.draw()

    def get_events(self):

        _ = super().get_events()

        response_slider = self.session.response_slider

        if self.phase == 2:
            if (not self.session.mouse.getPressed()[0]) and (self.session.mouse.getPos()[0] != response_slider.marker.pos[0]):
                try:
                    self.session.mouse.setPos((response_slider.marker.pos[0],0))
                    self.last_mouse_pos = response_slider.marker.pos[0]
                except Exception as e:
                    print(e)

        elif self.phase == 3:
            current_mouse_pos = self.session.mouse.getPos()[0]

            if np.abs(self.last_mouse_pos - current_mouse_pos) > response_slider.delta_rating_deg:
                self.session.response_slider.show_marker = True
                # direction = 1 if current_mouse_pos > self.last_mouse_pos else -1
                # response_slider.setMarkerPosition(response_slider.marker_position + direction)

                marker_position = response_slider.mouseToMarkerPosition(current_mouse_pos)
                response_slider.setMarkerPosition(marker_position)

                self.last_mouse_pos  = current_mouse_pos

            if self.session.mouse.getPressed()[0]:  # Check if the left mouse button is pressed
                self.parameters['response'] = response_slider.marker_position
                self.stop_phase()


class FeedbackSession(EstimationSession):

    def __init__(self, output_str, range, subject=None, output_dir=None, settings_file=None, run=None, eyetracker_on=False):
        super().__init__(output_str, range, output_dir=output_dir, settings_file=settings_file, eyetracker_on=eyetracker_on, subject=subject, run=run)
        aperture_radius = self.settings['cloud'].get('aperture_radius')
        # self.response_slider.pos = (0, -aperture_radius * 1.5)


    def create_trials(self):

        instruction_trial1 = InstructionTrial(self, 0, self.instructions['intro_part2'])
        instruction_trial2 = InstructionTrial(self, 0, self.instructions['intro_block'].format(range_low=self.settings['range'][0],
                                                                                               range_high=self.settings['range'][1]))
                                                                                               
        self.trials = [instruction_trial1, instruction_trial2]
        """Create trials."""
        n_examples = self.settings['feedback'].get('n_examples')
        ns = np.random.randint(self.settings['range'][0], self.settings['range'][1] + 1, n_examples)

        self.trials += [FeedbackTrial(self, i+1, n=n) for i, n in enumerate(ns)]

        self.trials.append(OutroTrial(session=self))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str, help='subject name', default='test')
    parser.add_argument('session', type=str, help='subject name', default='test')
    parser.add_argument('run', type=int, help='Run', default=0)
    parser.add_argument('range', choices=['narrow', 'wide'], help='Range (either narrow or wide)')
    parser.add_argument('--settings', type=str, default='default', help='Which settings to use (default=default)')

    args = parser.parse_args()
    output_dir, output_str = get_output_dir_str(args.subject, args.session, 'feedback', args.run)

    settings_fn, use_eyetracker = get_settings(args.settings)

    session = FeedbackSession(output_str=output_str,
                              subject=args.subject,
                              range=args.range,
                              eyetracker_on=use_eyetracker, output_dir=output_dir, settings_file=settings_fn, run=args.run)

    session.create_trials()
    session.run()