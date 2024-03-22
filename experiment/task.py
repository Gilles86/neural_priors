import argparse
import os.path as op
from psychopy.visual import Slider
from psychopy import event
from exptools2.core import PylinkEyetrackerSession, Trial
from utils import _create_stimulus_array, get_output_dir_str, DummyWaiterTrial, OutroTrial
from instruction import InstructionTrial
from stimuli import FixationLines, ResponseSlider
import numpy as np
import logging
from psychopy.visual import Line, Rect, TextStim
from session import EstimationSession

class TaskTrial(Trial):
    def __init__(self, session, trial_nr, phase_durations=None,
                jitter=1,
                stimulus_series=False,
                n=15, **kwargs):

        if phase_durations is None:
            if stimulus_series:
                phase_durations = [session.settings['durations']['first_fixation'],
                                session.settings['durations']['second_fixation'],
                                session.settings['durations']['array_duration'] / 4.,
                                session.settings['durations']['array_duration'] / 4.,
                                session.settings['durations']['array_duration'] / 4.,
                                session.settings['durations']['array_duration'] / 4.,
                                jitter,
                                session.settings['durations']['response_screen'],
                                session.settings['durations']['feedback'],
                                0.0]
            else:
                phase_durations = [session.settings['durations']['first_fixation'],
                                session.settings['durations']['second_fixation'],
                                session.settings['durations']['array_duration'],
                                jitter,
                                session.settings['durations']['response_screen'],
                                session.settings['durations']['feedback'],
                                0.0]

        self.total_duration = np.sum(phase_durations)

        self.stimulus_series = stimulus_series

        self.stimulus_phase = [2]
        self.response_phase = 4
        self.feedback_phase = 5

        if self.stimulus_series:
            self.stimulus_phase += [3, 4, 5]
            self.response_phase = 7
            self.feedback_phase = 8

        phase_names = ['fixation1', 'fixation2', 'stimulus', 'jitter', 'response', 'feedback', 'iti']

        if self.stimulus_series:
            phase_names = ['fixation1', 'fixation2', 'stimulus1', 'stimulus2', 'stimulus3', 'stimulus4', 'jitter', 'response', 'feedback', 'iti']

        super().__init__(session, trial_nr, phase_durations, phase_names=phase_names, **kwargs)

        self.parameters['n'] = n
        self.parameters['jitter'] = jitter
        self.stimulus_array = _create_stimulus_array(self.session.win, n, self.session.settings['cloud'].get('aperture_radius'), self.session.settings['cloud'].get('dot_radius'),)

        self.too_late_stimulus = TextStim(self.session.win, text='Too late!', pos=(0, 0), color=(1, -1, -1), height=0.5)
        self.parameters['start_marker_position'] = np.random.randint(self.session.settings['range'][0], self.session.settings['range'][1] + 1)

    def get_events(self):

        response_slider = self.session.response_slider

        if self.phase == (self.response_phase - 1):

            if (not self.session.mouse.getPressed()[0]) and (self.session.mouse.getPos()[0] != response_slider.marker.pos[0]):
                try:
                    self.session.mouse.setPos((response_slider.marker.pos[0],0))
                    self.last_mouse_pos = response_slider.marker.pos[0]
                except Exception as e:
                    print(e)

            self.last_mouse_pos = self.session.mouse.getPos()[0]/self.session.settings['interface']['mouse_multiplier']

        elif self.phase == self.response_phase:

            if not hasattr(self, 'response_onset'):
                current_mouse_pos = self.session.mouse.getPos()[0]/self.session.settings['interface']['mouse_multiplier']
                if np.abs(self.last_mouse_pos - current_mouse_pos) > response_slider.delta_rating_deg:
                    marker_position = response_slider.mouseToMarkerPosition(current_mouse_pos)
                    response_slider.setMarkerPosition(marker_position)
                    self.last_mouse_pos  = current_mouse_pos
                    response_slider.show_marker = True
                
                if self.session.mouse.getPressed()[0]:
                    self.response_onset = self.session.clock.getTime()
                    self.parameters['response_time'] = self.response_onset - self.session.global_log.iloc[-1]['onset']
                    self.parameters['response'] = response_slider.marker_position

                    time_so_far = self.session.clock.getTime() - self.start_trial
                    self.phase_durations[6] = self.total_duration - time_so_far - self.phase_durations[5]
                    self.stop_phase()

        super().get_events()

    def draw(self):

        if self.session.win.mouseVisible:
            self.session.win.mouseVisible = False

        if (self.phase == self.feedback_phase) & (not hasattr(self, 'response_onset')):
            self.session.fixation_lines.draw(draw_fixation_cross=False)
        else:
            self.session.fixation_lines.draw()

        response_slider = self.session.response_slider

        if self.phase == 0:
            self.session.fixation_lines.setColor((-1, .5, -1), fixation_cross_only=True)
        elif self.phase == 1:
            self.session.fixation_lines.setColor((1, -1, -1), fixation_cross_only=True)
        elif self.phase in self.stimulus_phase:

            if self.stimulus_series:
                if self.previous_phase != self.phase:
                    if self.phase == 3:
                        self.stimulus_array.xys[:, 0] *= -1
                    if self.phase == 4:
                        self.stimulus_array.xys[:, 1] *= -1
                    if self.phase == 5:
                        self.stimulus_array.xys[:, 0] *= -1

            self.stimulus_array.draw()

        elif self.phase == (self.response_phase - 1):
            response_slider.setMarkerPosition(self.parameters['start_marker_position'])
            response_slider.show_marker = False

        elif self.phase == self.response_phase:
            response_slider.marker.inner_color = self.session.settings['slider'].get('color')
            response_slider.draw()

        elif self.phase == self.feedback_phase:
            if hasattr(self, 'response_onset'):
                response_slider.marker.inner_color = self.session.settings['slider'].get('feedbackColor')
                response_slider.draw()
            else:
                self.too_late_stimulus.draw()
                    
        self.previous_phase = self.phase
class TaskSession(EstimationSession):


    def create_trials(self, include_instructions=True):
        """Create trials."""


        instruction_trial1 = InstructionTrial(self, 0, self.instructions['intro_part3'].format(run=self.settings['run']))
        instruction_trial2 = InstructionTrial(self, 0, self.instructions['intro_block'].format(range_low=self.settings['range'][0],
                                                                                               range_high=self.settings['range'][1]))

        dummy_trial = DummyWaiterTrial(self, 0, n_triggers=self.settings['mri']['n_dummy_scans'])
        
        self.trials = [instruction_trial1, instruction_trial2, dummy_trial]

        if not include_instructions:
            self.trials = self.trials[1:]

        n_trials = self.settings['task'].get('n_trials')
        range = self.settings['range']
        ns = np.random.randint(range[0], range[1] + 1, n_trials)

        possible_isis = self.settings['durations'].get('isi')
        isis = possible_isis * int(np.ceil(n_trials / len(possible_isis)))
        isis = isis[:n_trials]
        np.random.shuffle(isis)

        self.trials += [TaskTrial(self, i+1, jitter=jitter, n=n, stimulus_series=self.settings['cloud']['stimulus_series']) for i, (n, jitter) in enumerate(zip(ns, isis))]

        self.trials.append(OutroTrial(session=self))

def main(subject, session, run, range, settings='default'):

    settings_fn = op.join(op.dirname(__file__), 'settings', f'{settings}.yml')
    output_dir, output_str = get_output_dir_str(subject, session, 'estimation_task', run)

    session = TaskSession(output_str=output_str, subject=subject,
                          output_dir=output_dir, settings_file=settings_fn, 
                          run=run, range=range, eyetracker_on=False)

    session.create_trials()
    session.run()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('subject', type=str, help='Subject nr')
    argparser.add_argument('session', type=str, help='Session')
    argparser.add_argument('run', type=int, help='Run')
    argparser.add_argument('range', choices=['narrow', 'wide'], help='Range (either narrow or wide)')
    argparser.add_argument('--settings', type=str, help='Settings label', default='default')

    args = argparser.parse_args()

    main(args.subject, args.session, args.run, args.range, args.settings)