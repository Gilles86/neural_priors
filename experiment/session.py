from exptools2.core import PylinkEyetrackerSession, Trial
from psychopy import event
from stimuli import ResponseSlider, FixationLines
import yaml
import os.path as op

class EstimationSession(PylinkEyetrackerSession):
    def __init__(self, output_str, range, subject=None, output_dir=None, settings_file=None, run=None, eyetracker_on=False):

        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file, eyetracker_on=eyetracker_on)

        # self.win.color = (-.25, -.25, -.25)

        self.mouse = event.Mouse(visible=False)

        self.instructions = yaml.safe_load(open(op.join(op.dirname(__file__), 'instruction_texts.yml'), 'r'))

        self.settings['subject'] = subject
        self.settings['run'] = run
        self.settings['range'] = self.settings['ranges'].get(range)


        self.fixation_lines = FixationLines(self.win,
                                            self.settings['cloud'].get('aperture_radius'),
                                            color=(1, -1, -1),
                                            **self.settings['fixation_lines'])

        self._setup_response_slider()

    def _setup_response_slider(self):

        position_slider = (0, 0)
        max_range = self.settings['slider'].get('max_range')[1] - self.settings['slider'].get('max_range')[0]
        prop_max_rating = (self.settings['range'][1] - self.settings['range'][0]) / max_range
        length_line = prop_max_rating * self.settings['slider'].get('max_length')

        self.response_slider = ResponseSlider(self.win,
                                         position_slider,
                                         length_line,
                                         self.settings['slider'].get('height'),
                                         self.settings['slider'].get('color'),
                                         self.settings['slider'].get('borderColor'),
                                         self.settings['range'],
                                         marker_position=None,
                                         markerColor=self.settings['slider'].get('markerColor'),
                                         borderWidth=self.settings['slider'].get('borderWidth'),
                                         text_height=self.settings['slider'].get('text_height'))

    def run(self):
        """ Runs experiment. """
        if self.eyetracker_on:
            self.calibrate_eyetracker()

        self.start_experiment()

        if self.eyetracker_on:
            self.start_recording_eyetracker()
        for trial in self.trials:
            trial.run()

        self.close()