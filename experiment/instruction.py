from psychopy.visual import TextStim
from exptools2.core import Trial
import numpy as np

class InstuctionTrial(Trial):

    def __init__(self, session, trial_nr, txt, bottom_txt=None, show_phase=0, keys=None, **kwargs):

        phase_durations = np.ones(12) * 1e-6
        phase_durations[show_phase] = np.inf
        self.keys = keys

        super().__init__(session, trial_nr, phase_durations=phase_durations, **kwargs)

        txt_height = self.session.settings['various'].get('text_height')
        txt_width = self.session.settings['various'].get('text_width')

        self.text = TextStim(session.win, txt,
                             pos=(0.0, 6.0), height=txt_height, wrapWidth=txt_width, color=(0, 1, 0))

        if bottom_txt is None:
            bottom_txt = "Press any button to continue"

        self.text2 = TextStim(session.win, bottom_txt, pos=(
            0.0, -6.0), height=txt_height, wrapWidth=txt_width,
            color=(0, 1, 0))

    def get_events(self):

        events = Trial.get_events(self)

        if self.keys is None:
            if events:
                self.stop_phase()
        else:
            for key, t in events:
                if key in self.keys:
                    self.stop_phase()

    def draw(self):
        if self.phase != 0:
            self.session.fixation_lines.setColor((1, -1, -1))

        if self.phase < 9:
            super().draw()
        else:
            self.session.fixation_lines.draw()
            if self.phase == 10:
                self.choice_stim.draw()
            elif self.phase == 11:
                self.certainty_stim.draw()

        self.text.draw()
        self.text2.draw()