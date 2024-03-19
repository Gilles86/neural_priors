from psychopy.visual import TextStim
from exptools2.core import Trial
import numpy as np

class InstructionTrial(Trial):

    def __init__(self, session, trial_nr, txt, bottom_txt=None, keys=None, phase_durations=None, 
                 phase_names=None, **kwargs):

        self.keys = keys

        if phase_durations is None:
            phase_durations = [.5, np.inf]

        if phase_names is None:
            phase_names = ['instruction'] * len(phase_durations)

        super().__init__(session, trial_nr, phase_durations=phase_durations, phase_names=phase_names, **kwargs)

        txt_height = self.session.settings['various'].get('text_height')
        txt_width = self.session.settings['various'].get('text_width')
        txt_color = self.session.settings['various'].get('text_color')

        self.text = TextStim(session.win, txt,
                             pos=(0.0, 0.0), height=txt_height, wrapWidth=txt_width, color=txt_color)

        if bottom_txt is None:
            bottom_txt = "Press any button to continue"

        self.text2 = TextStim(session.win, bottom_txt, pos=(
            0.0, -6.0), height=txt_height, wrapWidth=txt_width,
            color=txt_color)

    def get_events(self):

        events = Trial.get_events(self)

        if self.keys is None:
            if events:
                self.stop_phase()
        else:
            for key, t in events:
                if key in self.keys:
                    self.stop_phase()

        if self.phase > 0:
            if self.session.mouse.getPressed()[0]:
                self.stop_phase()

    def draw(self):

        if self.session.win.mouseVisible:
            self.session.win.mouseVisible = False

        self.session.fixation_lines.draw(draw_fixation_cross=False)
        self.text.draw()
        self.text2.draw()