from psychopy.visual import Circle, Line, Rect, TextStim
import numpy as np


class FixationLines(object):

    def __init__(self, win, circle_radius, color, center_fixation_size=0.25, *args, **kwargs):

        win_size = win.size
        max_dimension = np.max(win_size)

        coord = circle_radius * 1.05 * np.cos(np.pi / 4)
        self.line1 = Line(win, start=(-coord, -coord),
                          end=(-max_dimension, -max_dimension), lineColor=color, *args, **kwargs)

        self.line2 = Line(win, start=(coord, coord),
                          end=(max_dimension, max_dimension), lineColor=color, *args, **kwargs)

        self.line3 = Line(win, start=(-coord, coord),
                          end=(-max_dimension, max_dimension), lineColor=color, *args, **kwargs)

        self.line4 = Line(win, start=(coord, -coord),
                            end=(max_dimension, -max_dimension), lineColor=color, *args, **kwargs)

        self.line5 = Line(win, start=(-center_fixation_size, -center_fixation_size),
                          end=(center_fixation_size, center_fixation_size), lineColor=color, *args, **kwargs)
        
        self.line6 = Line(win, start=(-center_fixation_size, center_fixation_size),
                            end=(center_fixation_size, -center_fixation_size), lineColor=color, *args, **kwargs)

        self.aperture = Circle(win, radius=circle_radius * 1.05, fillColor=(0, 0, 0))

        self.lines = [self.line1, self.line2, self.line3, self.line4, self.line5, self.line6]

    def draw(self, draw_fixation_cross=True):

        self.aperture.draw()
        
        if draw_fixation_cross:
            for line in self.lines:
                line.draw()
        else:
            for line in self.lines[:-2]:
                line.draw()


    def setColor(self, color):
        for line in self.lines:
            line.color = color

class RoundedRectangle(object):

    def __init__(self, win, pos, width, height, corner_radius, color):

        x, y = pos

        self.width = width
        self.height = height
        self.corner_radius = corner_radius

        self.border_corners = [
            Circle(win, radius=corner_radius, pos=[x - width/2 + corner_radius, y + height/2 - corner_radius], fillColor=color),
            Circle(win, radius=corner_radius, pos=[x + width/2 - corner_radius, y + height/2 - corner_radius], fillColor=color),
            Circle(win, radius=corner_radius, pos=[x - width/2 + corner_radius, y - height/2 + corner_radius], fillColor=color),
            Circle(win, radius=corner_radius, pos=[x + width/2 - corner_radius, y - height/2 + corner_radius], fillColor=color)
        ]

        self.border_sides = [
            Rect(win, width=width-2*corner_radius, height=corner_radius*2, pos=[x, y - height/2 + corner_radius], fillColor=color,),
            Rect(win, width=width-2*corner_radius, height=corner_radius*2, pos=[x, y + height/2 - corner_radius], fillColor=color,),
            Rect(win, width=corner_radius*2, height=height-2*corner_radius, pos=[x - width/2 + corner_radius, y], fillColor=color,),
            Rect(win, width=corner_radius*2, height=height-2*corner_radius, pos=[x + width/2 - corner_radius, y], fillColor=color,)
        ]

        self.inner_rectangle = Rect(win, width=width-2*corner_radius, height=height-2*corner_radius, pos=[x, y], fillColor=color)
        self.color = color

    def draw(self):
        for shape in self.border_corners + self.border_sides:
            shape.draw()
        self.inner_rectangle.draw()
    
    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value
        self.update_position()

    def update_position(self):

        x, y = self._pos

        self.border_corners[0].pos = [x - self.width/2 + self.corner_radius, y + self.height/2 - self.corner_radius]
        self.border_corners[1].pos = [x + self.width/2 - self.corner_radius, y + self.height/2 - self.corner_radius]
        self.border_corners[2].pos = [x - self.width/2 + self.corner_radius, y - self.height/2 + self.corner_radius]
        self.border_corners[3].pos = [x + self.width/2 - self.corner_radius, y - self.height/2 + self.corner_radius]

        self.border_sides[0].pos = [x, y - self.height/2 + self.corner_radius]
        self.border_sides[1].pos = [x, y + self.height/2 - self.corner_radius]
        self.border_sides[2].pos = [x - self.width/2 + self.corner_radius, y]
        self.border_sides[3].pos = [x + self.width/2 - self.corner_radius, y]

        self.inner_rectangle.pos = [x, y]
        self.inner_rectangle.pos = [x, y]

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        for shape in self.border_corners + self.border_sides + [self.inner_rectangle]:
            shape.fillColor = value

        self._color = value


class RoundedRectangleWithBorder(object):

    def __init__(self, win, pos, width, height, corner_radius, inner_color, outer_color, borderWidth=0.05):
        adjusted_corner_radius = corner_radius - borderWidth
        self.outer_rectangle = RoundedRectangle(win, pos, width, height, corner_radius, outer_color)
        self.inner_rectangle = RoundedRectangle(win, pos, width-borderWidth*2, height-borderWidth*2, adjusted_corner_radius, inner_color)

    def draw(self):
        self.outer_rectangle.draw()
        self.inner_rectangle.draw()

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value
        self.update_position()

    def update_position(self):
        self.outer_rectangle.pos = self._pos
        self.inner_rectangle.pos = self._pos

    # Make a property 'inner_color' that sets the color of the inner rectangle
    @property
    def inner_color(self):
        return self.inner_rectangle.color
    
    @inner_color.setter
    def inner_color(self, value):
        self.inner_rectangle.color = value


class ResponseSlider(object):

    def __init__(self, win, position, length, height, color, borderColor, range, marker_position, show_marker=False,
                 show_number=True,
                 markerColor=None,
                 *args, **kwargs):

        self.range = range
        self.height = height

        self.show_number = show_number

        if self.show_number:
            self.number = TextStim(win, text='0', pos=(position[0], position[1] - height*1.5), color=(1, 1, 1), units='deg')

        if marker_position is None:
            marker_position = np.random.randint(range[0], range[1]+1) 
       
        if markerColor is None:
            markerColor = color

        self.delta_rating_deg = length / (range[1] - range[0])

        self.show_marker = show_marker

        self.bar = Rect(win, width=length, height=height, pos=position,
                        lineColor=borderColor, color=color)

        self.marker = RoundedRectangleWithBorder(win, position, height*.5, height*1.5, height*.15, markerColor, borderColor, borderWidth=0.1)

        self.setMarkerPosition(marker_position)


    def draw(self):
        self.bar.draw()

        if self.show_marker:
            self.marker.draw()

            if self.show_number:
                self.number.text = str(self.marker_position)
                self.number.draw()

    def setMarkerPosition(self, number):
        number = np.clip(number, self.range[0], self.range[1])
        position = self.bar.pos[0] + (number - self.range[0]) / (self.range[1] - self.range[0]) * self.bar.width - self.bar.width/2., self.bar.pos[1]
        self.marker.pos = position
        self.marker_position = number

    def mouseToMarkerPosition(self, mouse_pos):
        return int((mouse_pos - self.bar.pos[0] + self.bar.width/2) / self.bar.width * (self.range[1] - self.range[0]) + self.range[0])

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value
        self.update_position()

    def update_position(self):
        self.bar.pos = self._pos
        self.setMarkerPosition(self.marker_position)

        if self.show_number:
            self.number.pos = (self._pos[0], self._pos[1] - self.bar.height*1.75)