from psychopy.visual import Circle, Line, Rect
import numpy as np


class FixationLines(object):

    def __init__(self, win, circle_radius, color, *args, **kwargs):
        self.line1 = Line(win, start=(-circle_radius, -circle_radius),
                          end=(circle_radius, circle_radius), lineColor=color, *args, **kwargs)
        self.line2 = Line(win, start=(-circle_radius, circle_radius),
                          end=(circle_radius, -circle_radius), lineColor=color, *args, **kwargs)

    def draw(self):
        self.line1.draw()
        self.line2.draw()

    def setColor(self, color):
        self.line1.color = color
        self.line2.color = color

class RoundedRectangle(object):

    def __init__(self, win, pos, width, height, corner_radius, color):

        x, y = pos

        self.width = width
        self.height = height
        self.corner_radius = corner_radius
        self.color = color

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

class ResponseSlider(object):

    def __init__(self, win, position, length, height, color, borderColor, range, marker_position, show_marker=False,
                 borderWidth=1.,
                 markerColor=None,
                 *args, **kwargs):

        self.range = range

        if marker_position is None:
            marker_position = np.random.randint(range[0], range[1]+1) 
       
        if markerColor is None:
            markerColor = color

        self.delta_rating_deg = length / (range[1] - range[0])

        self.show_marker = show_marker

        self.bar = Rect(win, width=length, height=height, pos=position,
                        lineColor=borderColor, color=color, *args, **kwargs)

        self.marker = RoundedRectangleWithBorder(win, position, height*.5, height*1.5, height*.15, markerColor, borderColor, borderWidth=0.1)

        self.setMarkerPosition(marker_position)


    def draw(self):
        self.bar.draw()

        if self.show_marker:
            self.marker.draw()

    def setMarkerPosition(self, number):
        number = np.clip(number, self.range[0], self.range[1])
        position = self.bar.pos[0] + (number - self.range[0]) / (self.range[1] - self.range[0]) * self.bar.width - self.bar.width/2., self.bar.pos[1]
        self.marker.pos = position
        self.marker_position = number

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

