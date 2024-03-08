from psychopy import visual, core

# Create a window
win = visual.Window([800, 600], color=[1, 1, 1])


class RoundedRectangle(object):

    def __init__(self, pos, width, height, corner_radius, color):
        x, y = pos
        self.border_corners = [
            visual.Circle(win, radius=corner_radius, pos=[x - width/2 + corner_radius, y + height/2 - corner_radius], fillColor=color),
            visual.Circle(win, radius=corner_radius, pos=[x + width/2 - corner_radius, y + height/2 - corner_radius], fillColor=color),
            visual.Circle(win, radius=corner_radius, pos=[x - width/2 + corner_radius, y - height/2 + corner_radius], fillColor=color),
            visual.Circle(win, radius=corner_radius, pos=[x + width/2 - corner_radius, y - height/2 + corner_radius], fillColor=color)
        ]

        self.border_sides = [
            visual.Rect(win, width=width-2*corner_radius, height=corner_radius*2, pos=[x, y - height/2 + corner_radius], fillColor=color,),
            visual.Rect(win, width=width-2*corner_radius, height=corner_radius*2, pos=[x, y + height/2 - corner_radius], fillColor=color,),
            visual.Rect(win, width=corner_radius*2, height=height-2*corner_radius, pos=[x - width/2 + corner_radius, y], fillColor=color,),
            visual.Rect(win, width=corner_radius*2, height=height-2*corner_radius, pos=[x + width/2 - corner_radius, y], fillColor=color,)
        ]

        self.inner_rectangle = visual.Rect(win, width=width-2*corner_radius, height=height-2*corner_radius, pos=[x, y], fillColor=color)

    def draw(self):
        for shape in self.border_corners + self.border_sides:
            shape.draw()
        self.inner_rectangle.draw()

class RoundedRectangleWithBorder(object):

    def __init__(self, pos, width, height, corner_radius, inner_color, outer_color, border_thickness=0.05):
        adjusted_corner_radius = corner_radius - border_thickness
        self.outer_rectangle = RoundedRectangle(pos, width, height, corner_radius, outer_color)
        self.inner_rectangle = RoundedRectangle(pos, width-border_thickness*2, height-border_thickness*2, adjusted_corner_radius, inner_color)

    def draw(self):
        self.outer_rectangle.draw()
        self.inner_rectangle.draw()

# Create a RoundedRectangle
rect = RoundedRectangleWithBorder([0, 0], .25, 1, 0.1, [1, 1, 1], [-1, -1, -1], border_thickness=0.02)
rect.draw()

# Flip the window
win.flip()

# Wait for 5 seconds
core.wait(50.0)

# Close the window
win.close()