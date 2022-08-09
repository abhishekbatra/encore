#!./venv/bin/python
from abc import ABC, abstractmethod
from copy import copy
from enum import Enum, auto
import fileinput
from functools import reduce
import sys


class Point:
    pass


class Point:
    """2D point with an implicit origin
    """

    def __init__(self, x=0, y=0) -> None:
        self.x = x
        self.y = y

    def __eq__(self, __o: Point) -> bool:
        return __o.x is self.x and __o.y == self.y

    def __add__(self, __o: Point) -> Point:
        return Point(self.x + __o.x, self.y + __o.y)

    def __sub__(self, __o: Point) -> Point:
        return Point(self.x - __o.x, self.y - __o.y)


class TetrisObject:
    pass


class World(TetrisObject):
    pass


class TetrisObject(ABC):
    """Base class for renderable objects in the `World`
    """

    def __init__(self, parent: TetrisObject = None, pos: Point = None) -> None:
        """

        Args:
            parent (TetrisObject, optional): Defaults to None.
            pos (Point, optional): position relative to `parent`. Defaults to Point(0, 0).
        """
        self.parent = parent
        self.pos = pos or Point()

    @property
    def pos_in_world_coord(self):
        """transform position to world coordinates

        Returns:
            Point: position in world coordinates
        """
        current_pos = copy(self.pos)
        parent: TetrisObject = self.parent
        world = World.get_world_instance()
        while parent is not world:
            current_pos = current_pos + parent.pos
            parent = parent.parent

        return current_pos


class Unit(TetrisObject):
    """class signifying the unit block in the game
    """

    def __init__(self, parent: TetrisObject, pos: Point = None) -> None:
        super().__init__(parent, pos)


UnitRow = list[Unit]


class Shape(TetrisObject):
    """A composition of `Unit` blocks that decsribe a tetris shape
    """

    def __init__(self, parent: TetrisObject) -> None:
        super().__init__(parent)
        self.unit_rows: list[UnitRow] = []
        self.init_units()

    @abstractmethod
    def init_units(self):
        pass

    @property
    def top_most_y(self) -> int:
        """Gets the y coordinate of the topmost unit in world coordinates

        Returns:
            int: y coordinate of the topmost unit in world coordinates
        """
        if len(self.unit_rows) > 0:
            unit: Unit = self.unit_rows[-1][0]
            return unit.pos_in_world_coord.y
        else:
            return 0

    def find_unit_with_pos(self, pos: Point) -> Unit:
        """finds unit with given position in world coordinates

        Args:
            pos (Point): position of unit in world coordinates

        Returns:
            Unit: 
        """
        unit_row: UnitRow
        unit: Unit

        for unit_row in self.unit_rows:
            for unit in unit_row:
                if unit.pos_in_world_coord == pos:
                    return unit

        return None

    def delete_row_at_y(self, unit_y: int):
        """deletes a row of units from the shape

        Args:
            unit_y (int): world origin y coordinate for row to delete
        """
        # This can probably accept unit's y relative to self, since unit contains that information
        unit_row_found = False
        unit_row_index = 0
        while unit_row_found is False:
            row = self.unit_rows[unit_row_index]
            first_unit = row[0]
            if first_unit.pos_in_world_coord.y == unit_y:
                unit_row_found = True
            else:
                unit_row_index += 1

        if unit_row_found is True:
            unit: Unit
            unit_row = self.unit_rows[unit_row_index]
            for unit in unit_row:
                del unit
            del self.unit_rows[unit_row_index]

            for i in range(unit_row_index, len(self.unit_rows)):
                unit_row = self.unit_rows[i]
                for unit in unit_row:
                    unit.pos.y -= 1


class CollisionDetecter:
    def do_shapes_collide(shape1: Shape, shape2: Shape) -> bool:
        """returns `True` if at least on of the units of the shape are coincident

        Args:
            shape1 (Shape): 
            shape2 (Shape): 

        Returns:
            bool: 
        """
        if shape1 is None or shape2 is None or shape1 is shape2:
            return False

        unit_row: UnitRow
        unit: Unit

        for unit_row in shape2.unit_rows:
            for unit in unit_row:
                if shape1.find_unit_with_pos(unit.pos_in_world_coord) is not None:
                    return True

        return False


class ShapeQ(Shape):
    def __init__(self, parent: TetrisObject) -> None:
        super().__init__(parent)

    def init_units(self):
        self.unit_rows.append([Unit(self, Point()), Unit(self, Point(1, 0))])
        self.unit_rows.append(
            [Unit(self, Point(0, 1)), Unit(self, Point(1, 1))])


class ShapeZ(Shape):
    def __init__(self, parent: TetrisObject) -> None:
        super().__init__(parent)

    def init_units(self):
        self.unit_rows.append(
            [Unit(self, Point(1, 0)), Unit(self, Point(2, 0))])
        self.unit_rows.append(
            [Unit(self, Point(0, 1)), Unit(self, Point(1, 1))])


class ShapeS(Shape):
    def __init__(self, parent: TetrisObject) -> None:
        super().__init__(parent)

    def init_units(self):
        self.unit_rows.append(
            [Unit(self, Point(0, 0)), Unit(self, Point(1, 0))])
        self.unit_rows.append(
            [Unit(self, Point(1, 1)), Unit(self, Point(2, 1))])


class ShapeT(Shape):
    def __init__(self, parent: TetrisObject) -> None:
        super().__init__(parent)

    def init_units(self):
        self.unit_rows.append([Unit(self, Point(1, 0))])
        self.unit_rows.append([Unit(self, Point(0, 1)), Unit(
            self, Point(1, 1)), Unit(self, Point(2, 1))])


class ShapeI(Shape):
    def __init__(self, parent: TetrisObject) -> None:
        super().__init__(parent)

    def init_units(self):
        self.unit_rows.append([Unit(self, Point(0, 0)), Unit(
            self, Point(1, 0)), Unit(self, Point(2, 0)), Unit(self, Point(3, 0))])


class ShapeL(Shape):
    def __init__(self, parent: TetrisObject) -> None:
        super().__init__(parent)

    def init_units(self):
        self.unit_rows.append(
            [Unit(self, Point(0, 0)), Unit(self, Point(1, 0))])
        self.unit_rows.append([Unit(self, Point(0, 1))])
        self.unit_rows.append([Unit(self, Point(0, 2))])


class ShapeJ(Shape):
    def __init__(self, parent: TetrisObject) -> None:
        super().__init__(parent)

    def init_units(self):
        self.unit_rows.append(
            [Unit(self, Point(0, 0)), Unit(self, Point(1, 0))])
        self.unit_rows.append([Unit(self, Point(1, 1))])
        self.unit_rows.append([Unit(self, Point(1, 2))])


class ShapeType(Enum):
    Q = auto()
    Z = auto()
    S = auto()
    T = auto()
    I = auto()
    L = auto()
    J = auto()


ShapeFactory: dict[ShapeType, type] = {
    ShapeType.Q: ShapeQ,
    ShapeType.Z: ShapeZ,
    ShapeType.S: ShapeS,
    ShapeType.T: ShapeT,
    ShapeType.I: ShapeI,
    ShapeType.L: ShapeL,
    ShapeType.J: ShapeJ,
}


class World(TetrisObject):
    """Singleton world canvas that contains all renderables
    """
    VICINITY_THRESHOLD_X = 3
    VICINITY_THRESHOLD_Y = 2
    MAX_HEIGHT = 100
    __WorldInstance: World = None
    WORLD_WIDTH = 10

    def get_world_instance() -> World:
        if World.__WorldInstance is None:
            World.__WorldInstance = World()
        return World.__WorldInstance

    def reset_world_instance():
        World.__WorldInstance = None

    def __init__(self) -> None:
        super().__init__()
        self.shapes: list[Shape] = []
        self.net_height = 0
        # first index signifies y, second index x
        self.units_matrix: dict[int, dict[int, Unit]] = {}

    def is_shape_in_vicinity(shape: Shape, pos: Point) -> bool:
        """checks if `shape` is within threshold distance from `pos` such that a collision is possible

        Args:
            shape (Shape): 
            pos (Point): 

        Returns:
            bool: 
        """
        is_left = (shape.pos.x + World.VICINITY_THRESHOLD_X >=
                   pos.x) and shape.pos.y + World.VICINITY_THRESHOLD_Y >= pos.y
        is_right = (shape.pos.x - World.VICINITY_THRESHOLD_X <=
                    pos.x and shape.pos.y + World.VICINITY_THRESHOLD_Y >= pos.y)

        return is_left or is_right

    def find_shapes_near(self, pos: Point) -> list[Shape]:
        """gets all shapes in vicinity of `pos`

        Args:
            pos (Point): 

        Returns:
            list[Shape]: 
        """
        return [shape for shape in self.shapes if World.is_shape_in_vicinity(shape, pos)]

    def find_top_unit_at_x(self, x: int, shadow_parent: TetrisObject) -> Unit:
        """finds the `unit` with largest `y` coordinate with x coordinate == `x`

        Args:
            x (int): 
            shadow_parent (TetrisObject): `parent` whose units are to be excluded from search

        Returns:
            Unit: 
        """
        top_unit: Unit = None
        units: dict[int, Unit]
        for y, units in self.units_matrix.items():
            unit_at_x = units.get(x)
            if unit_at_x is not None and top_unit is not None and unit_at_x.pos_in_world_coord.y > top_unit.pos_in_world_coord.y and unit_at_x.parent is not shadow_parent:
                top_unit = unit_at_x
            elif unit_at_x is not None and top_unit is None and unit_at_x.parent is not shadow_parent:
                top_unit = unit_at_x

        return top_unit

    def place_shape(self, shape: Shape, col: int) -> bool:
        """Places shape in free space according to specified column

        Args:
            shapeType (ShapeType): Type of shape to create
            col (int): specifies the x coordinate

        Returns:
            bool: `True` if placement succeeded else `False`
        """
        placed = False
        pos = Point(col, 0)

        top_unit_at_x: Unit = self.find_top_unit_at_x(col, shape)
        if top_unit_at_x is not None and top_unit_at_x.pos_in_world_coord.y > 0:
            pos.y = top_unit_at_x.pos_in_world_coord.y - 1

        current_y = 0

        while placed is False and current_y < World.MAX_HEIGHT:
            nearby_shapes: list[Shape] = self.find_shapes_near(pos)
            shape.pos = pos
            collision_detected = reduce(lambda prev, s: prev or CollisionDetecter.do_shapes_collide(
                shape, s), nearby_shapes, False)

            if collision_detected:
                current_y += 1
                pos.y = current_y
            else:
                placed = True
                self.net_height = max(
                    self.net_height, shape.top_most_y + 1)

        return placed

    def store_units_in_cache(self, shape: Shape):
        """store references to individual units for faster lookup

        Args:
            shape (Shape): 
        """
        unit_row: UnitRow
        for unit_row in shape.unit_rows:
            row_index = unit_row[0].pos_in_world_coord.y
            unit: Unit
            for unit in unit_row:
                row_cache = self.units_matrix.get(row_index, {})
                row_cache[unit.pos_in_world_coord.x] = unit
                self.units_matrix[row_index] = row_cache

    def get_row_to_delete(self) -> int:
        """Finds first filled row from bottom to delete

        Returns:
            int: index of row to delete or -1 if none found.
        """
        row_index: int
        row: dict[int, Unit]

        for row_index, row in self.units_matrix.items():
            gap_found = False
            for i in range(0, World.WORLD_WIDTH):
                unit = row.get(i)
                if unit is None:
                    gap_found = True
            if gap_found is False:
                return row_index
        return -1

    def delete_row(self, row_index: int):
        """deletes a row of units

        Args:
            row_index (int): `y` coordinate of the row of units to delete
        """
        parent: Shape = None
        next_parent: Shape = None
        unit_row: dict[int, Unit] = self.units_matrix[row_index]
        unit: Unit
        unit_x: int

        for unit_x in unit_row:
            unit = unit_row[unit_x]
            next_parent = unit.parent

            if next_parent is not parent:
                next_parent.delete_row_at_y(row_index)
                parent = next_parent

        self.units_matrix = {}

        self.net_height -= 1
        shape: Shape
        for shape in self.shapes:
            self.place_shape(shape, shape.pos.x)
            self.store_units_in_cache(shape)


class TetrisGame:
    StrToShapeTypeMap = dict(
        Q=ShapeType.Q,
        Z=ShapeType.Z,
        S=ShapeType.S,
        T=ShapeType.T,
        I=ShapeType.I,
        L=ShapeType.L,
        J=ShapeType.J,
    )

    def __init__(self) -> None:
        self.world: World = World.get_world_instance()

    def __del__(self):
        self.world = None
        World.reset_world_instance()

    def push_new_shape(self, shapeType: ShapeType, col: int):
        new_shape_cls = ShapeFactory.get(shapeType)
        if new_shape_cls is not None:
            new_shape: Shape = new_shape_cls(self.world)
            placed = self.world.place_shape(new_shape, col)

            if placed is False:
                return World.MAX_HEIGHT
            else:
                self.world.shapes.append(new_shape)

            self.world.store_units_in_cache(new_shape)
            more_rows_to_delete = True

            while more_rows_to_delete:
                row_to_delete = self.world.get_row_to_delete()
                if row_to_delete == -1:
                    more_rows_to_delete = False
                else:
                    self.world.delete_row(row_to_delete)


class Driver:
    def process_input_line(self, line: str):
        tetris_game = TetrisGame()
        words = line.split(',')
        for word in words:
            self.process_word(word, tetris_game)
        sys.stdout.write(str(tetris_game.world.net_height) + "\n")

    def process_word(self, word: str, game: TetrisGame):
        shape_str = word[0]
        shape_type: ShapeType = game.StrToShapeTypeMap[shape_str]
        col: int = int(word[1:])
        game.push_new_shape(shape_type, col)

    def run(self):
        for line in fileinput.input():
            self.process_input_line(line.rstrip())


if __name__ == '__main__':
    driver = Driver()
    driver.run()
    # driver.process_input_line("Q0,Q2,Q4,Q6,Q8")
