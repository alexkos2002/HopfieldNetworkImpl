import numpy as np
from PIL import Image

OUTLINE_CHAR = "$"
BACKGROUND_CHAR = "-"
SHAPE_ROWS_NUM = 5
SHAPE_COLS_NUM = 7
CYCLES_NUM = 10

BLACK_COLOR_DOT = [0, 0, 0]
WHITE_COLOR_DOT = [255, 255, 255]

test_learning_shapes_file_paths = ["test_shape1.txt", "test_shape2.txt"]
learning_shapes_file_paths = ["Ю.txt", "Б.txt", "Ж.txt"]
shapes_to_recognize_file_paths = ["Ю_змінена.txt", "Б_змінена.txt", "Ж_змінена.txt", "black_noise.txt", "white_noise.txt", "ЖтаЮНакладені.txt"]

class HopfieldNetwork:

    def __init__(self, learning_shapes_file_paths, shape_rows_num, shape_cols_num, outline_char, background_char, cycles_num):
        self.learning_shape_file_paths = learning_shapes_file_paths
        self.learning_shapes = []
        self.shape_rows_num = shape_rows_num
        self.shape_cols_num = shape_cols_num
        self.outline_char = outline_char
        self.background_char = background_char
        self.cycles_num = cycles_num
        self.shape_size = shape_rows_num * shape_cols_num
        self.weights = np.zeros((self.shape_size, self.shape_size))

    def count_shape_rows(self, shape):
        return shape.count("\n")

    def read_shape_from_file(self, file_path):
        shape = []
        print(f"Reading shape from file {file_path}.")
        with open(file_path, "r") as f:
            shape = f.read()
        print(f"Read shape is:\n{shape}")
        return shape

    def load_all_learning_shapes(self):
        for file_path in self.learning_shape_file_paths:
            shape = self.read_shape_from_file(file_path)
            cur_shape_rows_num = self.count_shape_rows(shape)
            shape = self.parse_shape_to_list(shape)
            cur_shape_cols_num = len(shape) // cur_shape_rows_num
            if not self.validate_shape_sizes(cur_shape_rows_num, cur_shape_cols_num):
                continue
            self.learning_shapes.append(shape)
            print(self.learning_shapes)

    def validate_shape_sizes(self, shape_rows_num, shape_cols_num):
        if shape_rows_num != self.shape_rows_num or shape_cols_num != self.shape_cols_num:
            print(f"Shape has incorrect num of cols or rows:\n"
                  f" ROWS_EXPECTED: {self.shape_rows_num}\n"
                  f" ROWS_ACTUAL: {shape_rows_num}\n"
                  f" COLS_EXPECTED: {self.shape_cols_num}\n"
                  f" COLS_ACTUAL: {shape_cols_num}\n")
            return False
        return True

    def parse_shape_to_list(self, shape):
        shape = shape.replace("\n", "")
        shape = shape.replace("\r", "")
        return [1 if char == OUTLINE_CHAR else -1 for char in shape]

    def calculate_wights_coef(self):
        return round(1 / self.shape_size, 3)

    def start_network_learning(self):
        N = self.calculate_wights_coef()
        for learning_shape in self.learning_shapes:
            for i in range(len(learning_shape)):
                self.weights[i] = self.weights[i] + np.multiply(np.array(learning_shape), learning_shape[i])
        self.weights = np.multiply(self.weights, N)
        print(self.weights)

    def recognize_shape(self, shape_to_rec_file_path):
        shape_to_recognize = self.read_shape_from_file(shape_to_rec_file_path)
        cur_shape_rows_num = self.count_shape_rows(shape_to_recognize)
        shape_to_recognize = self.parse_shape_to_list(shape_to_recognize)
        #memorize the shape to recognize before starting modifying it via Hopfield Network cycles
        initial_shape_to_recognize = shape_to_recognize
        cur_shape_cols_num = len(shape_to_recognize) // cur_shape_rows_num
        if not self.validate_shape_sizes(cur_shape_rows_num, cur_shape_cols_num):
            return
        cycles_counter = 0
        is_recognized = False
        while cycles_counter < self.cycles_num:
            shape_to_recognize = self.execute_one_shape_rec_cycle(shape_to_recognize)
            matchedShapeIndex = self.find_match_with_learning_shapes_for_shape(shape_to_recognize.tolist())
            if matchedShapeIndex != -1:
                is_recognized = True
                break
            cycles_counter += 1
        if (is_recognized):
            print(f"The shape from file {shape_to_rec_file_path} has been successfully recognized for {cycles_counter + 1} cycles."
                  f" It is a shape from file {self.learning_shape_file_paths[matchedShapeIndex]}")
        else:
            print(f"Can't recognize the shape from file {shape_to_rec_file_path}.")
        initial_shape_to_recognize_draw_version = np.array(initial_shape_to_recognize)
        self.draw_result(initial_shape_to_recognize_draw_version, shape_to_recognize)

    def execute_one_shape_rec_cycle(self, shape):
        shape = self.sum_function(shape)
        return self.transfer_function(shape)

    def compare_shapes(self, shape1, shape2):
        for ns1, ns2 in zip(shape1, shape2):
            if ns1 != ns2:
                return False
        return True

    def find_match_with_learning_shapes_for_shape(self, shape):
        for i in range(len(self.learning_shapes)):
            if self.compare_shapes(self.learning_shapes[i], shape):
                return i
        return -1

    def sum_function(self, shape):
        return self.weights.dot(shape)

    # transfer function is signum
    def transfer_function(self, shape):
        return np.array([-1 if shape[i] < 0 else 1 for i in range(len(shape))])

    def draw_result(self, shape_to_recognize, recognized_shape):
        shape_to_recognize = self.convert_shape_to_draw_version(shape_to_recognize)
        recognized_shape = self.convert_shape_to_draw_version(recognized_shape)
        self.draw_shape(shape_to_recognize)
        self.draw_shape(recognized_shape)

    def draw_shape(self, shape):
        img = Image.fromarray(shape)
        img.show()

    def convert_shape_to_draw_version(self, shape):
        shape = np.reshape(shape, (SHAPE_ROWS_NUM, -1))
        shape = np.array([[BLACK_COLOR_DOT if shape[i][j] == 1 else WHITE_COLOR_DOT for j in range(len(shape[i]))] for i in range(len(shape))])
        shape = shape.astype(np.uint8)
        return shape


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    hopfieldNetwork = HopfieldNetwork(learning_shapes_file_paths, SHAPE_ROWS_NUM, SHAPE_COLS_NUM, OUTLINE_CHAR,
                                      BACKGROUND_CHAR, CYCLES_NUM)
    hopfieldNetwork.load_all_learning_shapes()
    hopfieldNetwork.start_network_learning()
    hopfieldNetwork.recognize_shape(shapes_to_recognize_file_paths[5])



