# Данная программа разработана для выполнения лабораторной №1 по предмету МРЗвИС
# Подготовил студент группы 821701 Киселёв Никита Владимирович
# Вариант №1. Реализовать модель линейной рециркуляционной сети.

import numpy
import matplotlib.pyplot
import matplotlib.image


def save_and_show(array_image, save_name):
    read_image = 1 * (array_image + 1) / 2
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.imshow(read_image)
    matplotlib.pyplot.savefig(save_name, transparent=True)
    matplotlib.pyplot.show()


def alpha(y):
    elements_sum = sum(numpy.matmul(element, element) for element in y)
    if elements_sum == 0:
        return MAX_ALPHA
    else:
        return 1 / elements_sum


def img_to_rect(image_h, image_w):
    print("1")
    rectangles = []
    for i in range(image_h // RECT_H):
        for j in range(image_w // RECT_W):
            rect = []
            for y in range(RECT_H):
                for x in range(RECT_W):
                    for color in range(3):
                        rect.append(array_image[i * RECT_H + y, j * RECT_W + x, color])
            rectangles.append(rect)
    print("dsfsdfds")
    return numpy.array(rectangles)


def training():
    image_h, image_w = numpy.size(array_image, 0), numpy.size(array_image, 1)
    rect_number = int((image_h * image_w) / (RECT_H * RECT_W))
    rectangles = img_to_rect(image_h, image_w).reshape(rect_number, 1, INPUT_LAYER)

    first_layer = numpy.random.rand(INPUT_LAYER, HIDDEN_LAYER) * 2 - 1
    temp = numpy.copy(first_layer)
    second_layer = temp.transpose()

    current_error = MAX_ERROR + 1
    iteration = 0
    print(len(rectangles))
    while current_error > MAX_ERROR:
        current_error = 0
        iteration += 1
        for rect in rectangles:
            y = rect @ first_layer
            x1 = y @ second_layer
            delta = x1 - rect
            first_layer -= alpha(y) * numpy.matmul(numpy.matmul(rect.transpose(), delta), second_layer.transpose())
            second_layer -= alpha(y) * numpy.matmul(y.transpose(), delta)
        for rect in rectangles:
            y = rect @ first_layer
            x1 = y @ second_layer
            delta = x1 - rect
            current_error += (delta * delta).sum()

        print('Iteration ', iteration, '   ', 'Error ', current_error)

    image_h = numpy.size(array_image, 0)
    image_w = numpy.size(array_image, 1)
    rect_number = int((image_h * image_w) / (RECT_H * RECT_W))

    z = (numpy.size(first_layer, 0) * rect_number) / ((numpy.size(first_layer, 0) + rect_number) * HIDDEN_LAYER + 2),
    print(' z ', z)
    return first_layer, second_layer


def rect_to_matrix(rectangles, image_h, image_w):
    matrix = []
    rect_in_line = image_w // RECT_W
    for i in range(image_h // RECT_H):
        for y in range(RECT_H):
            line = []
            for j in range(rect_in_line):
                for x in range(RECT_W):
                    dot = []
                    for color in range(3):
                        dot.append(rectangles[i * rect_in_line + j, (y * RECT_W * 3) + (x * 3) + color])
                    line.append(dot)
            matrix.append(line)
    return numpy.array(matrix)


def start(input_name, output_name):
    image_h, image_w = numpy.size(array_image, 0), numpy.size(array_image, 1)
    rect_number = int((image_h * image_w) / (RECT_H * RECT_W))
    rectangles = img_to_rect(image_h, image_w).reshape(rect_number, 1, INPUT_LAYER)
    first_layer, second_layer = training()

    result = []
    for rect in rectangles:
        result.append(rect.dot(first_layer).dot(second_layer))
    result = numpy.array(result)

    save_and_show(array_image, input_name)
    save_and_show(rect_to_matrix(result.reshape(rect_number, INPUT_LAYER), image_h, image_w), output_name)


RECT_H = 8
RECT_W = 8
HIDDEN_LAYER = 80
INPUT_LAYER = RECT_H * RECT_H * 3
MAX_ERROR = 400
MAX_ALPHA = 0.001


array_image = (2.0 * matplotlib.image.imread("image.png") / 1.0) - 1.0
start("image_input.png", "image_output.png")

array_image = (2.0 * matplotlib.image.imread("pumpkin.png") / 1.0) - 1.0
start("pumpkin_input.png", "pumpkin_output.png")

array_image = (2.0 * matplotlib.image.imread("mouse.png") / 1.0) - 1.0
start("mouse_input.png", "mouse_output.png")


array_image = (2.0 * matplotlib.image.imread("doge.png") / 1.0) - 1.0
start("doge_input.png", "doge_output.png")