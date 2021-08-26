# This is a sample Python script.

from typing import List

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


class DataPoint:
    height: float = 0
    weight: float = 0

    def __init__(self, weight: float, height: float):
        self.weight = weight
        self.height = height


data_set = [DataPoint(0.5, 1.4), DataPoint(2.3, 1.9), DataPoint(2.9, 3.2)]


def print_data(data: List[float]):
    for data_point in data:
        print("Weight = %f" % data_point.weight)
        print("Height = %f" % data_point.height)


def predict_height(intercept: float, slope: float, weight: float) -> float:
    """
    Assumes y = mx + b form, returns y for an assumed m, when giving x and b
    :param intercept: a float
    :param slope: a float
    :param weight: a float
    :return: the height
    """
    return intercept + slope * weight


def sum_square_residuals(data: List[DataPoint], slope: float, intercept: float) -> float:
    """
    SSR Algo
    :param data: data set
    :param slope: a float
    :param intercept: a float
    :return: the sum of square residuals
    """
    ret = 0.0
    for data_point in data:
        diff = data_point.height - predict_height(intercept, slope, data_point.weight)
        ret += diff ** 2
    return ret


def derivative_of_ssr_wrt_intercept(data: List[DataPoint], slope: float, intercept: float) -> float:
    """
    returns the slope of the sum of square residuals line for a given intercept
    :param data: data set
    :param slope: a float
    :param intercept: a float
    :return: the slope of the SSR line at a given intercept
    """
    ret = 0
    for data_point in data:
        ret += -1 * 2 * (data_point.height - (intercept + slope * data_point.weight))
    return ret


def derivative_of_ssr_wrt_slope(data: List[DataPoint], slope: float, intercept: float) -> float:
    """
    returns the slope of the sum of square residuals line for a given slope
    :param data: data set
    :param slope: a float
    :param intercept: a float
    :return: the slope of the SSR line at a given slope
    """
    ret = 0
    for data_point in data:
        ret += -1 * data_point.weight * 2 * (data_point.height - (intercept + slope * data_point.weight))
    return ret


def perform_gradient_descent(data: List[DataPoint], learning_rate: float) -> float:
    # guess the intercept
    intercept_estimate = 0
    slope_estimate = 1

    i = 0
    while True:
        intercept_step_size = derivative_of_ssr_wrt_intercept(data, slope_estimate, intercept_estimate) * learning_rate
        slope_step_size = derivative_of_ssr_wrt_slope(data, slope_estimate, intercept_estimate) * learning_rate

        intercept_estimate -= intercept_step_size
        slope_estimate -= slope_step_size

        i += 1
        if (abs(intercept_step_size) + abs(slope_step_size) < 0.00001) or (i > 1000):
            break

    return intercept_estimate, slope_estimate


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(perform_gradient_descent(data_set, 0.01))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
