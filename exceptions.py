class ParametersError(Exception):
    """参数错误"""


if __name__ == '__main__':
    a = ParametersError("123")
    raise a
