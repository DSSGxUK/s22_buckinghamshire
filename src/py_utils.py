def safe_remove(lst, removes):
    return [x for x in lst if x not in removes]


def remove_suffix(s: str, suffix: str):
    if s.endswith(suffix):
        s = s[: -len(suffix)]
    return s

def safe_divide(x, y, default=0):
    try:
        return x / y
    except ZeroDivisionError:
        return default