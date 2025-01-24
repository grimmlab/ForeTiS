def result_string_to_dictionary(result_string: str) -> dict:
    """
    Convert result string saved in a .csv file to a dictionary

    :param result_string: string from .csv file

    :return: dictionary with info from .csv file
    """
    key_value_strings = result_string.split('\\')[0][2:-2].replace('\'', '').split(',')
    dict_result = {}
    for key_value_string in key_value_strings:
        key = key_value_string.split(':')[0].strip()
        value = key_value_string.split(':')[1].strip()
        try:
            value = float(value)
            value = int(value) if value == int(value) else value
        except:
            value = value
        dict_result[key] = value
    return dict_result