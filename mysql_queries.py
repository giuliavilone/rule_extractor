from mysql.connector import (connection)
from mysql.connector import errorcode, Error


def create_attribute_list(ruleset):
    ret = []
    for rule in ruleset:
        for item_number in range(len(rule['limits'])):
            item = rule['limits'][item_number]
            if type(item[0]) is list:
                for sub_item_number in range(len(item)):
                    sub_item = item[sub_item_number]
                    new_attribute = {'attribute': rule['columns'][sub_item_number], 'a_level': str(sub_item),
                                     'a_from': sub_item[0], 'a_to': sub_item[1]}
                    ret.append(new_attribute)
            else:
                new_attribute = {'attribute': rule['columns'][item_number], 'a_level': str(item), 'a_from': item[0],
                                 'a_to': item[1]}
                ret.append(new_attribute)
    return [i for n, i in enumerate(ret) if i not in ret[n + 1:]]


def create_conclusion_list(conclusion_list):
    ret = []
    conclusion_number = 0
    for conclusion in conclusion_list:
        new_conclusion = {'conclusion': conclusion, 'c_from': conclusion_number, 'c_to': conclusion_number}
        conclusion_number += 1
        ret.append(new_conclusion)
    return ret


def argument_formatter(argument):
    ret = ''
    for item_number in range(len(argument['limits'])):
        item = argument['limits'][item_number]
        if type(item[0]) is list:
            ret += '('
            for sub_item_number in range(len(item)):
                sub_item = item[sub_item_number]
                if sub_item_number == len(item) - 1:
                    ret += '"' + argument['columns'][sub_item_number] + ' ' + str(sub_item).replace(" ", "") + '")'
                else:
                    ret += '"' + argument['columns'][sub_item_number] + ' ' + str(sub_item).replace(" ", "") + '" AND '
            if item_number < len(argument['limits']) - 1:
                ret += " OR "
        else:
            if item_number == len(argument['limits']) - 1:
                ret += '"' + argument['columns'][item_number] + " " + str(item).replace(" ", "") + '"'
            else:
                ret += '"' + argument['columns'][item_number] + " " + str(item).replace(" ", "") + '" AND '
    return ret


def open_connection(database):
    """
    Open the connection to the MySQL database
    :return:
    """
    try:
        cnx = connection.MySQLConnection(user='root', password='', host='localhost', database=database)
        return cnx
    except Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)


def create_feature_set(conn, feature_set_name, email='giulia.vilone@tudublin.ie'):
    insert_query = """INSERT INTO user_featureset (email, featureset) VALUES(%s,%s)"""
    try:
        cursor = conn.cursor()
        cursor.execute(insert_query, (email, feature_set_name))
        conn.commit()
        cursor.close()
    except Error as err:
        print("Something went wrong while inserting data into 'user_featureset': {}".format(err))


def create_conclusions(conn, feature_set_name, conclusion_list):
    insert_query = """INSERT INTO conclusions (featureset, conclusion, c_from, c_to) VALUES(%s,%s,%s,%s)"""
    try:
        cursor = conn.cursor()
        for conclusion_dict in conclusion_list:
            cursor.execute(insert_query, (feature_set_name, str(conclusion_dict['conclusion']),
                                          float(conclusion_dict['c_from']), float(conclusion_dict['c_to'])
                                          )
                           )
        conn.commit()
        cursor.close()
    except Error as err:
        print("Something went wrong while inserting data into 'conclusions': {}".format(err))


def create_graphs(conn, feature_set_name, graph_name, attacks, font_size=30):
    insert_query = """INSERT INTO graphs (featureset, name, edges, font_size) VALUES(%s,%s,%s,%s)"""
    try:
        cursor = conn.cursor()
        attacks = str(attacks).replace("'", '"')
        attacks = attacks.replace(" ", "")
        print(attacks)
        cursor.execute(insert_query, (feature_set_name, graph_name, attacks, font_size))
        conn.commit()
        cursor.close()
    except Error as err:
        print("Something went wrong while inserting data into 'graphs': {}".format(err))


def create_attributes(conn, feature_set_name, attribute_list):
    insert_query = """INSERT INTO attributes (attribute, featureset, a_level, a_from, a_to) VALUES(%s,%s,%s,%s,%s)"""
    try:
        cursor = conn.cursor()
        for attribute in attribute_list:
            cursor.execute(insert_query, (attribute['attribute'], feature_set_name,
                                          str(attribute['a_level']).replace(" ", ""),
                                          float(attribute['a_from']), float(attribute['a_to'])
                                          )
                           )
        conn.commit()
        cursor.close()
    except Error as err:
        print("Something went wrong while inserting data into 'attributes': {}".format(err))


def create_arguments(conn, feature_set_name, graph_name, ruleset, conclusion_list):
    insert_query = """INSERT INTO arguments (argument, conclusion, x, y, label, graph, featureset) 
                      VALUES(%s,%s,%s,%s,%s,%s,%s)
                   """
    x_range = [i for i in range(10, len(ruleset) * 20 + 20, 20)]
    y_range = [j for j in range(0, len(ruleset) * 20, 20)]
    try:
        cursor = conn.cursor()
        for argument_number in range(len(ruleset)):
            argument = ruleset[argument_number]
            rule_class = int(argument['class'])
            conclusion = str(conclusion_list[rule_class]['conclusion']) \
                + " [" + str(conclusion_list[rule_class]['c_from']) + ", " + \
                str(conclusion_list[rule_class]['c_to']) + "]"
            argument_string = argument_formatter(argument)
            cursor.execute(insert_query, (argument_string, conclusion, x_range[argument_number],
                                          y_range[argument_number], argument['rule_number'], graph_name,
                                          feature_set_name
                                          )
                           )
        conn.commit()
        cursor.close()
    except Error as err:
        print("Something went wrong while inserting data into 'arguments': {}".format(err))


def mysql_queries_executor(**kwargs):
    """
    Create the MySQL queries to create a new feature-set in a database
    :param kwargs:
    :return:
    """
    conn = open_connection(kwargs['database'])
    attribute_list = create_attribute_list(kwargs['ruleset'])
    conclusion_list = create_conclusion_list(kwargs['conclusions'])
    create_feature_set(conn, kwargs['feature_set_name'])
    create_conclusions(conn, kwargs['feature_set_name'], conclusion_list)
    create_attributes(conn, kwargs['feature_set_name'], attribute_list)
    create_arguments(conn, kwargs['feature_set_name'], kwargs['graph_name'], kwargs['ruleset'], conclusion_list)
    create_graphs(conn, kwargs['feature_set_name'], kwargs['graph_name'], kwargs['attacks'])
