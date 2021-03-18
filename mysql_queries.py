from mysql.connector import (connection)
from mysql.connector import errorcode, Error


def open_connection():
    """
    Open the connection to the MySQL database
    :return:
    """
    try:
        cnx = connection.MySQLConnection(user='root', password='', host='localhost', database='first_test_db')
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
            cursor.execute(insert_query, (feature_set_name, conclusion_dict['conclusion'],
                                          conclusion_dict['from'], conclusion_dict['to']
                                          )
                           )
        conn.commit()
        cursor.close()
    except Error as err:
        print("Something went wrong while inserting data into 'conclusions': {}".format(err))


def create_graphs(conn, feature_set_name, graph_name, attacks, font_size=30):
    insert_query = """INSERT INTO conclusions (featureset, name, edges, font_size) VALUES(%s,%s,%s,%s)"""
    try:
        cursor = conn.cursor()
        cursor.execute(insert_query, (feature_set_name, graph_name, attacks, font_size))
        conn.commit()
        cursor.close()
    except Error as err:
        print("Something went wrong while inserting data into 'conclusions': {}".format(err))


def mysql_queries_executor():
    """
    Create the MySQL queries to create a new feature-set in the
    :param ruleset:
    :param attacks:
    :return:
    """
    conn = open_connection()

    return