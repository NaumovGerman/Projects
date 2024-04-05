import pymysql
from config import host, user, password, db_name


def connect():
    connection = pymysql.connect(
        host=host,
        port=3306,
        user=user,
        password=password,
        database=db_name,
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection


def show(self, x, sql_request: str):
    with x.cursor() as cursor:
        cursor.execute(sql_request)
        rows = cursor.fetchall()
        # Определение заголовков
        headers = [desc[0] for desc in cursor.description]
        # Установка заголовков в Treeview
        self.tree['columns'] = headers
        for header in headers:
            self.tree.heading(header, text=header)
            self.tree.column(header, anchor='center')  # Выравнивание по центру

        # Вставка новых данных в Treeview
        for row in rows:
            self.tree.insert('', 'end', values=tuple(row.values()))

    # Return the number of columns
    return len(headers)


