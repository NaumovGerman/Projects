# Кредиты:
credits_command_first = """
        SELECT *
        FROM credits"""


def credits_command_second(param):
    return f"""
        SELECT credit_id, loan_amount, loan_term
        FROM credits
        WHERE loan_amount >= {param}
        ORDER BY loan_amount desc
        """


credits_command_third = """
        SELECT cr.*
        FROM credits AS cr
        LEFT JOIN credit_statuses AS cs ON cr.credit_id = cs.credit_id
        WHERE cs.status_id = 0
        """
credits_command_fourth = """
        with temp as (
            select credit_id, sum(payment_amount) as overall
            from payments
            GROUP BY credit_id
        )
        SELECT cr.*, (cr.loan_amount - t.overall) as remain
        FROM credits AS cr
        JOIN temp AS t on cr.credit_id = t.credit_id
        """


def credits_command_fifth(param):
    return f"""
        SELECT distinct cr.*
        FROM payments as pm
        LEFT JOIN credits as cr on cr.credit_id = pm.credit_id
        where cr.credit_id in (
            SELECT credit_id
            FROM payments
            WHERE curdate() - payment_date >= {param}
        )
        """


def credits_command_sixth(param):
    return f"""
        SELECT *
        FROM credits
        WHERE loan_term > {param}
        """


# Клиенты:
clients_command_first = """
        SELECT *
        FROM clients
        """
clients_command_second = """
        with temp as (
            SELECT cc.*, cl.last_name, cl.first_name, cr.loan_amount
            FROM credit_contracts as cc
            LEFT JOIN credits as cr on cc.credit_id = cr.credit_id
            LEFT JOIN clients as cl on cc.client_id = cl.client_id
        )
        SELECT t.client_id, t.first_name, t.last_name, sum(t.loan_amount) as total_loan
        FROM temp as t
        GROUP BY t.client_id
        ORDER BY t.client_id
        """


def clients_command_third(param):
    return f"""
        SELECT *
        FROM clients
        WHERE address = '{param}'
        """


clients_command_fourth = """
        with temp as (
            SELECT credit_id, sum(payment_amount) as overall
            FROM payments
            GROUP BY credit_id
        )
        SELECT cl.*, sum(t.overall) as total
        FROM credits as cr
        LEFT JOIN temp as t on cr.credit_id = t.credit_id
        LEFT JOIN clients_credits as cc on cr.credit_id = cc.credit_id
        LEFT JOIN clients as cl on cc.client_id = cl.client_id
        GROUP BY cl.client_id
        ORDER BY cl.client_id
        """
clients_command_fifth = """
        SELECT cl.client_id, cl.first_name, cl.last_name, sum(cs.status_id) as number
        FROM clients_credits as cc
        LEFT JOIN credit_statuses as cs on cc.credit_id = cs.credit_id
        LEFT JOIN clients as cl on cc.client_id = cl.client_id
        GROUP BY client_id
        ORDER BY client_id
        """


def clients_command_sixth(param):
        first_name = param.split()[0]
        last_name = param.split()[1]
        return f"""
        with temp as (
            SELECT client_id, count(credit_id) as num
            FROM clients_credits
            GROUP BY client_id
        )
        SELECT cl.client_id, cl.first_name, cl.last_name, t.num
        FROM clients as cl
        LEFT JOIN temp as t on cl.client_id = t.client_id
        WHERE cl.first_name = '{first_name}' AND cl.last_name = '{last_name}'
        """


# Сотрудники :
employees_command_first = """
        SELECT *
        FROM bank_employees
        """
employees_command_second = """
        select be.*, SUM(cr.loan_amount) AS total
        FROM credit_contracts as cc
        LEFT JOIN bank_employees as be on cc.employee_id = be.employee_id
        LEFT JOIN credits as cr on cc.credit_id = cr.credit_id
        group by be.employee_id
        ORDER BY be.employee_id
        """
employees_command_third = """
        SELECT be.employee_id, be.first_name AS emp_name, be.last_name AS emp_surname, cl.client_id, cl.first_name AS client_name, cl.last_name AS client_surname
        FROM credit_contracts AS cc
        LEFT JOIN bank_employees AS be ON cc.employee_id = be.employee_id
        LEFT JOIN clients AS cl ON cc.client_id = cl.client_id
        ORDER BY be.employee_id
        """


def employees_command_fourth(param):
    return f"""
        SELECT *
        FROM bank_employees
        WHERE position = '{param}'
        """

def employees_command_fifth(param):
    return f"""
        SELECT *, (curdate() - employment_date) as experience
        FROM bank_employees
        WHERE (curdate() - employment_date) > {param}
        """


# Договоры :
credit_contracts_first = "SELECT * FROM `credit_contracts`"
credit_contracts_second = """
    SELECT
        cc.contract_id,
        c.first_name AS client_first_name,
        c.last_name AS client_last_name,
        e.first_name AS employee_first_name,
        e.last_name AS employee_last_name
    FROM
        credit_contracts cc
    JOIN
        clients c ON cc.client_id = c.client_id
    JOIN
        bank_employees e ON cc.employee_id = e.employee_id
    ORDER BY cc.contract_id;
    """

credit_contracts_third = """
    SELECT
        cc.contract_id,
        cc.requested_amount,
        c.loan_amount
    FROM
        credit_contracts cc
    JOIN
        credits c ON cc.credit_id = c.credit_id
    WHERE
        c.loan_amount < cc.requested_amount;
    """


def credit_contracts_fourth(date):
    return f"""
            SELECT 
                cc.contract_id,
                cl.first_name AS client_first_name,
                cl.last_name AS client_last_name,
                be.first_name AS employee_first_name,
                be.last_name AS employee_last_name,
                cc.contract_date
            FROM credit_contracts as cc
            JOIN bank_employees as be ON cc.employee_id = be.employee_id
            JOIN clients as cl on cc.client_id = cl.client_id
            WHERE cc.contract_date > '{date}';
            """


def receive_client_id(full_name):
    name = full_name.split()[0]
    surname = full_name.split()[1]
    return f"""
            select client_id
            from clients
            where first_name = '{name}' and last_name = '{surname}'
            """


def receive_emp_id(full_name):
    name = full_name.split()[0]
    surname = full_name.split()[1]
    return f"""
            select employee_id
            from bank_employees
            where first_name = '{name}' and last_name = '{surname}'
            """


def insert_into_cs(num_rows: int):
    num_rows = num_rows + 1
    return f"""
            insert into credit_statuses
            values ({num_rows}, 0);
            """


def insert_into_cv(num_rows: int, loan: int, rate: int, period: int):
    int_rate = int(rate) / 100
    num_rows += 1
    return f"""
            insert into credits
            values ({num_rows}, {int(loan) - 1000}, {int_rate}, {period});
            """


def insert_into_clcr(num_rows: int, client_id: int):
    num_rows += 1
    return f"""
            insert into clients_credits
            values ({num_rows}, {client_id}, {num_rows});
            """


def insert_into_crco(num_rows: int, client_id: int, emp_id: int, loan: int, type: int):
    num_rows += 1
    return f"""
            insert into credit_contracts
            values ({num_rows}, {loan}, curdate(), {emp_id}, {type}, {client_id}, {num_rows});
            """


# Платежи :
payments_command_first = "SELECT * FROM `payments`"


def payments_command_second(date_str):
    return f"""
        with temp as (
            SELECT *
            FROM payments
            WHERE payment_date > '{date_str}'
        )
        SELECT credit_id, sum(t.payment_amount) as total
        FROM temp as t
        GROUP BY t.credit_id
        ORDER BY t.credit_id
    """


def payments_command_third(amount):
    return f"""
    SELECT *
    FROM 
        payments
    WHERE 
        payment_amount > {amount};
    """


payments_command_fourth = """SELECT
    p.payment_id,
    cl.first_name,
    cl.last_name,
    p.payment_amount,
    p.payment_date
    FROM
        payments p
    JOIN
        credits c ON p.credit_id = c.credit_id
    JOIN
        clients_credits cc ON c.credit_id = cc.credit_id
    JOIN
        clients cl ON cc.client_id = cl.client_id;
    """


def payments_command_fifth(name):
    first_name = name.split()[0]
    last_name = name.split()[1]
    return f"""
        SELECT p.*
        FROM payments p
        JOIN credits c ON p.credit_id = c.credit_id
        JOIN clients_credits cc ON c.credit_id = cc.credit_id
        JOIN clients cl ON cc.client_id = cl.client_id
        WHERE cl.first_name = '{first_name}' AND cl.last_name = '{last_name}';
        """





