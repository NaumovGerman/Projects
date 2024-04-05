import tkinter
from tkinter import *
from tkinter import ttk
import customtkinter as ctk
from main import show, connect
import commands_sql as fn

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("dark-blue")

x = connect()


class App(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("App")
        self.geometry(f'{self.winfo_screenwidth()}x{self.winfo_screenheight()}-6-0')
        # self.geometry('1200x700')

        # Инициализация состояний
        self.init_states()

        # Инициализация в первом состоянии
        self.show_menu()

    def init_states(self):
        # Фрейм - основное меню
        self.menu_frame = Frame(self)
        self.init_menu_frame()

        # Фрейм - "Кредиты"
        self.credit_state_frame = Frame(self)
        self.init_credit_state_frame()

        # Фрейм - "Кредиты" - выкладки
        self.credit2_outlier_state_frame = Frame(self)
        self.init_credit2_outlier_state_frame()

        self.credit5_outlier_state_frame = Frame(self)
        self.init_credit5_outlier_state_frame()

        self.credit6_outlier_state_frame = Frame(self)
        self.init_credit6_outlier_state_frame()

        # Фрейм - "Клиенты"
        self.client_state_frame = Frame(self)
        self.init_client_state_frame()

        # Фрейм - "Клиенты" - выкладки
        self.client3_outlier_state_frame = Frame(self)
        self.init_client3_outlier_state_frame()

        self.client6_outlier_state_frame = Frame(self)
        self.init_client6_outlier_state_frame()

        # Фрейм - "Сотрудники"
        self.employee_state_frame = Frame(self)
        self.init_employee_state_frame()

        # Фрейм - "Сотрудники" - выкладки
        self.employee4_outlier_state_frame = Frame(self)
        self.init_employee4_outlier_state_frame()

        self.employee5_outlier_state_frame = Frame(self)
        self.init_employee5_outlier_state_frame()

        # Фрейм - "Договоры"
        self.contracts_state_frame = Frame(self)
        self.init_contracts_state_frame()

        # Фрейм - "Сотрудники" - выкладки
        self.contracts4_outlier_state_frame = Frame(self)
        self.init_contracts4_outlier_state_frame()

        # Фрейм - "Платежи"
        self.payments_state_frame = Frame(self)
        self.init_payments_state_frame()

        # Фрейм - "Платежи" - выкладки
        self.payment2_outlier_state_frame = Frame(self)
        self.init_payment2_outlier_state_frame()

        self.payment3_outlier_state_frame = Frame(self)
        self.init_payment3_outlier_state_frame()

        self.payment6_outlier_state_frame = Frame(self)
        self.init_payment6_outlier_state_frame()

        # Фрейм для четвертого состояния (результат show)
        self.result_state_frame = Frame(self)
        self.init_result_state_frame()

        # Обработка закрытия окна
        self.protocol("WM_DELETE_WINDOW", lambda: self.on_closing(x))

    def init_menu_frame(self):
        # Кнопка "Кредиты"
        self.credit_button = ctk.CTkButton(self.menu_frame, text="Кредиты",
                                           command=self.show_credit_state)
        self.credit_button.grid(row=0, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Клиенты"
        self.clients_button = ctk.CTkButton(self.menu_frame, text="Клиенты",
                                            command=self.show_client_state)
        self.clients_button.grid(row=1, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Сотрудники"
        self.employees_button = ctk.CTkButton(self.menu_frame, text="Сотрудники",
                                              command=self.show_employee_state)
        self.employees_button.grid(row=2, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Договоры"
        self.contracts_button = ctk.CTkButton(self.menu_frame, text="Договоры",
                                              command=self.show_contracts_state)
        self.contracts_button.grid(row=3, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Платежи"
        self.payments_button = ctk.CTkButton(self.menu_frame, text="Платежи",
                                             command=self.show_payments_state)
        self.payments_button.grid(row=4, column=0, padx=300, pady=25, sticky="nsew")

    def init_credit_state_frame(self):
        # Кнопка "Кредит 1"
        self.credit1_button = ctk.CTkButton(self.credit_state_frame, text="Все кредиты",
                                            command=lambda: self.show_result_state(fn.credits_command_first))
        self.credit1_button.grid(row=0, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Кредит 2"
        self.credit2_button = ctk.CTkButton(self.credit_state_frame, text="Кредит, больше определенной суммы",
                                            command=self.show_credit2_outlier_state)
        self.credit2_button.grid(row=1, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Кредит 3"
        self.credit3_button = ctk.CTkButton(self.credit_state_frame, text="Непогашенные кредиты",
                                            command=lambda: self.show_result_state(fn.credits_command_third))
        self.credit3_button.grid(row=2, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Кредит 4"
        self.credit4_button = ctk.CTkButton(self.credit_state_frame, text="Остаток кредита",
                                            command=lambda: self.show_result_state(fn.credits_command_fourth))
        self.credit4_button.grid(row=3, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Кредит 5"
        self.credit5_button = ctk.CTkButton(self.credit_state_frame, text="Кредиты, которые долго не выплачивали",
                                            command=self.show_credit5_outlier_state)
        self.credit5_button.grid(row=4, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Кредит 6"
        self.credit6_button = ctk.CTkButton(self.credit_state_frame, text="Долгосрочные кредиты",
                                            command=self.show_credit6_outlier_state)
        self.credit6_button.grid(row=5, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Назад"
        self.back_button_credit = ctk.CTkButton(self.credit_state_frame, text="Назад",
                                                command=self.show_menu, fg_color="grey")
        self.back_button_credit.grid(row=6, column=0, padx=300, pady=25, sticky="nsew")

    def init_credit2_outlier_state_frame(self):
        # Указание, что нужно вводить пользователю
        label = ctk.CTkLabel(self.credit2_outlier_state_frame, text='Введите сумму. Например: 10000')
        label.grid(row=0, column=1, padx=300, pady=25, sticky="nsew")

        # Элементы для ввода
        self.temp_entry_credit2 = Entry(self.credit2_outlier_state_frame)  # этот объект - поле для ввода, а не параметр,
        # который мы ввели
        self.temp_entry_credit2.grid(row=1, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Подтвердить"
        self.confirm_button_credit2 = ctk.CTkButton(self.credit2_outlier_state_frame, text="Подтвердить",
                                                    command=lambda: self.confirm_temp(fn.credits_command_second, self.temp_entry_credit2))
        # отправляем в confirm указатель на функцию, НЕ СТРОКУ
        self.confirm_button_credit2.grid(row=2, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Назад"
        self.back_button_cred_out2 = ctk.CTkButton(self.credit2_outlier_state_frame, text="Назад",
                                                   command=self.show_credit_state, fg_color="grey")
        self.back_button_cred_out2.grid(row=3, column=1, padx=300, pady=0, sticky="nsew")

    def init_credit5_outlier_state_frame(self):
        # Указание, что нужно вводить пользователю
        label = ctk.CTkLabel(self.credit5_outlier_state_frame, text='Введите количество дней. Например: 10')
        label.grid(row=0, column=1, padx=300, pady=25, sticky="nsew")

        # Элементы для ввода
        self.temp_entry_credit5 = Entry(self.credit5_outlier_state_frame)  # этот объект - поле для ввода, а не параметр,
        # который мы ввели
        self.temp_entry_credit5.grid(row=1, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Подтвердить"
        self.confirm_button_credit5 = ctk.CTkButton(self.credit5_outlier_state_frame, text="Подтвердить",
                                                    command=lambda: self.confirm_temp(fn.credits_command_fifth, self.temp_entry_credit5))
        # отправляем в confirm указатель на функцию, НЕ СТРОКУ
        self.confirm_button_credit5.grid(row=2, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Назад"
        self.back_button_cred_out5 = ctk.CTkButton(self.credit5_outlier_state_frame, text="Назад",
                                                   command=self.show_credit_state, fg_color="grey")
        self.back_button_cred_out5.grid(row=3, column=1, padx=300, pady=0, sticky="nsew")

    def init_credit6_outlier_state_frame(self):
        # Указание, что нужно вводить пользователю
        label = ctk.CTkLabel(self.credit6_outlier_state_frame, text='Введите количество месяцев. Например: 35')
        label.grid(row=0, column=1, padx=300, pady=25, sticky="nsew")

        # Элементы для ввода
        self.temp_entry_credit6 = Entry(self.credit6_outlier_state_frame)  # этот объект - поле для ввода, а не параметр,
        # который мы ввели
        self.temp_entry_credit6.grid(row=1, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Подтвердить"
        self.confirm_button_credit6 = ctk.CTkButton(self.credit6_outlier_state_frame, text="Подтвердить",
                                                    command=lambda: self.confirm_temp(fn.credits_command_sixth, self.temp_entry_credit6))
        # отправляем в confirm указатель на функцию, НЕ СТРОКУ
        self.confirm_button_credit6.grid(row=2, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Назад"
        self.back_button_cred_out6 = ctk.CTkButton(self.credit6_outlier_state_frame, text="Назад",
                                                   command=self.show_credit_state, fg_color="grey")
        self.back_button_cred_out6.grid(row=3, column=1, padx=300, pady=0, sticky="nsew")

    def init_client_state_frame(self):
        # Кнопка "Клиент 1"
        self.client1_button = ctk.CTkButton(self.client_state_frame, text="Все клиенты",
                                            command=lambda: self.show_result_state(fn.clients_command_first))
        self.client1_button.grid(row=0, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Клиент 2"
        self.client2_button = ctk.CTkButton(self.client_state_frame, text="Общая сумма кредитов у каждого клиента",
                                            command=lambda: self.show_result_state(fn.clients_command_second))
        self.client2_button.grid(row=1, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Клиент 3"
        self.client3_button = ctk.CTkButton(self.client_state_frame, text="Клиенты из определенного города",
                                            command=self.show_client3_outlier_state)
        self.client3_button.grid(row=2, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Клиент 4"
        self.client4_button = ctk.CTkButton(self.client_state_frame, text="Выплаты каждого клиента",
                                            command=lambda: self.show_result_state(fn.clients_command_fourth))
        self.client4_button.grid(row=3, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Клиент 5"
        self.client5_button = ctk.CTkButton(self.client_state_frame, text="Количество погашенных кредитов у клиентов",
                                            command=lambda: self.show_result_state(fn.clients_command_fifth))
        self.client5_button.grid(row=4, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Клиент 6"
        self.client6_button = ctk.CTkButton(self.client_state_frame, text="Количество взятых кредитов у клиента",
                                            command=self.show_client6_outlier_state)
        self.client6_button.grid(row=5, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Назад"
        self.back_button_client = ctk.CTkButton(self.client_state_frame, text="Назад",
                                                command=self.show_menu, fg_color="grey")
        self.back_button_client.grid(row=6, column=0, padx=300, pady=25, sticky="nsew")

    def init_client3_outlier_state_frame(self):
        # Указание, что нужно вводить пользователю
        label = ctk.CTkLabel(self.client3_outlier_state_frame, text='Введите город. Например: Dallas')
        label.grid(row=0, column=1, padx=300, pady=25, sticky="nsew")

        # Элементы для ввода
        self.temp_entry_client3 = Entry(self.client3_outlier_state_frame)  # этот объект - поле для ввода, а не параметр,
        # который мы ввели
        self.temp_entry_client3.grid(row=1, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Подтвердить"
        self.confirm_button_client3 = ctk.CTkButton(self.client3_outlier_state_frame, text="Подтвердить",
                                                    command=lambda: self.confirm_temp(fn.clients_command_third, self.temp_entry_client3))
        # отправляем в confirm указатель на функцию, НЕ СТРОКУ
        self.confirm_button_client3.grid(row=2, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Назад"
        self.back_button_clie_out3 = ctk.CTkButton(self.client3_outlier_state_frame, text="Назад",
                                                   command=self.show_client_state, fg_color="grey")
        self.back_button_clie_out3.grid(row=3, column=1, padx=300, pady=0, sticky="nsew")

    def init_client6_outlier_state_frame(self):
        # Указание, что нужно вводить пользователю
        label = ctk.CTkLabel(self.client6_outlier_state_frame, text='Введите клиента в формате "Имя Фамилия". Например Emily Brown')
        label.grid(row=0, column=1, padx=300, pady=25, sticky="nsew")

        # Элементы для ввода
        self.temp_entry_client6 = Entry(self.client6_outlier_state_frame)  # этот объект - поле для ввода, а не параметр,
        # который мы ввели
        self.temp_entry_client6.grid(row=1, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Подтвердить"
        self.confirm_button_client6 = ctk.CTkButton(self.client6_outlier_state_frame, text="Подтвердить",
                                                    command=lambda: self.confirm_temp(fn.clients_command_sixth, self.temp_entry_client6))
        # отправляем в confirm указатель на функцию, НЕ СТРОКУ
        self.confirm_button_client6.grid(row=2, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Назад"
        self.back_button_clie_out6 = ctk.CTkButton(self.client6_outlier_state_frame, text="Назад",
                                                   command=self.show_client_state, fg_color="grey")
        self.back_button_clie_out6.grid(row=3, column=1, padx=300, pady=0, sticky="nsew")

    def init_employee_state_frame(self):
        # Кнопка "Сотрудник 1"
        self.employee1_button = ctk.CTkButton(self.employee_state_frame, text="Все сотрудники",
                                              command=lambda: self.show_result_state(fn.employees_command_first))
        self.employee1_button.grid(row=0, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Сотрудник 2"
        self.employee2_button = ctk.CTkButton(self.employee_state_frame, text="Общая сумма выданных кредитов",
                                              command=lambda: self.show_result_state(fn.employees_command_second))
        self.employee2_button.grid(row=1, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Сотрудник 3"
        self.employee3_button = ctk.CTkButton(self.employee_state_frame, text="Закрепленные клиенты у каждого сотрудника",
                                              command=lambda: self.show_result_state(fn.employees_command_third))
        self.employee3_button.grid(row=2, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Сотрудник 4"
        self.employee4_button = ctk.CTkButton(self.employee_state_frame, text="Сотрудник определенной должности",
                                              command=self.show_employee4_outlier_state)
        self.employee4_button.grid(row=3, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Сотрудник 5"
        self.employee5_button = ctk.CTkButton(self.employee_state_frame, text="Сотрудники с определенным стажем работы",
                                              command=self.show_employee5_outlier_state)
        self.employee5_button.grid(row=4, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Назад"
        self.back_button_employee = ctk.CTkButton(self.employee_state_frame, text="Назад",
                                                  command=self.show_menu, fg_color="grey")
        self.back_button_employee.grid(row=5, column=0, padx=300, pady=25, sticky="nsew")

    def init_employee4_outlier_state_frame(self):
        # Указание, что нужно вводить пользователю
        label = ctk.CTkLabel(self.employee4_outlier_state_frame, text='Введите должность сотрудника. Например: Loan Officer')
        label.grid(row=0, column=1, padx=300, pady=25, sticky="nsew")

        # Элементы для ввода
        self.temp_entry_employee4 = Entry(self.employee4_outlier_state_frame)  # этот объект - поле для ввода, а не параметр,
        # который мы ввели
        self.temp_entry_employee4.grid(row=1, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Подтвердить"
        self.confirm_button_employee4 = ctk.CTkButton(self.employee4_outlier_state_frame, text="Подтвердить",
                                                      command=lambda: self.confirm_temp(fn.employees_command_fourth, self.temp_entry_employee4))
        # отправляем в confirm указатель на функцию, НЕ СТРОКУ
        self.confirm_button_employee4.grid(row=2, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Назад"
        self.back_button_empl_out4 = ctk.CTkButton(self.employee4_outlier_state_frame, text="Назад",
                                                   command=self.show_employee_state, fg_color="grey")
        self.back_button_empl_out4.grid(row=3, column=1, padx=300, pady=0, sticky="nsew")

    def init_employee5_outlier_state_frame(self):
        # Указание, что нужно вводить пользователю
        label = ctk.CTkLabel(self.employee5_outlier_state_frame, text='Введите количество дней стажа. Например: 212')
        label.grid(row=0, column=1, padx=300, pady=25, sticky="nsew")

        # Элементы для ввода
        self.temp_entry_employee5 = Entry(self.employee5_outlier_state_frame)  # этот объект - поле для ввода, а не параметр,
        # который мы ввели
        self.temp_entry_employee5.grid(row=1, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Подтвердить"
        self.confirm_button_employee5 = ctk.CTkButton(self.employee5_outlier_state_frame, text="Подтвердить",
                                                      command=lambda: self.confirm_temp(fn.employees_command_fifth, self.temp_entry_employee5))
        # отправляем в confirm указатель на функцию, НЕ СТРОКУ
        self.confirm_button_employee5.grid(row=2, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Назад"
        self.back_button_empl_out5 = ctk.CTkButton(self.employee5_outlier_state_frame, text="Назад",
                                                   command=self.show_employee_state, fg_color="grey")
        self.back_button_empl_out5.grid(row=3, column=1, padx=300, pady=0, sticky="nsew")

    def init_contracts_state_frame(self):
        # Кнопка "Договор 1"
        self.contracts1_button = ctk.CTkButton(self.contracts_state_frame, text="Информация о всех заключённых договорах",
                                               command=lambda: self.show_result_state(fn.credit_contracts_first))
        self.contracts1_button.grid(row=0, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Договор 2"
        self.contracts2_button = ctk.CTkButton(self.contracts_state_frame, text="ФИ сотрудника и клиента по идентификатору договора",
                                               command=lambda: self.show_result_state(fn.credit_contracts_second))
        self.contracts2_button.grid(row=1, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Договор 3"
        self.contracts3_button = ctk.CTkButton(self.contracts_state_frame, text="Договоры, заключённые на сумму меньше запрашиваемой",
                                               command=lambda: self.show_result_state(fn.credits_command_third))
        self.contracts3_button.grid(row=2, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Договор 4"
        self.contracts4_button = ctk.CTkButton(self.contracts_state_frame, text="Договоры, заключенные после определённой даты",
                                               command=self.show_contracts4_outlier_state)
        self.contracts4_button.grid(row=3, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Назад"
        self.back_button_contracts = ctk.CTkButton(self.contracts_state_frame, text="Назад",
                                                   command=self.show_menu, fg_color="grey")
        self.back_button_contracts.grid(row=4, column=0, padx=300, pady=25, sticky="nsew")

    def init_contracts4_outlier_state_frame(self):
        # Указание, что нужно вводить пользователю
        label = ctk.CTkLabel(self.contracts4_outlier_state_frame, text='Введите дату в формате YYYY-MM-DD. Например 2023-10-03')
        label.grid(row=0, column=1, padx=300, pady=25, sticky="nsew")

        # Элементы для ввода
        self.temp_entry_contracts4 = Entry(self.contracts4_outlier_state_frame)  # этот объект - поле для ввода, а не параметр,
        # который мы ввели
        self.temp_entry_contracts4.grid(row=1, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Подтвердить"
        self.confirm_button_contracts4 = ctk.CTkButton(self.contracts4_outlier_state_frame, text="Подтвердить",
                                                       command=lambda: self.confirm_temp(fn.credit_contracts_fourth, self.temp_entry_contracts4))
        # отправляем в confirm указатель на функцию, НЕ СТРОКУ
        self.confirm_button_contracts4.grid(row=2, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Назад"
        self.back_button_contr_out4 = ctk.CTkButton(self.contracts4_outlier_state_frame, text="Назад",
                                                    command=self.show_contracts_state, fg_color="grey")
        self.back_button_contr_out4.grid(row=3, column=1, padx=300, pady=0, sticky="nsew")

    def init_payments_state_frame(self):
        # Кнопка "Платеж 1"
        self.payment1_button = ctk.CTkButton(self.payments_state_frame, text="Все внесённые платежи",
                                             command=lambda: self.show_result_state(fn.payments_command_first))
        self.payment1_button.grid(row=0, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Платеж 2"
        self.payment2_button = ctk.CTkButton(self.payments_state_frame, text="Сумма платежей, внесённых после заданной даты",
                                             command=self.show_payment2_outlier_state)
        self.payment2_button.grid(row=1, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Платеж 3"
        self.payment3_button = ctk.CTkButton(self.payments_state_frame, text="Все платежи, большие заданной суммы",
                                             command=self.show_payment3_outlier_state)
        self.payment3_button.grid(row=2, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Платеж 4"
        self.payments4_button = ctk.CTkButton(self.payments_state_frame,
                                              text="Информация об отправителе по каждому платежу",
                                              command=lambda: self.show_result_state(fn.payments_command_fourth))
        self.payments4_button.grid(row=3, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Платеж 6"
        self.payments6_button = ctk.CTkButton(self.payments_state_frame,
                                              text="Сумма платежей заданного клиента по всем его кредитам",
                                              command=self.show_payment6_outlier_state)
        self.payments6_button.grid(row=4, column=0, padx=300, pady=25, sticky="nsew")

        # Кнопка "Назад"
        self.back_button_payments = ctk.CTkButton(self.payments_state_frame, text="Назад",
                                                  command=self.show_menu, fg_color="grey")
        self.back_button_payments.grid(row=5, column=0, padx=300, pady=25, sticky="nsew")

    def init_payment2_outlier_state_frame(self):
        # Указание, что нужно вводить пользователю
        label = ctk.CTkLabel(self.payment2_outlier_state_frame, text='Введите дату формате YYYY-MM-DD. Например: 2023-11-08')
        label.grid(row=0, column=1, padx=300, pady=25, sticky="nsew")

        # Элементы для ввода
        self.temp_entry_payment2 = Entry(
            self.payment2_outlier_state_frame)  # этот объект - поле для ввода, а не параметр
        # который мы ввели
        self.temp_entry_payment2.grid(row=1, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Подтвердить"
        self.confirm_button_payment2 = ctk.CTkButton(self.payment2_outlier_state_frame, text="Подтвердить",
                                                     command=lambda: self.confirm_temp(fn.payments_command_second, self.temp_entry_payment2))
        # отправляем в confirm указатель на функцию, НЕ СТРОКУ
        self.confirm_button_payment2.grid(row=2, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Назад"
        self.back_button_payment_out2 = ctk.CTkButton(self.payment2_outlier_state_frame, text="Назад",
                                                      command=self.show_payments_state, fg_color="grey")
        self.back_button_payment_out2.grid(row=3, column=1, padx=300, pady=0, sticky="nsew")

    def init_payment3_outlier_state_frame(self):
        # Указание, что нужно вводить пользователю
        label = ctk.CTkLabel(self.payment3_outlier_state_frame, text='Введите минимальную сумму платежа. Например: 1200')
        label.grid(row=0, column=1, padx=300, pady=25, sticky="nsew")

        # Элементы для ввода
        self.temp_entry_payment3 = Entry(
            self.payment3_outlier_state_frame)  # этот объект - поле для ввода, а не параметр
        # который мы ввели
        self.temp_entry_payment3.grid(row=1, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Подтвердить"
        self.confirm_button_payment3 = ctk.CTkButton(self.payment3_outlier_state_frame, text="Подтвердить",
                                                     command=lambda: self.confirm_temp(fn.payments_command_third, self.temp_entry_payment3))
        # отправляем в confirm указатель на функцию, НЕ СТРОКУ
        self.confirm_button_payment3.grid(row=2, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Назад"
        self.back_button_payment_out3 = ctk.CTkButton(self.payment3_outlier_state_frame, text="Назад",
                                                      command=self.show_payments_state, fg_color="grey")
        self.back_button_payment_out3.grid(row=3, column=1, padx=300, pady=0, sticky="nsew")

    def init_payment6_outlier_state_frame(self):
        # Указание, что нужно вводить пользователю
        label = ctk.CTkLabel(self.payment6_outlier_state_frame, text='Введите клиента в формате "Имя Фамилия". Например: Emily Brown')
        label.grid(row=0, column=1, padx=300, pady=25, sticky="nsew")

        # Элементы для ввода
        self.temp_entry_payment6 = Entry(
            self.payment6_outlier_state_frame)  # этот объект - поле для ввода, а не параметр
        # который мы ввели
        self.temp_entry_payment6.grid(row=1, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Подтвердить"
        self.confirm_button_payment6 = ctk.CTkButton(self.payment6_outlier_state_frame, text="Подтвердить",
                                                     command=lambda: self.confirm_temp(fn.payments_command_sixth, self.temp_entry_payment6))

        # отправляем в confirm указатель на функцию, НЕ СТРОКУ
        self.confirm_button_payment6.grid(row=2, column=1, padx=300, pady=25, sticky="nsew")

        # Кнопка "Назад"
        self.back_button_payment_out6 = ctk.CTkButton(self.payment6_outlier_state_frame, text="Назад",
                                                      command=self.show_payments_state, fg_color="grey")
        self.back_button_payment_out6.grid(row=3, column=1, padx=300, pady=0, sticky="nsew")

    def init_result_state_frame(self):
        # Treeview для отображения данных в виде таблицы
        self.tree = ttk.Treeview(self.result_state_frame)
        self.tree.grid(row=0, column=0, padx=(10, 10), pady=(25, 0), sticky="nsew")

        # Создаем виджеты Scrollbar
        self.vertical_scrollbar = Scrollbar(self.result_state_frame, orient="vertical",
                                            command=self.tree.yview)
        self.vertical_scrollbar.grid(row=0, column=1, sticky="ns")

        self.horizontal_scrollbar = Scrollbar(self.result_state_frame, orient="horizontal",
                                              command=self.tree.xview)
        self.horizontal_scrollbar.grid(row=1, column=0, sticky="ew")

        # Привязываем Scrollbar к Treeview
        self.tree.configure(yscrollcommand=self.vertical_scrollbar.set, xscrollcommand=self.horizontal_scrollbar.set)

        # Ограничиваем размеры Treeview
        self.tree.config(height=10, show="headings")

        # Кнопка "Назад"
        self.back_button_result = ctk.CTkButton(self.result_state_frame, text="Назад",
                                                command=self.show_menu, fg_color="grey")
        self.back_button_result.grid(row=2, column=0, padx=(10, 10), pady=(0, 50), sticky="nsew")
        # self.back_button_result.place(x=10, y=675)  # Задайте нужные вам координаты кнопки

    def hide_all_states(self):
        self.menu_frame.grid_forget()
        #
        self.credit_state_frame.grid_forget()
        self.credit2_outlier_state_frame.grid_forget()
        self.credit5_outlier_state_frame.grid_forget()
        self.credit6_outlier_state_frame.grid_forget()
        #
        self.client_state_frame.grid_forget()
        self.client3_outlier_state_frame.grid_forget()
        self.client6_outlier_state_frame.grid_forget()
        #
        self.employee_state_frame.grid_forget()
        self.employee4_outlier_state_frame.grid_forget()
        self.employee5_outlier_state_frame.grid_forget()
        #
        self.contracts_state_frame.grid_forget()
        self.contracts4_outlier_state_frame.grid_forget()
        #
        self.payments_state_frame.grid_forget()
        self.payment2_outlier_state_frame.grid_forget()
        self.payment3_outlier_state_frame.grid_forget()
        self.payment6_outlier_state_frame.grid_forget()
        #
        self.result_state_frame.grid_forget()

    def show_menu(self):
        self.hide_all_states()
        self.menu_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.13, pady=220, sticky="nsew")

    def show_credit_state(self):
        self.hide_all_states()
        self.credit_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220, sticky="nsew")

    def show_credit2_outlier_state(self):
        self.hide_all_states()
        self.credit2_outlier_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220, sticky="nsew")

    def show_credit5_outlier_state(self):
        self.hide_all_states()
        self.credit5_outlier_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220, sticky="nsew")

    def show_credit6_outlier_state(self):
        self.hide_all_states()
        self.credit6_outlier_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220, sticky="nsew")

    def show_client_state(self):
        self.hide_all_states()
        self.client_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220, sticky="nsew")

    def show_client3_outlier_state(self):
        self.hide_all_states()
        self.client3_outlier_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220, sticky="nsew")

    def show_client6_outlier_state(self):
        self.hide_all_states()
        self.client6_outlier_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220, sticky="nsew")

    def show_employee_state(self):
        self.hide_all_states()
        self.employee_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220, sticky="nsew")

    def show_employee4_outlier_state(self):
        self.hide_all_states()
        self.employee4_outlier_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220, sticky="nsew")

    def show_employee5_outlier_state(self):
        self.hide_all_states()
        self.employee5_outlier_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220, sticky="nsew")

    def show_contracts_state(self):
        self.hide_all_states()
        self.contracts_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220, sticky="nsew")

    def show_contracts4_outlier_state(self):
        self.hide_all_states()
        self.contracts4_outlier_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220, sticky="nsew")

    def show_payments_state(self):
        self.hide_all_states()
        self.payments_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220, sticky="nsew")

    def show_payment2_outlier_state(self):
        self.hide_all_states()
        self.payment2_outlier_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220,
                                               sticky="nsew")

    def show_payment3_outlier_state(self):
        self.hide_all_states()
        self.payment3_outlier_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220,
                                               sticky="nsew")

    def show_payment6_outlier_state(self):
        self.hide_all_states()
        self.payment6_outlier_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220,
                                               sticky="nsew")

    def confirm_temp(self, func, to_save):
        temp = to_save.get().strip()
        if not temp:
            return
        self.hide_all_states()
        s = func(temp)
        self.show_result_state(s)

    def show_result_state(self, sql_request):
        self.hide_all_states()
        num_columns = show(self, x, sql_request)
        if num_columns > 7:
            self.result_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 13, pady=220, sticky="nsew")
        elif num_columns == 7:
            self.result_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 8, pady=220, sticky="nsew")
        elif num_columns == 6:
            self.result_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 6, pady=220, sticky="nsew")
        else:
            self.result_state_frame.grid(row=0, column=0, padx=self.winfo_screenwidth() / 3.7, pady=220, sticky="nsew")

        # Очистка Treeview перед новыми данными
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Вызов функции show с передачей self в качестве первого аргумента
        show(self, x, sql_request)

    def on_closing(self, a):
        a.close()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
