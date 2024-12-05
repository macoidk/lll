using System;
using GradeBookSystem.Models;
using GradeBookSystem.Services;
using GradeBookSystem.Exceptions;

namespace GradeBookSystem.ConsoleMenu
{
    public class SearchMenu
    {
        private readonly GradeBook _gradeBook;
        private readonly string[] _menuItems = {
            "Пошук студента за прізвищем та ім'ям",
            "Пошук студентів певної групи",
            "Пошук студентів за середнім балом",
            "Пошук успішних/неуспішних студентів",
            "Повернутися до головного меню"
        };

        public SearchMenu(GradeBook gradeBook)
        {
            _gradeBook = gradeBook;
        }

        public void Show()
        {
            while (true)
            {
                Console.Clear();
                Console.WriteLine("\nМеню пошуку:");
                for (int i = 0; i < _menuItems.Length; i++)
                {
                    Console.WriteLine($"{i + 1}. {_menuItems[i]}");
                }
                Console.Write("\nОберіть опцію: ");

                if (int.TryParse(Console.ReadLine(), out int choice) && choice >= 1 && choice <= _menuItems.Length)
                {
                    Console.Clear();
                    try
                    {
                        if (ExecuteMenuItem(choice - 1))
                            return;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Помилка: {ex.Message}");
                    }

                    Console.WriteLine("\nНатисніть будь-яку клавішу, щоб продовжити...");
                    Console.ReadKey(true);
                }
                else
                {
                    Console.WriteLine("Невірний вибір. Спробуйте ще раз.");
                }
            }
        }

        private bool ExecuteMenuItem(int index)
        {
            switch (index)
            {
                case 0:
                    SearchStudentByName();
                    break;
                case 1:
                    SearchStudentsByGroup();
                    break;
                case 2:
                    SearchStudentsByAverageGrade();
                    break;
                case 3:
                    SearchSuccessfulStudents();
                    break;
                case 4:
                    return true;
            }
            return false;
        }

        private void SearchStudentByName()
        {
            Console.Write("Введіть прізвище студента: ");
            string lastName = Console.ReadLine();
            Console.Write("Введіть ім'я студента: ");
            string firstName = Console.ReadLine();

            try
            {
                var students = _gradeBook.SearchStudentsByName(lastName, firstName);
                if (students.Count == 0)
                {
                    throw new StudentNotFoundException($"Студентів з прізвищем {lastName} та ім'ям {firstName} не знайдено.");
                }

                Console.WriteLine("Знайдені студенти:");
                foreach (var student in students)
                {
                    Console.WriteLine($"{student.FirstName} {student.LastName} (ID: {student.StudentId})");
                }
            }
            catch (StudentNotFoundException ex)
            {
                Console.WriteLine(ex.Message);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Сталася непередбачена помилка: {ex.Message}");
            }
        }

        private void SearchStudentsByGroup()
        {
            Console.Write("Введіть назву групи: ");
            string groupName = Console.ReadLine();

            try
            {
                var students = _gradeBook.SearchStudentsByGroup(groupName);
                if (students.Count == 0)
                {
                    throw new StudentNotFoundException($"Студентів у групі {groupName} не знайдено.");
                }

                Console.WriteLine($"Студенти групи {groupName}:");
                foreach (var student in students)
                {
                    Console.WriteLine($"{student.FirstName} {student.LastName} (ID: {student.StudentId})");
                }
            }
            catch (StudentNotFoundException ex)
            {
                Console.WriteLine(ex.Message);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Сталася непередбачена помилка: {ex.Message}");
            }
        }

        private void SearchStudentsByAverageGrade()
        {
            try
            {
                Console.Write("Введіть мінімальний середній бал: ");
                if (!double.TryParse(Console.ReadLine(), out double minAverage))
                {
                    throw new ArgumentException("Некоректне введення. Будь ласка, введіть число.");
                }

                Console.Write("Введіть максимальний середній бал: ");
                if (!double.TryParse(Console.ReadLine(), out double maxAverage))
                {
                    throw new ArgumentException("Некоректне введення. Будь ласка, введіть число.");
                }

                if (minAverage > maxAverage)
                {
                    throw new ArgumentException("Мінімальний середній бал не може бути більшим за максимальний.");
                }

                var students = _gradeBook.SearchStudentsByAverageGrade(minAverage, maxAverage);
                if (students.Count == 0)
                {
                    throw new StudentNotFoundException($"Студентів з середнім балом від {minAverage} до {maxAverage} не знайдено.");
                }

                Console.WriteLine($"Студенти з середнім балом від {minAverage} до {maxAverage}:");
                foreach (var student in students)
                {
                    double avgGrade = student.GetAverageGrade();
                    Console.WriteLine($"{student.FirstName} {student.LastName} (ID: {student.StudentId}), Середній бал: {avgGrade:F2}");
                }
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine(ex.Message);
            }
            catch (StudentNotFoundException ex)
            {
                Console.WriteLine(ex.Message);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Сталася непередбачена помилка: {ex.Message}");
            }
        }

        private void SearchSuccessfulStudents()
        {
            try
            {
                Console.Write("Введіть мінімальний середній бал для успішних студентів: ");
                if (!double.TryParse(Console.ReadLine(), out double minAverage))
                {
                    throw new ArgumentException("Некоректне введення. Будь ласка, введіть число.");
                }

                var students = _gradeBook.SearchSuccessfulStudents(minAverage);
                if (students.Count == 0)
                {
                    throw new StudentNotFoundException($"Студентів з середнім балом вище {minAverage} не знайдено.");
                }

                Console.WriteLine($"Успішні студенти (середній бал вище {minAverage}):");
                foreach (var student in students)
                {
                    double avgGrade = student.GetAverageGrade();
                    Console.WriteLine($"{student.FirstName} {student.LastName} (ID: {student.StudentId}), Середній бал: {avgGrade:F2}");
                }
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine(ex.Message);
            }
            catch (StudentNotFoundException ex)
            {
                Console.WriteLine(ex.Message);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Сталася непередбачена помилка: {ex.Message}");
            }
        }
    }
}