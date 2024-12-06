using System;
using GradeBookSystem.Models;
using GradeBookSystem.Services;

namespace GradeBookSystem.ConsoleMenu
{
    public class GradeMenu
    {
        private readonly GradeBook _gradeBook;
        private readonly string[] _menuItems = {
            "Додати оцінку",
            "Змінити оцінку",
            "Переглянути оцінки студента",
            "Підрахувати середній бал студента",
            "Підрахувати середній бал групи",
            "Повернутися до головного меню"
        };

        public GradeMenu(GradeBook gradeBook)
        {
            _gradeBook = gradeBook;
        }

        public void Show()
        {
            while (true)
            {
                Console.Clear();
                Console.WriteLine("\nМеню управління оцінками:");
                for (int i = 0; i < _menuItems.Length; i++)
                {
                    Console.WriteLine($"{i + 1}. {_menuItems[i]}");
                }
                Console.Write("\nОберіть опцію: ");

                if (int.TryParse(Console.ReadLine(), out int choice) && choice >= 1 && choice <= _menuItems.Length)
                {
                    Console.Clear();
                    if (ExecuteMenuItem(choice - 1))
                        return;

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
                    AddGrade();
                    break;
                case 1:
                    EditGrade();
                    break;
                case 2:
                    ViewStudentGrades();
                    break;
                case 3:
                    CalculateStudentAverage();
                    break;
                case 4:
                    CalculateGroupAverage();
                    break;
                case 5:
                    return true;
            }
            return false;
        }

        private void AddGrade()
        {
            try
            {
                Console.Write("Введіть ID студента: ");
                string studentId = Console.ReadLine();
                Console.Write("Введіть назву предмета: ");
                string subject = Console.ReadLine();
                Console.Write("Введіть оцінку: ");
                if (int.TryParse(Console.ReadLine(), out int grade))
                {
                    _gradeBook.AddGrade(studentId, subject, grade);
                    Console.WriteLine("Оцінку успішно додано.");
                }
                else
                {
                    Console.WriteLine("Некоректне введення оцінки.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Помилка: {ex.Message}");
            }
        }

        private void EditGrade()
        {
            try
            {
                Console.Write("Введіть ID студента: ");
                string studentId = Console.ReadLine();
                Console.Write("Введіть назву предмета: ");
                string subject = Console.ReadLine();
                Console.Write("Введіть індекс оцінки для зміни: ");
                if (int.TryParse(Console.ReadLine(), out int index))
                {
                    Console.Write("Введіть нову оцінку: ");
                    if (int.TryParse(Console.ReadLine(), out int newGrade))
                    {
                        _gradeBook.EditGrade(studentId, subject, index, newGrade);
                        Console.WriteLine("Оцінку успішно змінено.");
                    }
                    else
                    {
                        Console.WriteLine("Некоректне введення оцінки.");
                    }
                }
                else
                {
                    Console.WriteLine("Некоректне введення індексу.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Помилка: {ex.Message}");
            }
        }

        private void ViewStudentGrades()
        {
            Console.Write("Введіть ID студента: ");
            string studentId = Console.ReadLine();

            try
            {
                var student = _gradeBook.GetStudentById(studentId);
                Console.WriteLine($"Оцінки студента {student.FirstName} {student.LastName}:");
                foreach (var subject in student.Grades.Keys)
                {
                    Console.WriteLine($"{subject}: {string.Join(", ", student.Grades[subject])}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Помилка: {ex.Message}");
            }
        }

        private void CalculateStudentAverage()
        {
            try
            {
                Console.Write("Введіть ID студента: ");
                string studentId = Console.ReadLine();
                double average = _gradeBook.CalculateStudentAverage(studentId);
                Console.WriteLine($"Середній бал студента: {average:F2}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Помилка: {ex.Message}");
            }
        }

        private void CalculateGroupAverage()
        {
            try
            {
                Console.Write("Введіть назву групи: ");
                string groupName = Console.ReadLine();
                double average = _gradeBook.CalculateGroupAverage(groupName);
                Console.WriteLine($"Середній бал групи: {average:F2}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Помилка: {ex.Message}");
            }
        }
    }
}