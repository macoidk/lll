using System;
using GradeBookSystem.Models;
using GradeBookSystem.Exceptions;
using GradeBookSystem.Services;

namespace GradeBookSystem.ConsoleMenu
{
    public class StudentMenu
    {
        private readonly GradeBook _gradeBook;
        private readonly string[] _menuItems = {
            "Додати студента",
            "Видалити студента",
            "Змінити дані студента",
            "Переглянути список всіх студентів",
            "Переглянути дані про успішність студента",
            "Повернутися до головного меню"
        };

        public StudentMenu(GradeBook gradeBook)
        {
            _gradeBook = gradeBook;
        }

        public void Show()
        {
            while (true)
            {
                Console.Clear();
                Console.WriteLine("\nМеню управління студентами:");
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
                    AddStudent();
                    break;
                case 1:
                    RemoveStudent();
                    break;
                case 2:
                    EditStudent();
                    break;
                case 3:
                    ViewAllStudents();
                    break;
                case 4:
                    ViewStudentGrades();
                    break;
                case 5:
                    return true;
            }
            return false;
        }

        private void AddStudent()
        {
            try
            {
                Console.Write("Введіть назву групи: ");
                string groupName = Console.ReadLine();
                Console.Write("Введіть ім'я студента: ");
                string firstName = Console.ReadLine();
                Console.Write("Введіть прізвище студента: ");
                string lastName = Console.ReadLine();
                Console.Write("Введіть ID студента(у форматі STU12345): ");
                string studentId = Console.ReadLine();

                var student = new Student
                {
                    FirstName = firstName,
                    LastName = lastName,
                    StudentId = studentId
                };

                _gradeBook.AddStudent(groupName, student);
                Console.WriteLine("Студента успішно додано.");
            }
            catch (InvalidStudentIdFormatException ex)
            {
                Console.WriteLine($"Помилка: {ex.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Помилка: {ex.Message}");
            }
        }

        private void RemoveStudent()
        {
            Console.Write("Введіть назву групи: ");
            string groupName = Console.ReadLine();
            Console.Write("Введіть ID студента: ");
            string studentId = Console.ReadLine();

            try
            {
                _gradeBook.RemoveStudent(groupName, studentId);
                Console.WriteLine("Студента успішно видалено.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Помилка: {ex.Message}");
            }
        }

        private void EditStudent()
        {
            try
            {
                Console.Write("Введіть назву групи: ");
                string groupName = Console.ReadLine();
                Console.Write("Введіть поточний ID студента: ");
                string currentId = Console.ReadLine();
                Console.Write("Введіть нове ім'я студента: ");
                string newFirstName = Console.ReadLine();
                Console.Write("Введіть нове прізвище студента: ");
                string newLastName = Console.ReadLine();
                Console.Write("Введіть новий ID студента: ");
                string newId = Console.ReadLine();

                _gradeBook.EditStudent(groupName, currentId, newFirstName, newLastName, newId);
                Console.WriteLine("Дані студента успішно змінено.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Помилка: {ex.Message}");
            }
        }

        private void ViewAllStudents()
        {
            var students = _gradeBook.GetAllStudents();
            if (students.Count == 0)
            {
                Console.WriteLine("Список студентів порожній.");
                return;
            }

            foreach (var student in students)
            {
                Console.WriteLine($"{student.FirstName} {student.LastName} (ID: {student.StudentId})");
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

                if (student.Grades.Count == 0)
                {
                    Console.WriteLine("У студента ще немає оцінок.");
                    return;
                }

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
    }
}