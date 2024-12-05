using System;
using GradeBookSystem.Models;
using GradeBookSystem.Services;

namespace GradeBookSystem.ConsoleMenu
{
    public class SubjectMenu
    {
        private readonly GradeBook _gradeBook;
        private readonly string[] _menuItems = {
            "Додати предмет",
            "Видалити предмет",
            "Переглянути список предметів",
            "Повернутися до головного меню"
        };

        public SubjectMenu(GradeBook gradeBook)
        {
            _gradeBook = gradeBook;
        }

        public void Show()
        {
            while (true)
            {
                Console.Clear();
                Console.WriteLine("\nМеню управління предметами:");
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
                    AddSubject();
                    break;
                case 1:
                    RemoveSubject();
                    break;
                case 2:
                    ViewAllSubjects();
                    break;
                case 3:
                    return true;
            }
            return false;
        }

        private void AddSubject()
        {
            try
            {
                Console.Write("Введіть назву предмета: ");
                string subjectName = Console.ReadLine();
                _gradeBook.AddSubject(subjectName);
                Console.WriteLine("Предмет успішно додано.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Помилка: {ex.Message}");
            }
        }

        private void RemoveSubject()
        {
            try
            {
                Console.Write("Введіть назву предмета: ");
                string subjectName = Console.ReadLine();
                _gradeBook.RemoveSubject(subjectName);
                Console.WriteLine("Предмет успішно видалено.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Помилка: {ex.Message}");
            }
        }

        private void ViewAllSubjects()
        {
            var subjects = _gradeBook.GetAllSubjects();
            if (subjects.Count == 0)
            {
                Console.WriteLine("Список предметів порожній.");
                return;
            }

            foreach (var subject in subjects)
            {
                Console.WriteLine(subject);
            }
        }
    }
}