using System;
using GradeBookSystem.Models;
using GradeBookSystem.Services;
using GradeBookSystem.Exceptions;

namespace GradeBookSystem.ConsoleMenu
{
    public class GroupMenu
    {
        private readonly GradeBook _gradeBook;        
        private readonly int _menuItemCount = 6;
        private int _selectedIndex = 0;

        private readonly string[] _menuItems = {
            "Додати групу",
            "Видалити групу",
            "Змінити дані групи",
            "Переглянути список груп",
            "Переглянути дані певної групи",
            "Повернутися до головного меню"
        };

        public GroupMenu(GradeBook gradeBook)
        {
            _gradeBook = gradeBook;
        }

        public void Show()
        {
            while (true)
            {
                Console.Clear();
                DrawMenu();

                ConsoleKeyInfo keyInfo = Console.ReadKey(true);
                switch (keyInfo.Key)
                {
                    case ConsoleKey.UpArrow:
                        _selectedIndex = (_selectedIndex - 1 + _menuItemCount) % _menuItemCount;
                        break;
                    case ConsoleKey.DownArrow:
                        _selectedIndex = (_selectedIndex + 1) % _menuItemCount;
                        break;
                    case ConsoleKey.Enter:
                        Console.Clear();
                        try
                        {
                            ExecuteMenuItem();
                            if (_selectedIndex == _menuItemCount - 1) // Return to main menu option
                            {
                                return;
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.ForegroundColor = ConsoleColor.Red;
                            Console.WriteLine($"\nПомилка: {ex.Message}");
                            Console.ResetColor();
                        }
                        Console.WriteLine("\nНатисніть будь-яку клавішу, щоб повернутися до меню груп...");
                        Console.ReadKey(true);
                        break;
                }
            }
        }

        private void DrawMenu()
        {
            Console.WriteLine("Меню управління групами:");
            for (int i = 0; i < _menuItems.Length; i++)
            {
                if (i == _selectedIndex)
                {
                    Console.BackgroundColor = ConsoleColor.Gray;
                    Console.ForegroundColor = ConsoleColor.Black;
                }

                Console.WriteLine($"{i + 1}. {_menuItems[i]}");

                Console.ResetColor();
            }
        }

        private void ExecuteMenuItem()
        {
            switch (_selectedIndex)
            {
                case 0:
                    AddGroup();
                    break;
                case 1:
                    RemoveGroup();
                    break;
                case 2:
                    EditGroup();
                    break;
                case 3:
                    ViewAllGroups();
                    break;
                case 4:
                    ViewGroupDetails();
                    break;
            }
        }

        private void AddGroup()
        {
            try
            {
                Console.Write("Введіть назву групи: ");
                string groupName = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(groupName))
                {
                    throw new ArgumentException("Назва групи не може бути порожньою.");
                }

                _gradeBook.AddGroup(groupName);
                Console.WriteLine("Групу успішно додано.");
            }
            catch (ArgumentException ex)
            {
                throw new ArgumentException($"Помилка при додаванні групи: {ex.Message}");
            }
        }

        private void RemoveGroup()
        {
            try
            {
                Console.Write("Введіть назву групи: ");
                string groupName = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(groupName))
                {
                    throw new ArgumentException("Назва групи не може бути порожньою.");
                }

                _gradeBook.RemoveGroup(groupName);
                Console.WriteLine("Групу успішно видалено.");
            }
            catch (GroupNotFoundException ex)
            {
                throw new GroupNotFoundException($"Помилка при видаленні групи: {ex.Message}");
            }
        }

        private void EditGroup()
        {
            try
            {
                Console.Write("Введіть поточну назву групи: ");
                string currentName = Console.ReadLine();
                Console.Write("Введіть нову назву групи: ");
                string newName = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(currentName) || string.IsNullOrWhiteSpace(newName))
                {
                    throw new ArgumentException("Назва групи не може бути порожньою.");
                }

                _gradeBook.EditGroup(currentName, newName);
                Console.WriteLine("Дані групи успішно змінено.");
            }
            catch (GroupNotFoundException ex)
            {
                throw new GroupNotFoundException($"Помилка при редагуванні групи: {ex.Message}");
            }
            catch (ArgumentException ex)
            {
                throw new ArgumentException($"Помилка при редагуванні групи: {ex.Message}");
            }
        }

        private void ViewAllGroups()
        {
            var groups = _gradeBook.GetAllGroups();
            if (!groups.Any())
            {
                Console.WriteLine("Список груп порожній.");
                return;
            }

            foreach (var group in groups)
            {
                Console.WriteLine(group.GroupName);
            }
        }

        private void ViewGroupDetails()
        {
            try
            {
                Console.Write("Введіть назву групи: ");
                string groupName = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(groupName))
                {
                    throw new ArgumentException("Назва групи не може бути порожньою.");
                }

                var group = _gradeBook.GetGroupByName(groupName);
                Console.WriteLine($"Група: {group.GroupName}");
                Console.WriteLine("Студенти:");

                if (!group.Students.Any())
                {
                    Console.WriteLine("У групі немає студентів.");
                    return;
                }

                foreach (var student in group.Students)
                {
                    Console.WriteLine($"{student.FirstName} {student.LastName} (ID: {student.StudentId})");
                }
            }
            catch (GroupNotFoundException ex)
            {
                throw new GroupNotFoundException($"Помилка при перегляді даних групи: {ex.Message}");
            }
        }
    }
}