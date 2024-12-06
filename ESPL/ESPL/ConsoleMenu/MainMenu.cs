using System;

using GradeBookSystem.Models;
using GradeBookSystem.Services;

namespace GradeBookSystem.ConsoleMenu
{
    public class MainMenu
    {
        private readonly GradeBook _gradeBook;
        private readonly int _menuItemCount = 6;
        private int _selectedIndex = 0;

        private readonly string[] _menuItems = {
            "Управління групами",
            "Управління студентами",
            "Управління предметами",
            "Управління оцінками",
            "Пошук",
            "Зберегти та вийти"
        };

        public MainMenu(GradeBook gradeBook)
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
                        ExecuteMenuItem();
                        if (_selectedIndex == _menuItemCount - 1) // Exit option
                        {
                            _gradeBook.SaveData();
                            Console.CursorVisible = true;
                            return;
                        }
                        Console.WriteLine("\nНатисніть будь-яку клавішу, щоб повернутися до головного меню...");
                        Console.ReadKey(true);
                        break;
                }
            }
        }

        private void DrawMenu()
        {
            Console.WriteLine("Головне меню:");
            for (int i = 0; i < _menuItems.Length; i++)
            {
                if (i == _selectedIndex)
                {
                    Console.BackgroundColor = ConsoleColor.Yellow;
                    Console.ForegroundColor = ConsoleColor.Blue;
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
                    var groupMenu = new GroupMenu(_gradeBook);
                    groupMenu.Show();
                    break;
                case 1:
                    var studentMenu = new StudentMenu(_gradeBook);
                    studentMenu.Show();
                    break;
                case 2:
                    var subjectMenu = new SubjectMenu(_gradeBook);
                    subjectMenu.Show();
                    break;
                case 3:
                    var gradeMenu = new GradeMenu(_gradeBook);
                    gradeMenu.Show();
                    break;
                case 4:
                    var searchMenu = new SearchMenu(_gradeBook);
                    searchMenu.Show();
                    break;
                case 5:
                    Console.WriteLine("Дані збережено. Дякуємо за використання програми!");
                    break;
            }
        }
    }
}