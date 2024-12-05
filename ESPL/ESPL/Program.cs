using System;
using System.Text;
using System.Globalization;
using GradeBookSystem.ConsoleMenu;
using GradeBookSystem.Models;
using GradeBookSystem.Services;

namespace GradeBookSystem
{
    class Program
    {
        static GradeBook gradeBook;

        static Program()
        {
            try
            {
                gradeBook = new GradeBook();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Помилка при ініціалізації GradeBook: {ex.Message}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
                }
                Environment.Exit(1);
            }
        }

        static void Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.InputEncoding = Encoding.UTF8;
            CultureInfo.CurrentCulture = new CultureInfo("uk-UA");
            CultureInfo.CurrentUICulture = new CultureInfo("uk-UA");

            var mainMenu = new MainMenu(gradeBook);
            mainMenu.Show();
        }
    }
}
