using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using GradeBookSystem.Exceptions;

namespace GradeBookSystem.Models
{
    public class Student : IPerson, IComparable<Student>
    {
        private string firstName;
        private string lastName;
        private string studentId;

        public string FirstName
        {
            get => firstName;
            set
            {
                if (!IsValidName(value))
                    throw new ArgumentException("Ім'я не може містити цифри.");
                firstName = value;
            }
        }

        public string LastName
        {
            get => lastName;
            set
            {
                if (!IsValidName(value))
                    throw new ArgumentException("Прізвище не може містити цифри.");
                lastName = value;
            }
        }

        public string StudentId
        {
            get => studentId;
            set
            {
                if (!IsValidStudentId(value))
                    throw new InvalidStudentIdFormatException("ID студента повинен бути у форматі STU12345.");
                studentId = value;
            }
        }

        public Dictionary<string, List<int>> Grades { get; set; } = new Dictionary<string, List<int>>();

        private bool IsValidName(string name)
        {
            return !string.IsNullOrWhiteSpace(name) && !name.Any(char.IsDigit);
        }

        private bool IsValidStudentId(string id)
        {
            return Regex.IsMatch(id, @"^STU\d{5}$");
        }

        public double GetAverageGrade()
        {
            if (Grades.Count == 0) return 0;
            return Grades.SelectMany(g => g.Value).Average();
        }

        public int CompareTo(Student other)
        {
            return this.GetAverageGrade().CompareTo(other.GetAverageGrade());
        }
    }
}