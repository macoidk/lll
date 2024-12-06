using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GradeBookSystem.Models;
using GradeBookSystem.Exceptions;
using Newtonsoft.Json;

namespace GradeBookSystem.Services
{
    public class GradeBook
    {
        private List<Group> groups = new List<Group>();
        private List<string> subjects = new List<string>();
        private const string JsonDataFile = "gradebook.json";
        private const string TextDataFile = "gradebook.txt";

        public GradeBook()
        {
            LoadData();
        }

        public void AddGroup(string groupName)
        {
            if (groups.Any(g => g.GroupName == groupName))
                throw new DuplicateGroupException($"Група {groupName} вже існує.");
            groups.Add(new Group { GroupName = groupName });
        }

        public void RemoveGroup(string groupName)
        {
            var group = FindGroup(groupName);
            groups.Remove(group);
        }

        public void EditGroup(string currentName, string newName)
        {
            var group = FindGroup(currentName);
            group.GroupName = newName;
        }

        public List<Group> GetAllGroups()
        {
            return groups;
        }

        public Group GetGroupByName(string groupName)
        {
            return FindGroup(groupName);
        }

        public void AddStudent(string groupName, Student student)
        {
            var group = FindGroup(groupName);
            if (group.Students.Any(s => s.StudentId == student.StudentId))
                throw new DuplicateStudentException($"Студент з ID {student.StudentId} вже існує в групі {groupName}.");

            try
            {
                // Перевірка імені, прізвища та ID відбувається при присвоєнні значень властивостям
                student.FirstName = student.FirstName;
                student.LastName = student.LastName;
                student.StudentId = student.StudentId;
            }
            catch (ArgumentException ex)
            {
                throw new ArgumentException($"Помилка при додаванні студента: {ex.Message}");
            }
            catch (InvalidStudentIdFormatException ex)
            {
                throw new ArgumentException($"Помилка при додаванні студента: {ex.Message}");
            }

            group.Students.Add(student);
        }

        public void RemoveStudent(string groupName, string studentId)
        {
            var group = FindGroup(groupName);
            var student = FindStudent(group, studentId);
            group.Students.Remove(student);
        }

        public void EditStudent(string groupName, string currentId, string newFirstName, string newLastName, string newId)
        {
            var group = FindGroup(groupName);
            var student = FindStudent(group, currentId);
            student.FirstName = newFirstName;
            student.LastName = newLastName;
            student.StudentId = newId;
        }

        public List<Student> GetAllStudents()
        {
            return groups.SelectMany(g => g.Students).ToList();
        }

        public Student GetStudentById(string studentId)
        {
            return groups.SelectMany(g => g.Students).FirstOrDefault(s => s.StudentId == studentId)
                ?? throw new StudentNotFoundException($"Студента з ID {studentId} не знайдено.");
        }

        public void AddSubject(string subjectName)
        {
            if (subjects.Contains(subjectName))
                throw new DuplicateSubjectException($"Предмет {subjectName} вже існує.");
            subjects.Add(subjectName);
        }

        public void RemoveSubject(string subjectName)
        {
            if (!subjects.Remove(subjectName))
                throw new SubjectNotFoundException($"Предмет {subjectName} не знайдено.");
            foreach (var student in GetAllStudents())
            {
                student.Grades.Remove(subjectName);
            }
        }

        public List<string> GetAllSubjects()
        {
            return subjects;
        }

        public void AddGrade(string studentId, string subject, int grade)
        {
            var student = GetStudentById(studentId);
            if (!subjects.Contains(subject))
                throw new SubjectNotFoundException($"Предмет {subject} не знайдено.");
            if (!student.Grades.ContainsKey(subject))
                student.Grades[subject] = new List<int>();
            student.Grades[subject].Add(grade);
        }

        public void EditGrade(string studentId, string subject, int index, int newGrade)
        {
            var student = GetStudentById(studentId);
            if (!student.Grades.ContainsKey(subject))
                throw new SubjectNotFoundException($"Предмет {subject} не знайдено для цього студента.");
            if (index < 0 || index >= student.Grades[subject].Count)
                throw new IndexOutOfRangeException("Невірний індекс оцінки.");
            student.Grades[subject][index] = newGrade;
        }

        public double CalculateStudentAverage(string studentId)
        {
            var student = GetStudentById(studentId);
            return student.GetAverageGrade();
        }

        public double CalculateGroupAverage(string groupName)
        {
            var group = FindGroup(groupName);
            return group.Students.Average(s => s.GetAverageGrade());
        }

        public List<Student> SearchStudentsByName(string lastName, string firstName)
        {
            return GetAllStudents().Where(s => s.LastName == lastName && s.FirstName == firstName).ToList();
        }

        public List<Student> SearchStudentsByGroup(string groupName)
        {
            var group = FindGroup(groupName);
            return group.Students;
        }

        public List<Student> SearchStudentsByAverageGrade(double minAverage, double maxAverage)
        {
            return GetAllStudents().Where(s => s.GetAverageGrade() >= minAverage && s.GetAverageGrade() <= maxAverage).ToList();
        }

        public List<Student> SearchSuccessfulStudents(double minAverage)
        {
            return GetAllStudents().Where(s => s.GetAverageGrade() >= minAverage).ToList();
        }

        public void SaveData()
        {
            SaveJsonData();
            SaveTextData();
        }

        private void SaveJsonData()
        {
            var data = new { Groups = groups, Subjects = subjects };
            File.WriteAllText(JsonDataFile, JsonConvert.SerializeObject(data, Formatting.Indented));
        }

        private void SaveTextData()
        {
            using (StreamWriter writer = new StreamWriter(TextDataFile))
            {
                writer.WriteLine("Subjects:");
                foreach (var subject in subjects)
                {
                    writer.WriteLine(subject);
                }

                writer.WriteLine("\nGroups and Students:");
                foreach (var group in groups)
                {
                    writer.WriteLine($"Group: {group.GroupName}");
                    foreach (var student in group.Students)
                    {
                        writer.WriteLine($"  Student: {student.FirstName} {student.LastName}, ID: {student.StudentId}");
                        foreach (var grade in student.Grades)
                        {
                            writer.WriteLine($"    Subject: {grade.Key}, Grades: {string.Join(", ", grade.Value)}");
                        }
                    }
                }
            }
        }

        private void LoadData()
        {
            if (File.Exists(JsonDataFile))
            {
                LoadJsonData();
            }
            else if (File.Exists(TextDataFile))
            {
                LoadTextData();
            }
        }

        private void LoadJsonData()
        {
            var data = JsonConvert.DeserializeObject<dynamic>(File.ReadAllText(JsonDataFile));
            groups = JsonConvert.DeserializeObject<List<Group>>(data.Groups.ToString());
            subjects = JsonConvert.DeserializeObject<List<string>>(data.Subjects.ToString());
        }

        private void LoadTextData()
        {
            groups.Clear();
            subjects.Clear();

            using (StreamReader reader = new StreamReader(TextDataFile))
            {
                string line;
                Group currentGroup = null;
                Student currentStudent = null;

                while ((line = reader.ReadLine()) != null)
                {
                    if (line.StartsWith("Subjects:"))
                    {
                        while ((line = reader.ReadLine()) != null && !string.IsNullOrWhiteSpace(line))
                        {
                            subjects.Add(line.Trim());
                        }
                    }
                    else if (line.StartsWith("Group:"))
                    {
                        string groupName = line.Substring(7).Trim();
                        currentGroup = new Group { GroupName = groupName };
                        groups.Add(currentGroup);
                    }
                    else if (line.Trim().StartsWith("Student:"))
                    {
                        string[] studentInfo = line.Trim().Substring(9).Split(',');
                        string[] nameParts = studentInfo[0].Trim().Split(' ');
                        string studentId = studentInfo[1].Trim().Substring(4);
                        currentStudent = new Student
                        {
                            FirstName = nameParts[0],
                            LastName = nameParts[1],
                            StudentId = studentId
                        };
                        currentGroup.Students.Add(currentStudent);
                    }
                    else if (line.Trim().StartsWith("Subject:"))
                    {
                        string[] subjectInfo = line.Trim().Substring(9).Split(',');
                        string subject = subjectInfo[0].Trim();
                        string[] grades = subjectInfo[1].Trim().Substring(8).Split(',');
                        currentStudent.Grades[subject] = grades.Select(int.Parse).ToList();
                    }
                }
            }
        }
        private Group FindGroup(string groupName)
        {
            return groups.FirstOrDefault(g => g.GroupName == groupName)
                ?? throw new GroupNotFoundException($"Групу {groupName} не знайдено.");
        }

        private Student FindStudent(Group group, string studentId)
        {
            return group.Students.FirstOrDefault(s => s.StudentId == studentId)
                ?? throw new StudentNotFoundException($"Студента з ID {studentId} не знайдено в групі {group.GroupName}.");
        }
    }
}