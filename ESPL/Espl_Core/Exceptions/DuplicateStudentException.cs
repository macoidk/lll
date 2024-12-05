using System;

namespace GradeBookSystem.Exceptions
{
    public class DuplicateStudentException : Exception
    {
        public DuplicateStudentException(string message) : base(message) { }
    }
}