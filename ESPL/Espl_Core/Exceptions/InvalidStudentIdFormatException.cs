using System;

namespace GradeBookSystem.Exceptions
{
    public class InvalidStudentIdFormatException : Exception
    {
        public InvalidStudentIdFormatException(string message) : base(message) { }
    }
}