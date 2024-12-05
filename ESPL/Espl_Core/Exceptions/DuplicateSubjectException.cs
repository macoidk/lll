using System;

namespace GradeBookSystem.Exceptions
{
    public class DuplicateSubjectException : Exception
    {
        public DuplicateSubjectException(string message) : base(message) { }
    }
}