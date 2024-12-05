using System;

namespace GradeBookSystem.Exceptions
{
    public class SubjectNotFoundException : Exception
    {
        public SubjectNotFoundException(string message) : base(message) { }
    }
}