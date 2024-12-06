using System;

namespace GradeBookSystem.Exceptions
{
    public class DuplicateGroupException : Exception
    {
        public DuplicateGroupException(string message) : base(message) { }
    }
}