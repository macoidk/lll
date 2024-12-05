using System;

namespace GradeBookSystem.Exceptions
{
    public class GroupNotFoundException : Exception
    {
        public GroupNotFoundException(string message) : base(message) { }
    }
}