using System.Collections.Generic;

namespace GradeBookSystem.Models
{
    public class Group
    {
        public string GroupName { get; set; }
        public List<Student> Students { get; set; } = new List<Student>();
    }
}