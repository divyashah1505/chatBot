"""
Knowledge Base for College AI Assistant in Surat.
Contains comprehensive structured domain knowledge about courses, admissions, fees, 
syllabus, career prospects, facilities, contact info, and policies.
"""

COLLEGE_INFO = {
    "name": "Divya College of Higher Education",
    "location": "Surat, Gujarat",
    "address": "College Campus, University Road, Surat, Gujarat - 395007",
    "phone": "+91 261 2345678 / +91 98765 43210",
    "email": "admissions@college-surat.edu.in",
    "website": "www.college-surat.edu.in",
    "office_hours": "Monday to Saturday, 9:00 AM to 5:00 PM",
    "facilities": [
        "Modern air-conditioned computer labs with high-speed fiber internet",
        "Central digital library with 25,000+ books, journals, and e-learning portals",
        "Separate secure hostels for boys and girls with hygienic mess",
        "Sports complex (Cricket ground, Football turf, Badminton and Table Tennis)",
        "Cafeteria serving fresh and nutritious multi-cuisine food",
        "Active training and placement cell with dedicated interview preparation rooms"
    ]
}

PROGRAMS = {
    "mca": {
        "full_name": "Master of Computer Applications (MCA)",
        "level": "Postgraduate (PG)",
        "duration": "2 Years (4 Semesters)",
        "overview": (
            "MCA is an advanced 2-year postgraduate program designed to build deep expertise in software engineering, "
            "modern computing technologies, cloud architecture, and data systems. It prepares students for high-impact technical "
            "and leadership roles in the global IT industry."
        ),
        "eligibility": (
            "Passed BCA / Bachelor Degree in Computer Science / IT or equivalent degree, or passed B.Sc / B.Com / B.A "
            "with Mathematics at 10+2 level or at Graduation level. Minimum 50% aggregate marks (45% for reserved categories)."
        ),
        "fees": {
            "per_semester": "₹45,000 to ₹52,000 per semester",
            "annual": "Approx ₹90,000 - ₹1,04,000 per year",
            "details": "Installment payment options available. Government scholarships (MYSY, Digital Gujarat) applicable for eligible students."
        },
        "syllabus": [
            "Semester 1: Advanced Data Structures & Algorithms, Modern Database Systems, Object-Oriented Software Engineering, Computer Networks & Security",
            "Semester 2: Full-Stack Web Technologies (React, Node.js), Python for Data Science, Cloud Computing & DevOps, Mobile App Development",
            "Semester 3: Artificial Intelligence & Machine Learning, Big Data Analytics, Cyber Security & Ethical Hacking, Elective Specialization",
            "Semester 4: Major Industry Internship / Capstone Project and Seminar"
        ],
        "careers": [
            "Full Stack Software Developer",
            "Cloud Solutions Architect",
            "AI / Machine Learning Engineer",
            "Data Scientist / Analyst",
            "Cybersecurity Specialist",
            "Database Administrator / IT Project Manager"
        ],
        "placements": "Average package ranges between ₹4.5 LPA to ₹8.5 LPA. Top recruiters include TCS, Infosys, Wipro, Capgemini, L&T Infotech, and leading tech product companies."
    },
    
    "bca": {
        "full_name": "Bachelor of Computer Applications (BCA)",
        "level": "Undergraduate (UG)",
        "duration": "3 Years (6 Semesters)",
        "overview": (
            "BCA is a popular 3-year undergraduate course that lays a strong foundation in computer programming, web design, "
            "database management, and software development fundamentals. Perfect for students seeking an early entry into the tech industry."
        ),
        "eligibility": (
            "Passed 10+2 (Higher Secondary / HSC) in any stream (Commerce, Science, or Arts) with English and Mathematics / Statistics / Computer "
            "as one of the subjects, with a minimum of 45% aggregate marks."
        ),
        "fees": {
            "per_semester": "₹28,000 to ₹35,000 per semester",
            "annual": "Approx ₹56,000 - ₹70,000 per year",
            "details": "Scholarships and merit fee concessions available."
        },
        "syllabus": [
            "Year 1: Programming in C/C++, Web Design (HTML/CSS/JavaScript), Digital Computer Electronics, Mathematical Foundations",
            "Year 2: Object Oriented Programming in Java, Database Management Systems (SQL), Data Structures, Operating Systems",
            "Year 3: Python Programming, Software Testing, PHP/Web Frameworks, Mini Projects and Final Project"
        ],
        "careers": [
            "Junior Software Developer",
            "Frontend / Web Developer",
            "QA & Software Tester",
            "Technical Support Engineer",
            "Database Operator / Junior Analyst"
        ],
        "placements": "Average placement package is ₹2.8 LPA to ₹4.5 LPA with leading tech services firms."
    },
    
    "mba": {
        "full_name": "Master of Business Administration (MBA)",
        "level": "Postgraduate (PG)",
        "duration": "2 Years (4 Semesters)",
        "overview": (
            "MBA is a premier 2-year management degree designed to develop executive leadership, business strategy, financial acumen, "
            "and entrepreneurial abilities with dual specialization options."
        ),
        "eligibility": (
            "Bachelor's degree in any discipline from a recognized university with at least 50% marks (45% for reserved category). "
            "Valid score in entrance exams (CMAT / CAT / State CET) is preferred."
        ),
        "fees": {
            "per_semester": "₹50,000 to ₹60,000 per semester",
            "annual": "Approx ₹1,00,000 - ₹1,20,000 per year",
            "details": "Education loan assistance and scholarship schemes supported."
        },
        "syllabus": [
            "Specializations offered: Marketing Management, Financial Management, Human Resource Management (HR), Operations & Supply Chain, IT Management",
            "Includes Live Case Studies, Business Simulations, Summer Internships, and Industry Mentorship."
        ],
        "careers": [
            "Marketing & Brand Manager",
            "Financial Analyst / Investment Consultant",
            "HR Business Partner",
            "Business Development Executive",
            "Operations Manager"
        ],
        "placements": "Average package is ₹5.0 LPA to ₹9.0 LPA in banking, corporate, retail, and FMCG sectors."
    },
    
    "bba": {
        "full_name": "Bachelor of Business Administration (BBA)",
        "level": "Undergraduate (UG)",
        "duration": "3 Years (6 Semesters)",
        "overview": (
            "BBA provides comprehensive training in core business administration, financial principles, marketing strategies, "
            "and organizational management to nurture future business leaders and entrepreneurs."
        ),
        "eligibility": (
            "Passed 10+2 examination from any stream (Commerce, Science, Arts) with minimum 45% marks and English as a subject."
        ),
        "fees": {
            "per_semester": "₹26,000 to ₹32,000 per semester",
            "annual": "Approx ₹52,000 - ₹64,000 per year",
            "details": "Concessions for meritorious students."
        },
        "syllabus": [
            "Year 1: Principles of Management, Business Economics, Accounting Fundamentals, Business Communication",
            "Year 2: Marketing Management, Human Resource Management, Financial Management, Business Law",
            "Year 3: Strategic Management, Entrepreneurship Development, Elective Specialization, Project Work"
        ],
        "careers": [
            "Business Analyst",
            "Sales & Marketing Executive",
            "HR Associate",
            "Client Relationship Officer",
            "Entrepreneur / Startup Founder"
        ],
        "placements": "Average salary package ₹2.5 LPA to ₹4.0 LPA."
    }
}

ADMISSION_GUIDE = {
    "steps": [
        "1. Online / Offline Application: Fill out the application form on our website or visit the campus admission desk in Surat.",
        "2. Document Verification: Submit required academic certificates (10th, 12th, Graduation marksheets for PG, Leaving Certificate, ID proof).",
        "3. Merit Evaluation & Counseling: Seats are allocated based on academic performance and counseling rounds.",
        "4. Fee Payment & Seat Confirmation: Pay the initial semester fees to confirm your provisional admission."
    ],
    "documents": [
        "10th and 12th Standard Marksheets & Passing Certificates",
        "Graduation Marksheets & Degree Certificate (for MCA / MBA)",
        "Transfer Certificate (TC) / School Leaving Certificate (LC)",
        "Aadhar Card / Government Photo ID Proof",
        "Passport-size recent photographs (4 copies)",
        "Category / Income Certificate (if applying for government scholarships)"
    ],
    "deadlines": "Admissions for the upcoming academic session are currently open! We recommend registering early as seats are limited."
}

SCHOLARSHIPS_INFO = (
    "We offer several scholarship and fee assistance schemes:\n"
    "• Merit Scholarships: Up to 25% tuition fee waiver for meritorious students.\n"
    "• Government Schemes: Full support for MYSY (Mukhyamantri Yuva Swavalamban Yojana), Digital Gujarat SC/ST/OBC/SEBC scholarships.\n"
    "• Sibling & Sports Concessions: Special fee discounts for state/national athletes and sibling enrollments."
)
