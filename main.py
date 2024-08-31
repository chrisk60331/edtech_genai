import csv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
from langchain import OpenAI
from typing import List, Dict

llm = OpenAI(temperature=0.0)


# Student class to store student-related data
class Student:
    def __init__(self, student_id: str, name: str, grade: int):
        self.student_id = student_id
        self.name = name
        self.grade = grade
        self.test_scores = {}  # Dictionary to hold test scores by subject
        self.curriculum = {}  # Dictionary to hold curriculum objectives by subject
        self.focus_areas = []
        self.challenges = []

    def add_test_score(self, subject: str, score: float):
        self.test_scores[subject] = score

    def add_curriculum(self, subject: str, learning_objectives: List[str]):
        self.curriculum[subject] = learning_objectives


# Curriculum class to store curriculum standards
class Curriculum:
    def __init__(self, subject: str, grade: int, learning_objectives: List[str]):
        self.subject = subject
        self.grade = grade
        self.learning_objectives = learning_objectives


# TeacherInput class to store teacher-specific inputs for each student
class TeacherInput:
    def __init__(self, student_id: str, focus_areas: List[str], challenges: List[str]):
        self.student_id = student_id
        self.focus_areas = focus_areas
        self.challenges = challenges


# Define the prompt template for generating lesson plans
lesson_plan_prompt = PromptTemplate(
    input_variables=["student_name", "grade", "test_scores", "curriculum", "focus_areas", "challenges"],
    template="""
    You are an expert educator creating a personalized lesson plan for a student.
    The student's name is {student_name}, they are in grade {grade}.

    The student's test scores are as follows: {test_scores}.
    The curriculum objectives for this grade are: {curriculum}.

    The student needs to focus on the following areas: {focus_areas}.
    They also face the following challenges: {challenges}.

    Based on this information, generate a detailed lesson plan for the student, including specific topics to cover, recommended activities, and any additional resources.
    """
)

# Create the LangChain LLM chain
lesson_plan_chain = LLMChain(llm=llm, prompt=lesson_plan_prompt)


def generate_lesson_plan_for_student(student: Student) -> str:
    # Prepare input data for the model
    student_name = student.name
    grade = student.grade
    test_scores = json.dumps(student.test_scores)
    curriculum_str = json.dumps(student.curriculum)
    focus_areas = student.focus_areas
    challenges = student.challenges

    # Generate the lesson plan
    lesson_plan = lesson_plan_chain.run(
        student_name=student_name,
        grade=grade,
        test_scores=test_scores,
        curriculum=curriculum_str,
        focus_areas=focus_areas,
        challenges=challenges
    )

    return lesson_plan


# Example usage:
# Generate a lesson plan for the first student in the dataset
with open("combined_student_data.csv", 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    dynamic_data = [row for row in reader]

student_data = dynamic_data[0]

student_id = student_data['Student ID']
name = student_data['Name']
grade = int(student_data['Grade'])

# Create the student object
student = Student(student_id, name, grade)

# Add focus areas and challenges
student.focus_areas = student_data['Focus Areas'].split('; ')
student.challenges = student_data['Challenges'].split('; ')

# Add test scores
if 'Math Score' in student_data:
    student.add_test_score('Math', float(student_data['Math Score']))
if 'Science Score' in student_data:
    student.add_test_score('Science', float(student_data['Science Score']))

# Add curriculum (parsed from the CSV fields)
if 'Math Learning Objectives' in student_data:
    student.add_curriculum('Math', student_data['Math Learning Objectives'].split('; '))
if 'Science Learning Objectives' in student_data:
    student.add_curriculum('Science', student_data['Science Learning Objectives'].split('; '))

# Generate the lesson plan
lesson_plan = generate_lesson_plan_for_student(student)

print("Generated Lesson Plan:\n", lesson_plan)
