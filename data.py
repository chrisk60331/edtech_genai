from langchain import OpenAI
import json
from typing import List, Dict
import random

from main import Student, Curriculum, TeacherInput
import csv

# Initialize the OpenAI model
llm = OpenAI(temperature=0.0)


def save_combined_data_to_csv(data: Dict, subjects: List[str]):
    with open('combined_student_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Define the header
        headers = ['Student ID', 'Name', 'Grade']
        for subject in subjects:
            headers.append(f'{subject} Score')
            headers.append(f'{subject} Learning Objectives')
        headers.extend(['Focus Areas', 'Challenges'])

        writer.writerow(headers)

        # Write data for each student
        for student in data['students']:
            row = [student.student_id, student.name, student.grade]

            # Add test scores and curriculum objectives for each subject
            for subject in subjects:
                score = student.test_scores.get(subject, 'N/A')
                curriculum_objectives = next(
                    (c.learning_objectives for c in data['curriculums'] if
                     c.subject == subject and c.grade == student.grade), []
                )
                curriculum_str = "; ".join(curriculum_objectives)

                row.append(score)
                row.append(curriculum_str)

            # Add teacher input
            teacher_input = next(
                (ti for ti in data['teacher_inputs'] if ti.student_id == student.student_id), None
            )
            if teacher_input:
                focus_areas_str = "; ".join(teacher_input.focus_areas)
                challenges_str = "; ".join(teacher_input.challenges)
            else:
                focus_areas_str = ''
                challenges_str = ''

            row.extend([focus_areas_str, challenges_str])

            writer.writerow(row)


def generate_dynamic_data(num_students: int, subjects: List[str], grades: List[int]) -> Dict:
    generated_data = {
        "students": [],
        "curriculums": [],
        "teacher_inputs": []
    }

    for i in range(num_students):
        grade = random.choice(grades)

        # Generate student profile, test scores, and curriculum in one JSON response
        prompt = f"""
        Generate a JSON object with the following keys:
        - "student": A student profile with "name", "interests", and "description" for a student in grade {grade}.
        - "test_scores": A list of dictionaries, each with "subject" and "score" keys, representing test scores for the subjects {', '.join(subjects)}.
        - "curriculum": A dictionary where each key is a subject and the value is a list of learning objectives for that subject in grade {grade}.
        """
        response = llm(prompt)
        data = json.loads(response)

        # Extract student information
        student_name = data['student']['name']
        student_id = f"STU{str(i + 1).zfill(3)}"
        student = Student(student_id, student_name, grade)

        # Add test scores to the student
        for score in data['test_scores']:
            student.add_test_score(score['subject'], score['score'])

        generated_data["students"].append(student)

        # Add curriculum information
        for subject, objectives in data['curriculum'].items():
            generated_data["curriculums"].append(Curriculum(subject, grade, objectives))

        # Generate teacher input
        teacher_input_prompt = f"""
        Generate a JSON object with the following keys:
        - "focus_areas": A list of focus areas for the student named {student_name}.
        - "challenges": A list of challenges the student faces in their studies.
        """
        teacher_input_response = llm(teacher_input_prompt)
        teacher_input_data = json.loads(teacher_input_response)

        teacher_input = TeacherInput(
            student_id,
            teacher_input_data['focus_areas'],
            teacher_input_data['challenges']
        )
        generated_data["teacher_inputs"].append(teacher_input)

    return generated_data


# Generate a dataset with 5 students, 2 subjects, and grades ranging from 5 to 6
subjects = ["Math", "Science"]
grades = [5, 6]
dynamic_data = generate_dynamic_data(20, subjects, grades)

# Save the combined data to a single CSV file
save_combined_data_to_csv(dynamic_data, subjects)
