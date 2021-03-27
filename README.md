# csc311project

# 1. Introduction
One of CSC311’s main objectives is to prepare you to apply machine learning algorithms to realworld tasks. The final project aims to help you get started in this direction. You will be performing
the following tasks:
• Try out existing algorithms to real-world tasks.
• Modify an existing algorithm to improve performance.
• Write a short report analyzing the result.
The final project is not intended to be a stressful experience. It is a good chance for you to
experiment, think, play, and hopefully have fun. These tasks are what you will be doing daily as a
data analyst/scientist or machine learning engineer.

# 2. Background & Task
Online education services, such as Khan Academy and Coursera, provide a broader audience with
access to high-quality education. On these platforms, students can learn new materials by watching
a lecture, reading course material, and talking to instructors in a forum. However, one disadvantage
of the online platform is that it is challenging to measure students’ understanding of the course
material. To deal with this issue, many online education platforms include an assessment component
to ensure that students understand the core topics. The assessment component is often composed
of diagnostic questions, each a multiple choice question with one correct answer. The diagnostic
question is designed so that each of the incorrect answers highlights a common misconception.
An example of the diagnostic problem is shown in figure 1. When students incorrectly answer
the diagnostic question, it reveals the nature of their misconception and, by understanding these
misconceptions, the platform can offer additional guidance to help resolve them.

In this project, you will build machine learning algorithms to predict whether a student can correctly
answer a specific diagnostic question based on the student’s previous answers to other questions
and other students’ responses. Predicting the correctness of students’ answers to as yet unseen
diagnostic questions helps estimate the student’s ability level in a personalized education platform.
Moreover, these predictions form the groundwork for many advanced customized tasks. For instance, using the predicted correctness, the online platform can automatically recommend a set of
diagnostic questions of appropriate difficulty that fit the student’s background and learning status.
You will begin by applying existing machine learning algorithms you learned in this course. You
will then compare the performances of different algorithms and analyze their advantages and disadvantages. Next, you will modify existing algorithms to predict students’ answers with higher
accuracy. Lastly, you will experiment with your modification and write up a short report with the
results.
You will measure the performance of the learning system in terms of prediction accuracy, although
you are welcome to include other metrics in your report if you believe they provide additional
insight:
Prediction Accuracy = number of correct prediction / number of total prediction

# 3. Data
We subsampled answers of 542 students to 1774 diagnostic questions from the dataset provided by
Eedi2, an online education platform that is currently being used in many schools [1]. The platform
offers crowd-sourced mathematical diagnostic questions to students from primary to high school
(between 7 and 18 years old). The truncated dataset is provided in the folder /data.
