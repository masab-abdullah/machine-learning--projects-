import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# Step 1: Read the CSV file
data = pd.read_csv("student_data.csv")

print("Student data:")
print(data)


# Step 2: Select input columns
# These are the values used to make a prediction
X = data[["study_hours", "attendance", "previous_marks"]]


# Step 3: Select the output column
# This is what the model will predict
y = data["result"]


# Step 4: Scale the input values
# Scaling makes the different values comparable for KNN
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Step 5: Create the KNN model
# The model will check the 3 nearest students
model = KNeighborsClassifier(n_neighbors=3)


# Step 6: Train the model
model.fit(X_scaled, y)


# Step 7: Take new student data from the user
study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
previous_marks = float(input("Enter previous marks: "))


# Step 8: Put the new student data into a DataFrame
new_student = pd.DataFrame(
    [[study_hours, attendance, previous_marks]],
    columns=["study_hours", "attendance", "previous_marks"]
)


# Step 9: Scale the new student's data
new_student_scaled = scaler.transform(new_student)


# Step 10: Predict Pass or Fail
prediction = model.predict(new_student_scaled)


# Step 11: Display the result
print("\nPrediction:", prediction[0])